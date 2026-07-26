# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Per-stage performance telemetry for the Multi-Stage Ultra nodes.

Always on, zero configuration, deliberately paranoid: every probe is wrapped so
a missing nvidia-ml-py, an NVML hiccup, or any CUDA quirk degrades to omitting
that field - never raising into the generation path.

Emits one line per denoise stage, e.g.:

  [EricKrea2-Telem] Stage 2: 10 steps in 516.2s (51.6 s/step, min 49.8 / max
      55.1), SM 2400/2617 MHz, MEM 10251/14001 MHz, 566W (cap 575W), 46C,
      cuda alloc 41.2 GB / reserved 66.0 GB

Reading the line:
  * s/step spread: uniformly slow (min ~ max, both high) points at
    clocks/bandwidth; jittery (low min, high max) points at external
    interference or host-side stalls between steps.
  * SM / MEM current-vs-max: the direct answer to "is the card downclocked".

NVML handles are matched to the torch device by GPU UUID, NOT by index:
CUDA orders devices fastest-first while NVML uses PCI bus order, so
index-matching silently reads the WRONG card on multi-GPU boxes (the exact
`nvidia-smi -i 0` trap this module was written to close).

Requires nvidia-ml-py (`python -m pip install nvidia-ml-py`, no dependencies);
without it the line still prints with timing + torch memory, just no clocks.
"""

from __future__ import annotations

import statistics
import time

import torch

_nvml = None          # pynvml module, once successfully initialised
_nvml_failed = False  # sticky: don't retry a failed init every stage
_handle_cache = {}    # torch cuda index -> NVML handle (or None)
_prev_run_end_alloc = {}  # torch cuda index -> allocated bytes at last run close (leak detector)


def _nvml_module():
    global _nvml, _nvml_failed
    if _nvml is not None or _nvml_failed:
        return _nvml
    try:
        import pynvml  # provided by the nvidia-ml-py package
        pynvml.nvmlInit()
        _nvml = pynvml
    except Exception:
        _nvml_failed = True
        _nvml = None
    return _nvml


def _norm_uuid(u):
    if isinstance(u, bytes):
        u = u.decode("utf-8", "ignore")
    return str(u).lower().removeprefix("gpu-")


def _handle_for(torch_index):
    """NVML handle for a torch cuda device index, matched by GPU UUID (same-index
    fallback only if UUIDs are unavailable). Cached; None on any failure."""
    if torch_index in _handle_cache:
        return _handle_cache[torch_index]
    handle = None
    nv = _nvml_module()
    if nv is not None:
        try:
            want = getattr(torch.cuda.get_device_properties(torch_index), "uuid", None)
            if want is not None:
                want = _norm_uuid(want)
                for i in range(nv.nvmlDeviceGetCount()):
                    h = nv.nvmlDeviceGetHandleByIndex(i)
                    if _norm_uuid(nv.nvmlDeviceGetUUID(h)) == want:
                        handle = h
                        break
            if handle is None:
                handle = nv.nvmlDeviceGetHandleByIndex(int(torch_index))
        except Exception:
            handle = None
    _handle_cache[torch_index] = handle
    return handle


def _sample(handle):
    """One NVML poll -> {sm, mem, watts, temp}; any field may be absent."""
    out = {}
    nv = _nvml
    if nv is None or handle is None:
        return out
    try:
        out["sm"] = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)
    except Exception:
        pass
    try:
        out["mem"] = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM)
    except Exception:
        pass
    try:
        out["watts"] = nv.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        pass
    try:
        out["temp"] = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
    except Exception:
        pass
    return out


def _limits(handle):
    """Static maxima -> {sm_max, mem_max, cap}; any field may be absent."""
    out = {}
    nv = _nvml
    if nv is None or handle is None:
        return out
    try:
        out["sm_max"] = nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_SM)
    except Exception:
        pass
    try:
        out["mem_max"] = nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_MEM)
    except Exception:
        pass
    try:
        out["cap"] = nv.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    except Exception:
        pass
    return out


class StageTracker:
    """begin(tag) auto-closes the previous stage; step() timestamps each solver
    step and polls NVML (sub-millisecond); close() flushes the final stage.
    Every public method is exception-proof - telemetry must never break a
    generation. Stage duration is measured begin -> last step, so inter-stage
    work (VAE upscales, decodes) is NOT counted against a stage's s/step."""

    def __init__(self, device=None):
        idx = 0
        try:
            if device is not None:
                idx = torch.device(device).index
                if idx is None:
                    idx = 0
        except Exception:
            idx = 0
        self._index = idx
        self._tag = None
        self._t0 = 0.0
        self._steps = []
        self._samples = []
        # Run-start ALLOCATED baseline for the end-of-run leak report. Taken at
        # StageTracker construction (before the stack is realized / stages run).
        self._run_alloc0 = None
        try:
            self._run_alloc0 = torch.cuda.memory_allocated(idx)
        except Exception:
            pass

    # -- public ---------------------------------------------------------

    def begin(self, tag):
        try:
            self._flush()
            self._tag = str(tag)
            self._t0 = time.perf_counter()
            self._steps = []
            self._samples = []
        except Exception:
            self._tag = None

    def step(self):
        try:
            if self._tag is None:
                return
            self._steps.append(time.perf_counter())
            s = _sample(_handle_for(self._index))
            if s:
                self._samples.append(s)
        except Exception:
            pass

    def close(self):
        try:
            self._flush()
        except Exception:
            pass
        # Run-level VRAM accounting (the leak detector, added 2026-07-21):
        #   ALLOCATED = bytes held by live tensor references. Growth from one
        #     run's end to the next run's end = a REAL leak (something retains
        #     tensors across runs) - that number is the one to watch.
        #   RESERVED = the CUDA caching allocator's pool. It ratchets up with
        #     fragmentation and is never returned to the driver, so nvidia-smi
        #     'filling up' can be this alone, which is NOT a leak. If reserved
        #     climbs while the run-over-run alloc delta stays ~0, try
        #     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
        try:
            idx = self._index
            alloc = torch.cuda.memory_allocated(idx)
            res = torch.cuda.memory_reserved(idx)
            parts = [f"end-of-run cuda alloc {alloc / 1e9:.2f} GB / reserved {res / 1e9:.2f} GB"]
            if self._run_alloc0 is not None:
                parts.append(f"alloc delta this run {(alloc - self._run_alloc0) / 1e9:+.3f} GB")
            prev = _prev_run_end_alloc.get(idx)
            if prev is not None:
                parts.append(f"vs previous run's end {(alloc - prev) / 1e9:+.3f} GB")
            _prev_run_end_alloc[idx] = alloc
            print("[EricKrea2-Telem] " + ", ".join(parts))
        except Exception:
            pass

    # -- internals ------------------------------------------------------

    def _flush(self):
        tag, t0, steps, samples = self._tag, self._t0, self._steps, self._samples
        self._tag = None
        if not tag or not steps:
            return
        total = steps[-1] - t0
        durs = [steps[0] - t0] + [b - a for a, b in zip(steps, steps[1:])]
        parts = [f"{len(steps)} steps in {total:.1f}s "
                 f"({statistics.median(durs):.1f} s/step, "
                 f"min {min(durs):.1f} / max {max(durs):.1f})"]
        lim = _limits(_handle_for(self._index))

        def _med(key):
            vals = [s[key] for s in samples if key in s]
            return statistics.median(vals) if vals else None

        sm, mem, watts, temp = _med("sm"), _med("mem"), _med("watts"), _med("temp")
        if sm is not None:
            sfx = f"/{lim['sm_max']:.0f}" if "sm_max" in lim else ""
            parts.append(f"SM {sm:.0f}{sfx} MHz")
        if mem is not None:
            sfx = f"/{lim['mem_max']:.0f}" if "mem_max" in lim else ""
            parts.append(f"MEM {mem:.0f}{sfx} MHz")
        if watts is not None:
            sfx = f" (cap {lim['cap']:.0f}W)" if "cap" in lim else ""
            parts.append(f"{watts:.0f}W{sfx}")
        if temp is not None:
            parts.append(f"{temp:.0f}C")
        try:
            alloc = torch.cuda.memory_allocated(self._index) / 1e9
            res = torch.cuda.memory_reserved(self._index) / 1e9
            parts.append(f"cuda alloc {alloc:.1f} GB / reserved {res:.1f} GB")
        except Exception:
            pass
        print(f"[EricKrea2-Telem] {tag}: " + ", ".join(parts))
