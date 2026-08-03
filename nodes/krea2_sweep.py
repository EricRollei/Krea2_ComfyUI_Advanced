# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Sweep Plan / Sweep -> Preset
=======================================
Purpose-built parameter sweeping for the Ultra V2 node (spec:
docs/spec_sweep_system_2026-07-24_v2_final.md).

`EricKrea2SweepPlan` emits a KREA2_SWEEP plan (grid or JSON-lines combos) that
Ultra V2 executes inside ONE queue item when wired into its `sweep` input -
model + LoRA state stay resident, every existing pipeline feature runs
untouched per cell. The engine itself lives in `_sweep_core.py` and is
package-agnostic (the porting template for other node sets).

`EricKrea2SweepToPreset` promotes a swept cell from a sweep_manifest.json into
`ultra_presets.json` (same shape the ★ Save Preset endpoint writes), auto-tagged
with the checkpoint name so presets stay per-fine-tune.
"""

from __future__ import annotations

import csv
import json
import os

from .krea2_multistage_ultra import _SAMPLERS, _SCHEDULES, _NOISE_TYPES

_SHIFT_MODES = ["fixed", "resolution", "manual"]

_STAGES = {"s1": ("s1",), "s2": ("s2",), "s3": ("s3",), "all_stages": ("s1", "s2", "s3")}

def _sweep_preset_choices():
    """['custom'] + saved sweep preset names, refreshed whenever the GUI asks
    for the node definition (graph reload)."""
    names = ["custom"]
    try:
        from .. import _settings as _st
        names += [n for n in sorted(_st.list_preset_names("sweep")) if n != "custom"]
    except Exception:
        pass
    return names


def _default_schedulers():
    """Everything sweepable in the schedulers field right now: the built-in
    schedules plus the user's saved sigma-shape presets. Evaluated when the
    GUI asks for INPUT_TYPES, so newly saved ★ presets appear after a graph
    reload. Falls back to just the built-ins headless."""
    names = list(_SCHEDULES)
    try:
        from .. import _settings as _st
        extra = [n for n in _st.list_preset_names("sigmas") if n != "custom"]
        names += [n for n in extra if n not in names]
    except Exception:
        pass
    return ", ".join(names)


_SORT_CHOICES = ["plan_order", "sharpness_laplacian", "sharpness_tenengrad",
                 "noise_sigma", "flat_chroma_var", "clip_fraction", "time_s",
                 "uniqueness"]


def _split(s: str) -> list:
    return [t.strip() for t in str(s or "").replace("\n", ",").split(",") if t.strip()]


def _floats(s: str, name: str) -> list:
    out = []
    for t in _split(s):
        try:
            out.append(float(t))
        except ValueError:
            raise ValueError(f"SweepPlan: '{name}' entry '{t}' is not a number")
    return out


def _ints(s: str, name: str) -> list:
    return [int(round(v)) for v in _floats(s, name)]


def _validate(vals: list, allowed: list, name: str):
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(f"SweepPlan: unknown {name} {bad}; allowed: {allowed}")


class EricKrea2SweepPlan:
    """Build a KREA2_SWEEP plan. Grid axes left blank are not swept (the Ultra
    panel's live value is used). Hard-refuses over-cap grids with an axis-size
    breakdown instead of silently truncating."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["grid", "combos"], {"default": "grid"}),
                "stage_scope": (["s1", "s2", "s3", "all_stages"], {"default": "s1",
                    "tooltip": "Which stage(s) the swept keys apply to. all_stages sets the same "
                               "value on s1+s2+s3 together (NOT a per-stage cross product). "
                               "Workflow intent: perfect S1 first (quick_screen), then sweep "
                               "s2/s3 refinement on the short list."}),
            },
            "optional": {
                "samplers": ("STRING", {"multiline": True,
                    "default": ", ".join(_SAMPLERS),
                    "tooltip": f"Comma/newline-separated. Allowed: {', '.join(_SAMPLERS)}. "
                               f"Pre-populated with all of them - delete what you don't want. "
                               f"Blank = not swept. Crossed with hybrid_steps, only the hybrid "
                               f"samplers expand per split value; other samplers keep one cell "
                               f"(inert-split dedupe)."}),
                "schedulers": ("STRING", {"multiline": True,
                    "default": _default_schedulers(),
                    "tooltip": f"Built-in schedules AND saved sigma-shape ★ preset names, mixed "
                               f"freely: each entry is tried as a schedule "
                               f"({', '.join(_SCHEDULES)}) first, then as a sigmas preset "
                               f"(applied as a full profile for that cell); unknown names are "
                               f"SKIPPED with a console warning instead of aborting. "
                               f"Pre-populated with everything available - delete what you "
                               f"don't want. Blank = not swept."}),
                "pair_lock": ("BOOLEAN", {"default": False,
                    "tooltip": "ON: zip samplers+schedulers line-by-line into fixed pairs (lists must "
                               "be equal length) - for testing known recipes. OFF: full cross product."}),
                "steps": ("STRING", {"default": "",
                    "tooltip": "e.g. '8, 10, 12'. Maps to <stage>_steps."}),
                "windows": ("STRING", {"default": "",
                    "tooltip": "Refine windows 'start-end', e.g. '5-12, 6-12, 7-12'. Maps to "
                               "<stage>_start_step/<stage>_end_step. Pair with matching steps or "
                               "leave steps blank and window within the panel's step count."}),
                "etas": ("STRING", {"default": "", "tooltip": "e.g. '0, 0.3, 1.0' -> <stage>_eta."}),
                "noise_types": ("STRING", {"default": "",
                    "tooltip": f"Allowed: {', '.join(_NOISE_TYPES)}. lcm-family samplers force white "
                               "(per the 2026-07-23 rainbow fix); those cells are auto-coerced and "
                               "labeled, never silently divergent."}),
                "cfgs": ("STRING", {"default": "",
                    "tooltip": "e.g. '1.0, 1.5, 2.0' -> <stage>_cfg. Diagnostic for RAW-drifted fine "
                               "tunes: tolerance of cfg>1 tracks how far training eroded the turbo "
                               "distillation."}),
                "shift_modes": ("STRING", {"default": "",
                    "tooltip": "distilled_shift values to sweep: fixed / resolution / manual."}),
                "mus": ("STRING", {"default": "",
                    "tooltip": "shift mu values for the scoped stage(s), e.g. '1.15, 2.1, 3.0'. "
                               "Forces distilled_shift=manual on those cells (unless shift_modes is "
                               "also swept, which then wins the cross)."}),
                "combos_file": ("STRING", {"default": "",
                    "tooltip": "combos mode: path to a JSON file with the runs - either a JSON "
                               "array of override objects, {\"combos\": [...]}, or JSON-lines. "
                               "Takes precedence over combos_json below. Keeps sweep campaigns "
                               "as versionable files."}),
                "combos_json": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "combos mode: one JSON object per line using Ultra V2 widget keys "
                               "(same keys as presets / PNG recipes), e.g.\n"
                               '{"s2_sampler": "lcm_hybrid", "s2_steps": 12, "s2_start_step": 6}\n'
                               "Copy-paste directly from a preset or a stamped PNG."}),
                "seed_policy": (["fixed", "seeds_axis"], {"default": "fixed",
                    "tooltip": "fixed: every cell uses the Ultra panel seed (required for fair "
                               "comparison). seeds_axis: 'seeds' below becomes one more grid axis."}),
                "seeds": ("STRING", {"default": "", "tooltip": "seeds_axis values, e.g. '1, 2, 3'."}),
                "quick_screen": ("BOOLEAN", {"default": False,
                    "tooltip": "Force S1-only at screen_megapixels for every cell (upscale_to_stage2/3 "
                               "= 0). Cheap wide screening; re-run the short list full-pipeline after."}),
                "screen_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 8.0, "step": 0.25}),
                "score_metrics": ("BOOLEAN", {"default": True,
                    "tooltip": "Compute technical metrics per cell (CPU, ms): Laplacian + Tenengrad "
                               "sharpness, wavelet noise sigma, flat-region chroma variance "
                               "(splotch detector), clip fraction. Scores annotate + can sort the "
                               "sheet; they never hide a cell."}),
                "sort_by": (_SORT_CHOICES, {"default": "plan_order",
                    "tooltip": "Contact-sheet order. Metrics sort best-first; time_s fastest-first. "
                               "Manifest keeps plan order + this ordering separately."}),
                "tile_px": ("INT", {"default": 448, "min": 128, "max": 1024, "step": 32}),
                "max_combos": ("INT", {"default": 64, "min": 1, "max": 4096,
                    "tooltip": "Hard cap. Over-cap plans REFUSE with an axis-size breakdown "
                               "(no silent truncation)."}),
                "output_folder": ("STRING", {"default": "",
                    "tooltip": "Where sweep folders are created. Blank = ComfyUI output/sweeps. "
                               "An absolute path is used as-is; a relative path is created "
                               "under the ComfyUI output directory."}),
                "hybrid_steps": ("STRING", {"default": "",
                    "tooltip": "Sweep the hybrids' handoff in EXACT lcm steps, e.g. '3, 4, 5, 6' "
                               "-> <stage>_hybrid_steps for the scoped stage(s). Only "
                               "lcm_hybrid / lcm_hybrid2 read it - pair with a hybrid in the "
                               "samplers axis (or on the panel); inert cells are deduped. "
                               "Values clamp inside each stage's window so both halves always "
                               "run; sweeping values past window-1 collapses to the same cell."}),
                "megapixels": ("STRING", {"default": "",
                    "tooltip": "Sweep s1_megapixels, e.g. '0.5, 1.0, 2.0, 4.5'. Always targets "
                               "Stage 1 regardless of stage_scope (stage 2/3 sizes chain from it "
                               "via the upscale factors). Mixed cell sizes are fine: metrics and "
                               "the contact sheet use each cell's native output; the IMAGE batch "
                               "output center-pads smaller cells to the largest. quick_screen "
                               "does NOT override a swept megapixels value. Note: resolution "
                               "changes the noise grid, so composition shifts with size even at "
                               "a fixed seed - that is part of what you are measuring."}),
                "lora_mode": (["none", "bakeoff", "strength_ladder"], {"default": "none",
                    "tooltip": "bakeoff: each LoRA in lora_files at each lora_strengths value - "
                               "by default the swept LoRA runs ALONE (panel stack replaced) so "
                               "every style is measured against the bare checkpoint; "
                               "keep_panel_stack layers it on top instead. strength_ladder: keep "
                               "the panel-declared stack, step ONLY the lora_target entry "
                               "through lora_strengths. Sweep-injected entries are ephemeral "
                               "(fused per cell, cleaned per cell)."}),
                "lora_files": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "bakeoff: one LoRA per line - a name from the loras folder "
                               "(same resolution as the LoRA Stack node) or a full path."}),
                "lora_strengths": ("STRING", {"default": "",
                    "tooltip": "e.g. '0.6, 0.8, 1.0'. One value drives all three stage weights "
                               "(weight_s1=s2=s3). bakeoff: every file at every value; "
                               "ladder: the target entry's values."}),
                "lora_target": ("STRING", {"default": "1",
                    "tooltip": "strength_ladder: which panel-stack entry to vary - 1-based "
                               "index, or a name substring (matches lora_name/filename)."}),
                "include_baseline": ("BOOLEAN", {"default": True,
                    "tooltip": "Add one reference cell: bakeoff = no LoRA at all (or the panel "
                               "stack alone when keep_panel_stack is ON); ladder = the panel "
                               "stack exactly as declared."}),
                "keep_panel_stack": ("BOOLEAN", {"default": False,
                    "tooltip": "bakeoff only: append the swept LoRA ON TOP of the panel stack "
                               "instead of replacing it."}),
                "prompts": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "One prompt per line = a prompt axis (slowest axis, so the sheet "
                               "and CSV read in prompt-major blocks). Each prompt cell re-encodes "
                               "(prompt_conditioning is cleared; ~seconds/cell). Metric "
                               "comparisons are only meaningful WITHIN a prompt block."}),
                "sigma_profiles": ("STRING", {"default": "",
                    "tooltip": "Comma/newline list of saved sigma-shape preset NAMES (the ★ "
                               "presets of the Eric Krea2 Sigmas node). Each cell rebuilds the "
                               "full bundle from that preset and it OVERRIDES any wired sigmas "
                               "node for that cell. Blank = not swept."}),
                "lora_triggers": (["append", "prepend", "off"], {"default": "append",
                    "tooltip": "bakeoff: merge each cell's swept-LoRA trigger words into that "
                               "cell's prompt (from the LoRA Stack node's cached trigger "
                               "database; first-seen files may do one network lookup, "
                               "non-fatal on failure). Only sweep-injected entries contribute - "
                               "baseline and ladder cells are untouched. A cell whose prompt "
                               "text changes re-encodes automatically. Effective prompt + "
                               "triggers are recorded in the cell's metadata."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "Off = the node outputs nothing and Ultra V2 runs a normal single "
                               "generation - leave the sweep wired in the workflow and toggle it."}),
                "sweep_preset": (_sweep_preset_choices(), {"default": "custom",
                    "tooltip": "Named sweep plans from sweep_presets.json. Selecting one WRITES "
                               "its values into this panel (so you can see exactly what will "
                               "run); editing any field flips back to 'custom'. Save the panel "
                               "as a preset with the ★ button. Headless/API: the named preset "
                               "applies fully. Ask Claude to author experiment presets."}),
                "preset_notes": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "One or two sentences: what this sweep preset is FOR and what it "
                               "does. Saved and loaded with the ★ preset - so future-you "
                               "remembers why it exists."}),
            },
        }

    RETURN_TYPES = ("KREA2_SWEEP",)
    RETURN_NAMES = ("sweep",)
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"

    def build(self, enabled=True, sweep_preset="custom", **kwargs):
        """Wrapper: sweep on/off toggle + named sweep presets.

        enabled=False -> returns (None,): Ultra V2's sweep input treats None
        exactly like 'not connected' and runs a normal single generation, so
        the node can live in the workflow permanently.

        sweep_preset != 'custom' -> the stored plan WINS over the widgets for
        every field it contains (no JS writer exists for this node, so the
        dropdown is the whole mechanism; 'custom' = widgets rule). Unknown
        stored fields are dropped with a note; unknown preset names fall back
        to the widgets with a warning.
        """
        if not enabled:
            print("[EricKrea2-Sweep] sweep DISABLED (enabled=off) - normal generation")
            return (None,)
        # GUI: the ★ JS wrote the preset's values into the widgets on select
        # (and flips the dropdown to 'custom' on any manual edit), so when
        # widget kwargs are present the WIDGETS RULE - same doctrine as the
        # Sigmas node. Headless/API (no kwargs): the named preset is the sole
        # source and applies fully.
        if sweep_preset and sweep_preset != "custom" and not kwargs:
            pf = {}
            try:
                from .. import _settings as _st
                pf = _st.load_presets("sweep").get(sweep_preset, {})
                pf = pf.get("sweep", pf) if isinstance(pf, dict) else {}
            except Exception:
                pf = {}
            if pf:
                import inspect
                valid = set(inspect.signature(self._build).parameters) - {"self"}
                applied = {k: v for k, v in pf.items() if k in valid}
                kwargs.update(applied)
                note = str(pf.get("preset_notes", "")).strip()
                print(f"[EricKrea2-Sweep] sweep_preset '{sweep_preset}': "
                      f"applied {len(applied)} field(s) (headless)"
                      + (f' - "{note}"' if note else ""))
            else:
                print(f"[EricKrea2-Sweep] WARNING: sweep preset '{sweep_preset}' not found "
                      f"or empty - widgets rule for this run")
        return self._build(**kwargs)

    def _build(self, mode="grid", stage_scope="s1", samplers="", schedulers="", pair_lock=False,
              steps="", windows="", etas="", noise_types="", cfgs="", shift_modes="", mus="",
              combos_json="", combos_file="", seed_policy="fixed", seeds="", quick_screen=False,
              screen_megapixels=1.0, score_metrics=True, sort_by="plan_order",
              tile_px=448, max_combos=64, output_folder="", megapixels="", hybrid_steps="",
              preset_notes="",
              lora_mode="none", lora_files="", lora_strengths="", lora_target="1",
              include_baseline=True, keep_panel_stack=False, prompts="", sigma_profiles="",
              lora_triggers="append"):
        from .._sweep_core import expand_grid, diff_label
        stages = _STAGES[stage_scope]

        # v2 axes shared by both modes. Order matters: expand_grid makes the
        # FIRST axis the slowest, so prompts -> profile -> lora lead, giving
        # prompt-major (then profile-major) blocks on the sheet and CSV.
        lead_axes = {}
        pl = [ln.strip() for ln in str(prompts or "").splitlines() if ln.strip()]
        if pl:
            lead_axes["prompt"] = [{"prompt": s} for s in pl]
        sp = _split(sigma_profiles)
        if sp:
            try:
                from .. import _settings as _st
                known = set(_st.list_preset_names("sigmas"))
                known.discard("custom")
                bad = [s for s in sp if s not in known]
                if bad:
                    raise ValueError(f"SweepPlan: unknown sigma profile(s) {bad}; "
                                     f"saved: {sorted(known) or '(none)'}")
            except ImportError:
                pass
            lead_axes["sigma_profile"] = [{"sigma_profile": s} for s in sp]
        if lora_mode != "none":
            sv = _floats(lora_strengths, "lora_strengths")
            vals = []
            if include_baseline:
                vals.append({"lora_baseline": True})
            if lora_mode == "bakeoff":
                files = [ln.strip() for ln in str(lora_files or "").splitlines() if ln.strip()]
                if not files:
                    raise ValueError("SweepPlan: lora_mode=bakeoff but lora_files is empty")
                if not sv:
                    raise ValueError("SweepPlan: lora_mode=bakeoff but lora_strengths is empty")
                vals += [{"lora_file": f, "lora_strength": s} for f in files for s in sv]
            else:  # strength_ladder
                if not sv:
                    raise ValueError("SweepPlan: strength_ladder needs lora_strengths")
                tgt = str(lora_target or "1").strip()
                vals += [{"lora_strength": s, "lora_target": tgt} for s in sv]
            lead_axes["lora"] = vals

        def per_stage(key_suffix, value):
            return {f"{st}_{key_suffix}": value for st in stages}

        if mode == "combos":
            combos = []
            src_text, src_name = str(combos_json or ""), "combos_json"
            if str(combos_file or "").strip():
                path = str(combos_file).strip().strip('"')
                with open(path, "r", encoding="utf-8") as f:
                    src_text = f.read()
                src_name = os.path.basename(path)
                try:  # whole-file JSON first: [ {...}, ... ] or {"combos": [...]}
                    doc = json.loads(src_text)
                    if isinstance(doc, dict):
                        doc = doc.get("combos", [])
                    if isinstance(doc, list):
                        for i, d in enumerate(doc):
                            if not isinstance(d, dict) or not d:
                                raise ValueError(f"SweepPlan: {src_name} entry {i} is not a "
                                                 f"non-empty JSON object")
                        combos = list(doc)
                        src_text = ""  # consumed
                except ValueError:
                    raise
                except Exception:
                    pass  # fall through to JSON-lines parsing of the file text
            for ln, line in enumerate(src_text.splitlines(), 1):
                line = line.strip().rstrip(",")
                if not line or line.startswith("#"):
                    continue
                try:
                    d = json.loads(line)
                except Exception as e:
                    raise ValueError(f"SweepPlan {src_name} line {ln}: not valid JSON ({e}): {line[:120]}")
                if not isinstance(d, dict) or not d:
                    raise ValueError(f"SweepPlan combos line {ln}: expected a non-empty JSON object")
                combos.append(d)
            if not combos:
                raise ValueError(f"SweepPlan: combos mode but {src_name} has no entries")
            if lead_axes:
                cross = dict(lead_axes)
                cross["explicit"] = combos
                combos = expand_grid(cross)
        else:
            smp = _split(samplers)
            _validate(smp, _SAMPLERS, "sampler(s)")
            # Unified schedulers field: each entry is a built-in schedule OR a
            # saved sigma-shape preset name (applied as a full profile for
            # that cell). Unknown entries are skipped with a warning instead
            # of aborting - a stale preset name shouldn't kill a queued sweep.
            sch_raw = _split(schedulers)
            try:
                from .. import _settings as _st
                _prof_names = set(_st.list_preset_names("sigmas")) - {"custom"}
            except Exception:
                _prof_names = set()
            sch, skipped = [], []
            for t in sch_raw:
                if t in _SCHEDULES:
                    sch.append(("schedule", t))
                elif t in _prof_names:
                    sch.append(("profile", t))
                else:
                    skipped.append(t)
            if skipped:
                print(f"[EricKrea2-Sweep] WARNING: skipped unknown scheduler(s)/"
                      f"profile(s): {', '.join(skipped)} (not a built-in schedule "
                      f"or a saved sigmas preset)")
            def _sched_cell(kind, name):
                return (per_stage("schedule", name) if kind == "schedule"
                        else {"sigma_profile": name})
            nz = _split(noise_types)
            _validate(nz, _NOISE_TYPES, "noise_type(s)")
            shm = _split(shift_modes)
            _validate(shm, _SHIFT_MODES, "shift_mode(s)")
            axes = {}
            if pair_lock and smp and sch:
                if len(smp) != len(sch):
                    raise ValueError(f"SweepPlan: pair_lock needs equal-length lists "
                                     f"(samplers={len(smp)}, schedulers={len(sch)} "
                                     f"after skipping unknowns)")
                axes["pair"] = [dict(per_stage("sampler", a), **_sched_cell(k, b))
                                for a, (k, b) in zip(smp, sch)]
            else:
                if smp:
                    axes["sampler"] = [per_stage("sampler", v) for v in smp]
                if sch:
                    axes["scheduler"] = [_sched_cell(k, b) for k, b in sch]
            if steps.strip():
                axes["steps"] = [per_stage("steps", v) for v in _ints(steps, "steps")]
            if windows.strip():
                wvals = []
                for t in _split(windows):
                    try:
                        a, b = t.split("-")
                        wvals.append((int(a), int(b)))
                    except Exception:
                        raise ValueError(f"SweepPlan: window '{t}' must be 'start-end', e.g. '6-12'")
                axes["window"] = [dict(per_stage("start_step", a), **per_stage("end_step", b))
                                  for a, b in wvals]
            if etas.strip():
                axes["eta"] = [per_stage("eta", v) for v in _floats(etas, "etas")]
            if nz:
                axes["noise"] = [per_stage("noise", v) for v in nz]
            if cfgs.strip():
                axes["cfg"] = [per_stage("cfg", v) for v in _floats(cfgs, "cfgs")]
            if hybrid_steps.strip():
                hs = _ints(hybrid_steps, "hybrid_steps")
                bad = [v for v in hs if not (0 <= v <= 98)]
                if bad:
                    raise ValueError(f"SweepPlan: hybrid_steps out of range (0-98; "
                                     f"0 = auto half, like the Ultra widget): {bad}")
                axes["hybrid_steps"] = [per_stage("hybrid_steps", v) for v in hs]
            if megapixels.strip():
                mps = _floats(megapixels, "megapixels")
                bad = [v for v in mps if not (0.1 <= v <= 16)]
                if bad:
                    raise ValueError(f"SweepPlan: megapixels values out of range (0.1-16): {bad}")
                axes["megapixels"] = [{"s1_megapixels": v} for v in mps]
            if shm:
                axes["shift_mode"] = [{"distilled_shift": v} for v in shm]
            if mus.strip():
                mu_axis = []
                for v in _floats(mus, "mus"):
                    d = {f"shift_mu_{st}": v for st in stages}
                    if not shm:  # mu only acts under manual; force it unless modes are swept too
                        d["distilled_shift"] = "manual"
                    mu_axis.append(d)
                axes["mu"] = mu_axis
            if seed_policy == "seeds_axis":
                sv = _ints(seeds, "seeds")
                if not sv:
                    raise ValueError("SweepPlan: seed_policy=seeds_axis but 'seeds' is empty")
                axes["seed"] = [{"_seed": v} for v in sv]
            if lead_axes:
                merged = dict(lead_axes)
                merged.update(axes)
                axes = merged
            if not axes:
                raise ValueError("SweepPlan: grid mode but every axis is blank - nothing to sweep")
            n = 1
            for v in axes.values():
                n *= len(v)
            if n > max_combos:
                sizes = ", ".join(f"{k}={len(v)}" for k, v in axes.items())
                raise ValueError(f"SweepPlan: {n} combos exceeds max_combos={max_combos} "
                                 f"(axes: {sizes}). Trim an axis or raise the cap deliberately.")
            combos = expand_grid(axes)

        # lcm-family x colored-noise -> coerce to white and mark (2026-07-23 fix)
        for c in combos:
            for st in ("s1", "s2", "s3"):
                sm, nk = c.get(f"{st}_sampler", ""), f"{st}_noise"
                if str(sm).startswith("lcm") and c.get(nk) not in (None, "white"):
                    c[nk] = "white"
                    c["_coerced_noise"] = True
        # Drop cells that differ only by inert knobs (2026-07-25): euler clamps
        # eta to 0 by contract (and noise shaping only acts at eta>0), and lcm
        # ignores both (always full white re-noise). Identical effective
        # recipes would burn GPU minutes producing byte-identical results, so
        # duplicates are removed with a console note. lcm_hybrid/2 keep eta
        # (their tail sampler uses it).
        def _effective_key(c):
            # strip internal markers ("_coerced_noise" etc.) but KEEP _seed -
            # it's the seeds_axis channel and absolutely changes the image
            # (2026-07-28 fix: seed cells were collapsing as "duplicates").
            d = {k: v for k, v in c.items()
                 if not str(k).startswith("_") or k == "_seed"}
            for st in ("s1", "s2", "s3"):
                if str(d.get(f"{st}_sampler", "")) in ("euler", "lcm"):
                    d.pop(f"{st}_eta", None)
                    d.pop(f"{st}_noise", None)
            # Per-stage hybrid steps: only the hybrids read them. If a cell
            # explicitly sets that stage's sampler to a non-hybrid, that
            # stage's hybrid_steps key is inert for the cell. (No explicit
            # sampler -> the panel decides -> keep it, we can't know here.)
            for st in ("s1", "s2", "s3"):
                sm = d.get(f"{st}_sampler")
                if sm is not None and not str(sm).startswith("lcm_hybrid"):
                    d.pop(f"{st}_hybrid_steps", None)
            return json.dumps(d, sort_keys=True)
        _seen, _unique = set(), []
        for c in combos:
            k = _effective_key(c)
            if k in _seen:
                continue
            _seen.add(k)
            # also strip inert per-stage hybrid_steps from the STORED combo
            # (not just the dedupe key) so captions/manifest never show 'h7'
            # on a non-hybrid cell (cosmetic bug, Eric 2026-07-28).
            for st in ("s1", "s2", "s3"):
                sm = c.get(f"{st}_sampler")
                if sm is not None and not str(sm).startswith("lcm_hybrid"):
                    c.pop(f"{st}_hybrid_steps", None)
            _unique.append(c)
        if len(_unique) < len(combos):
            print(f"[EricKrea2-Sweep] removed {len(combos) - len(_unique)} inert duplicate "
                  f"cell(s) (eta/noise have no effect for euler and lcm)")
        combos = _unique
        if len(combos) > max_combos:
            raise ValueError(f"SweepPlan: {len(combos)} combos exceeds max_combos={max_combos}")
        plan = {"version": 2, "mode": mode, "stage_scope": stage_scope,
                "lora": ({"mode": lora_mode, "keep_panel_stack": bool(keep_panel_stack),
                          "triggers": str(lora_triggers)}
                         if lora_mode != "none" else None),
                "combos": combos, "labels": [diff_label(c) for c in combos],
                "seed_policy": seed_policy, "quick_screen": bool(quick_screen),
                "screen_megapixels": float(screen_megapixels),
                "score_metrics": bool(score_metrics), "sort_by": sort_by,
                "tile_px": int(tile_px), "output_folder": str(output_folder or "").strip()}
        print(f"[EricKrea2-Sweep] plan: {len(combos)} combo(s), scope={stage_scope}, "
              f"mode={mode}" + (", QUICK-SCREEN" if quick_screen else ""))
        return (plan,)


class EricKrea2SweepToPreset:
    """Promote a sweep cell into ultra_presets.json (the ★ preset library).
    Reads the cell's recipe from sweep_manifest.json; 'swept_only' saves a
    partial preset of just the swept keys (composable over any panel), while
    'full_recipe' snapshots the whole resolved ultra section."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": ("STRING", {"default": "", "tooltip":
                    "Wire from Ultra V2's sweep_manifest output, or paste the path."}),
                "cell_index": ("INT", {"default": 0, "min": 0, "max": 9999,
                    "tooltip": "The #index shown on the contact-sheet caption (plan order)."}),
                "preset_name": ("STRING", {"default": ""}),
            },
            "optional": {
                "save_scope": (["swept_only", "full_recipe"], {"default": "swept_only"}),
                "objective": ("STRING", {"default": "", "tooltip":
                    "Free-text tag stored with the preset, e.g. 'fast', 'quality'."}),
                "append_checkpoint": ("BOOLEAN", {"default": True, "tooltip":
                    "Append ' [checkpoint]' to the preset name - keeps presets per-fine-tune."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save"
    CATEGORY = "Eric/Krea2"
    OUTPUT_NODE = True

    def save(self, manifest_path, cell_index, preset_name, save_scope="swept_only",
             objective="", append_checkpoint=True):
        if not str(manifest_path).strip():
            raise ValueError("SweepToPreset: manifest_path is empty")
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        cell = next((c for c in man.get("cells", []) if c.get("index") == int(cell_index)), None)
        if cell is None:
            raise ValueError(f"SweepToPreset: no cell #{cell_index} in {manifest_path}")
        if "error" in cell:
            raise ValueError(f"SweepToPreset: cell #{cell_index} failed ({cell['error']}) - "
                             f"nothing to promote")
        fields = dict(cell["overrides"] if save_scope == "swept_only"
                      else cell.get("resolved", {}))
        fields.pop("seed", None)
        fields.pop("_seed", None)
        if not fields:
            raise ValueError("SweepToPreset: cell has no serializable fields")
        fields["_sweep_meta"] = {  # ignored on apply (key filter), kept for provenance
            "checkpoint": man.get("checkpoint", ""), "objective": objective,
            "sweep_id": man.get("sweep_id", ""), "cell": int(cell_index),
            "metrics": cell.get("metrics", {}), "time_s": cell.get("time_s")}
        name = str(preset_name).strip() or f"sweep_{man.get('sweep_id', 'x')}_c{cell_index}"
        if append_checkpoint and man.get("checkpoint"):
            tag = f" [{man['checkpoint']}]"
            if not name.endswith(tag):
                name += tag
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ultra_presets.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lib = json.load(f)
            if not isinstance(lib, dict):
                lib = {}
        except Exception:
            lib = {}
        lib[name] = {"ultra": fields}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(lib, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        n_real = len([k for k in fields if not k.startswith("_")])
        status = (f"saved preset '{name}' ({save_scope}, {n_real} field(s)) "
                  f"-> ultra_presets.json; dropdown refreshes after graph reload")
        print(f"[EricKrea2-Sweep] {status}")
        return (status,)



class EricKrea2SweepAutoPick:
    """Propose the best cell of a finished sweep for a given objective.

    Reads a sweep manifest and returns (cell_index, report). Wire cell_index
    into Sweep -> Preset's cell_index (right-click it -> convert widget to
    input) and the chain manifest -> Auto Pick -> Preset promotes a winner in
    one queue. The picker PROPOSES and explains; promotion still only happens
    because you queued the preset node - metrics never silently write the
    preset library.

    Objectives:
      fastest_acceptable  minimum gen time among cells passing the gates.
                          Gates self-calibrate against the sweep median
                          (noise/splotch/clip must not exceed median*(1+tol);
                          laplacian sharpness must reach median*(1-tol)), or
                          against reference_cell's own metrics when >= 0.
      most_unique         highest composition-uniqueness (written by the
                          sweep; recomputed here from the cell PNGs when an
                          older manifest lacks it).
      best_metric         best value of `metric` among built-in columns
                          (direction per metric is known).
      best_external       best value of `metric` column in scores.csv next
                          to the manifest (or external_csv), joined by
                          filename - the join point for UniPercept / blur-v7
                          / noise-v4 folder scoring. Blank metric = first
                          numeric column.
    """

    _ALIAS = {"sharp_lap": "sharpness_laplacian", "sharp_ten": "sharpness_tenengrad",
              "noise": "noise_sigma", "splotch": "flat_chroma_var",
              "clip": "clip_fraction", "clip_pct": "clip_fraction",
              "uniq": "uniqueness"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": ("STRING", {"default": "",
                    "tooltip": "Path to sweep_manifest.json (wire from Ultra V2's "
                               "sweep_manifest output, or paste)."}),
                "objective": (["fastest_acceptable", "most_unique", "most_typical",
                               "best_metric", "best_external"],
                              {"default": "fastest_acceptable"}),
            },
            "optional": {
                "metric": ("STRING", {"default": "",
                    "tooltip": "best_metric: one of sharp_lap, sharp_ten, noise, splotch, "
                               "clip, uniqueness, time_s. best_external: a scores.csv column "
                               "name (blank = first numeric column)."}),
                "tolerance": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 2.0, "step": 0.01,
                    "tooltip": "Gate slack for fastest_acceptable: quality metrics may be up to "
                               "this fraction worse than the reference (sweep median, or "
                               "reference_cell)."}),
                "reference_cell": ("INT", {"default": -1, "min": -1, "max": 9999,
                    "tooltip": "-1 = gates calibrate on the sweep median. >= 0 = gates "
                               "calibrate on this (eye-picked) cell's own metrics."}),
                "external_csv": ("STRING", {"default": "",
                    "tooltip": "best_external: CSV path. Blank = scores.csv in the sweep "
                               "folder. Joined to cells by filename."}),
                "external_higher_is_better": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("cell_index", "report")
    FUNCTION = "pick"
    CATEGORY = "EricKrea2/sweep"

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _median(vals):
        s = sorted(vals)
        m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    def _ensure_uniqueness(self, ok, sweep_dir):
        if all("uniqueness" in (r.get("metrics") or {}) for r in ok):
            return "manifest"
        try:
            import numpy as np
            from PIL import Image
            from .._sweep_core import _luma
            thumbs = {}
            for r in ok:
                fp = os.path.join(sweep_dir, r.get("file", ""))
                if not os.path.isfile(fp):
                    return "unavailable (cell files missing)"
                im = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
                h, w = im.shape[:2]
                ys = np.linspace(0, h - 1, 64).astype(int)
                xs = np.linspace(0, w - 1, 64).astype(int)
                thumbs[r["index"]] = _luma(im[ys][:, xs])
            groups = {}
            for r in ok:
                ov = r.get("overrides", {})
                groups.setdefault((str(ov.get("prompt", "")),
                                   str(ov.get("s1_megapixels", ""))), []).append(r)
            for grp in groups.values():
                for r in grp:
                    others = [o for o in grp if o["index"] != r["index"]]
                    if not others:
                        continue
                    d = float(np.mean([np.mean(np.abs(thumbs[r["index"]]
                                                      - thumbs[o["index"]]))
                                       for o in others]))
                    r.setdefault("metrics", {})["uniqueness"] = round(d * 100.0, 2)
            return "recomputed from cell PNGs"
        except Exception as e:
            return f"unavailable ({e})"

    def _gates(self, ok, tolerance, reference_cell):
        """Field-calibrated gates (Eric, 2026-07-29, from two 192-cell runs):
        clip%  - ABSOLUTE thresholds: > 6% rejected, 4-6% passes but flagged
                 suspect (his numbers - the single best bad-image predictor).
        sharp  - sharpness_tenengrad relative to the reference (median or
                 reference_cell); tenengrad proved more trustworthy than
                 laplacian in the field.
        uniq   - outlier gate: in a fixed-seed settings sweep HIGH uniqueness
                 means 'deviates from the consensus' and broken cells deviate
                 hardest; reject cells far above reference (2x tolerance
                 slack - style variation is legitimate). Skipped when absent.
        noise_sigma and flat_chroma_var carry no quality signal in the field
        (his finding) - they stay in the CSV but gate nothing."""
        with_m = [r for r in ok if r.get("metrics")]
        if not with_m:
            return ok, ["gates skipped: sweep ran with score_metrics off"]
        rel_keys = ["sharpness_tenengrad", "uniqueness"]
        if reference_cell >= 0:
            ref_rec = next((r for r in ok if r["index"] == reference_cell), None)
            if ref_rec is None or not ref_rec.get("metrics"):
                return ok, [f"gates skipped: reference_cell {reference_cell} not found/scored"]
            ref = {k: float(ref_rec["metrics"].get(k, 0.0)) for k in rel_keys}
            src = f"reference cell #{reference_cell}"
        else:
            ref = {k: self._median([float(r["metrics"].get(k, 0.0)) for r in with_m
                                    if k in r["metrics"]] or [0.0])
                   for k in rel_keys}
            src = "sweep median"
        passed, notes, suspects = [], [], []
        for r in with_m:
            fails = []
            m = r["metrics"]
            clip_pct = 100.0 * float(m.get("clip_fraction", 0.0))
            if clip_pct > 6.0:
                fails.append(f"clip {clip_pct:.1f}% > 6% (absolute)")
            elif clip_pct > 4.0:
                suspects.append(f"#{r['index']} suspect: clip {clip_pct:.1f}% (4-6% zone)")
            v = float(m.get("sharpness_tenengrad", 0.0))
            thr = ref["sharpness_tenengrad"] * (1.0 - tolerance)
            if v < thr:
                fails.append(f"sharp_ten {v:.4g} < {thr:.4g}")
            if "uniqueness" in m and ref.get("uniqueness", 0.0) > 0:
                u = float(m["uniqueness"])
                uthr = ref["uniqueness"] * (1.0 + 2.0 * tolerance)
                if u > uthr:
                    fails.append(f"uniq {u:.4g} > {uthr:.4g} (consensus outlier)")
            if fails:
                notes.append(f"#{r['index']} rejected: " + "; ".join(fails))
            else:
                passed.append(r)
        head = [f"gates vs {src}, tolerance {tolerance:.0%} "
                f"(clip absolute 4%/6%): {len(passed)}/{len(with_m)} cells pass"]
        return (passed if passed else with_m), head + notes + suspects + (
            [] if passed else ["WARNING: no cell passed - falling back to all scored cells"])

    # ---- main --------------------------------------------------------------
    def pick(self, manifest_path, objective, metric="", tolerance=0.25,
             reference_cell=-1, external_csv="", external_higher_is_better=True):
        mp = str(manifest_path or "").strip().strip('"')
        if not mp or not os.path.isfile(mp):
            raise ValueError(f"SweepAutoPick: manifest not found: '{mp}'")
        with open(mp, "r", encoding="utf-8") as f:
            man = json.load(f)
        sweep_dir = os.path.dirname(mp)
        ok = [r for r in man.get("cells", []) if "error" not in r and r.get("file")]
        if not ok:
            raise ValueError("SweepAutoPick: no successful cells in this manifest")
        lines = [f"objective: {objective}  ({len(ok)} candidate cell(s))"]
        mkey = self._ALIAS.get(metric.strip(), metric.strip())

        def mval(r, key):
            if key == "time_s":
                return r.get("time_s")
            return (r.get("metrics") or {}).get(key)

        if objective == "fastest_acceptable":
            pool, gate_notes = self._gates(ok, float(tolerance), int(reference_cell))
            lines += gate_notes
            ranked = sorted(pool, key=lambda r: (r.get("time_s", 1e9), r["index"]))
            score_of = lambda r: f"{r.get('time_s', 0):.1f}s"
        elif objective in ("most_unique", "most_typical"):
            src = self._ensure_uniqueness(ok, sweep_dir)
            lines.append(f"uniqueness source: {src}")
            pool = [r for r in ok if "uniqueness" in (r.get("metrics") or {})]
            if not pool:
                raise ValueError(f"SweepAutoPick: uniqueness {src}")
            # most_unique: max divergence (seed sweeps - variety is the point).
            # most_typical: min divergence = the consensus render (settings
            # sweeps - Eric's field finding 2026-07-29: high uniqueness there
            # flags broken outliers, the LOWEST scores looked right).
            sgn = -1 if objective == "most_unique" else 1
            ranked = sorted(pool, key=lambda r: (sgn * r["metrics"]["uniqueness"], r["index"]))
            score_of = lambda r: f"uniq {r['metrics']['uniqueness']:.2f}"
        elif objective == "best_metric":
            from .._sweep_core import METRIC_HIGHER_BETTER
            if not mkey:
                raise ValueError("SweepAutoPick: best_metric needs `metric` "
                                 "(sharp_lap, sharp_ten, noise, splotch, clip, "
                                 "uniqueness, time_s)")
            if mkey == "uniqueness":
                lines.append(f"uniqueness source: {self._ensure_uniqueness(ok, sweep_dir)}")
            pool = [r for r in ok if mval(r, mkey) is not None]
            if not pool:
                raise ValueError(f"SweepAutoPick: no cell carries metric '{mkey}'")
            hb = METRIC_HIGHER_BETTER.get(mkey, True)
            ranked = sorted(pool, key=lambda r: ((-1 if hb else 1) * float(mval(r, mkey)),
                                                 r["index"]))
            lines.append(f"metric: {mkey} ({'higher' if hb else 'lower'} is better)")
            score_of = lambda r: f"{mkey} {float(mval(r, mkey)):.2f}"
        else:  # best_external
            csv_path = (str(external_csv).strip().strip('"')
                        or os.path.join(sweep_dir, "scores.csv"))
            if not os.path.isfile(csv_path):
                raise ValueError(f"SweepAutoPick: external scores not found: '{csv_path}' "
                                 "- run your folder scorer over the sweep dir first")
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError(f"SweepAutoPick: '{csv_path}' is empty")
            fields = list(rows[0].keys())
            fname_col = next((c for c in fields
                              if c.lower() in ("file", "filename", "name", "image")),
                             fields[0])
            col = mkey
            if not col:
                for cand in fields:
                    if cand == fname_col:
                        continue
                    try:
                        float(rows[0][cand])
                        col = cand
                        break
                    except (TypeError, ValueError):
                        continue
            if not col or col not in fields:
                raise ValueError(f"SweepAutoPick: no numeric column found in '{csv_path}' "
                                 f"(have: {fields}); set `metric` to a column name")
            scores = {}
            for row in rows:
                try:
                    scores[os.path.basename(str(row[fname_col]))] = float(row[col])
                except (TypeError, ValueError):
                    continue
            pool = [r for r in ok if r.get("file") in scores]
            if not pool:
                raise ValueError(f"SweepAutoPick: no filename in '{csv_path}' matches "
                                 "this sweep's cells")
            hb = bool(external_higher_is_better)
            ranked = sorted(pool, key=lambda r: ((-1 if hb else 1) * scores[r["file"]],
                                                 r["index"]))
            lines.append(f"external: {os.path.basename(csv_path)} column '{col}' "
                         f"({'higher' if hb else 'lower'} is better), "
                         f"{len(pool)}/{len(ok)} cells matched")
            score_of = lambda r: f"{col} {scores[r['file']]:.3f}"

        from .._sweep_core import diff_label
        win = ranked[0]
        lines.append("")
        lines.append(f"PICK: cell #{win['index']}  {score_of(win)}  "
                     f"[{diff_label(win.get('overrides', {})) or 'base recipe'}]")
        lines.append(f"      file: {win.get('file', '')}")
        for r in ranked[1:4]:
            lines.append(f"  next: #{r['index']}  {score_of(r)}  "
                         f"[{diff_label(r.get('overrides', {})) or 'base recipe'}]")
        report = chr(10).join(lines)
        print("[EricKrea2-Sweep] AutoPick" + chr(10) + report)
        return (int(win["index"]), report)
NODE_CLASS_MAPPINGS = {
    "EricKrea2SweepPlan": EricKrea2SweepPlan,
    "EricKrea2SweepToPreset": EricKrea2SweepToPreset,
    "EricKrea2SweepAutoPick": EricKrea2SweepAutoPick,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EricKrea2SweepPlan": "Eric Krea2 Sweep Plan",
    "EricKrea2SweepToPreset": "Eric Krea2 Sweep → Preset",
    "EricKrea2SweepAutoPick": "Eric Krea2 Sweep Auto Pick",
}
