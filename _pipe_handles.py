# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 derived pipeline handles (2026-10-08)
===============================================
Nodes that pass the KREA2 pipeline dict downstream with a modified copy (LoRA,
LoRA stack, Unload-LoRA) used to return a plain `dict(pipeline)`. ComfyUI caches
that copy as the node's output, and it holds a STRONG reference to the pipe - so on
a model switch the loader's `_free_current` (which only knew its own handle) could
not release the old weights: two models in VRAM -> OOM (seen in the native-quant
e2e queue: LoRA run, then a different checkpoint, then every later load OOM'd).

`derive_handle` returns a weakref-able dict copy and registers it (builtins-anchored
so hot-reloads don't lose the registry). `release_derived(pipes)` nulls the
'pipeline' key of every registered handle that still points at one of `pipes` -
only those, so handles of a pipeline that stays loaded are untouched.

Author: Eric Hiss (GitHub: EricRollei)
"""

import builtins
import weakref


class PipeHandle(dict):
    """dict copy of a KREA2 pipeline output that supports weakref."""


_REG = getattr(builtins, "_ERIC_KREA2_DERIVED_HANDLES", None)
if not isinstance(_REG, list):
    _REG = []
    builtins._ERIC_KREA2_DERIVED_HANDLES = _REG


def _prune():
    _REG[:] = [r for r in _REG if r() is not None]


def derive_handle(pipeline):
    h = PipeHandle(pipeline)
    _prune()
    _REG.append(weakref.ref(h))
    return h


def register_handle(h):
    """Register an existing weakref-able dict (e.g. the loader's own output handle) so
    release_derived can null it too. Every loader return path registers - ComfyUI's
    cache may keep OLDER loader outputs alive (not only the latest one)."""
    _prune()
    if not any(r() is h for r in _REG):
        _REG.append(weakref.ref(h))
    return h


def release_derived(pipes, log=None):
    ids = {id(p) for p in pipes if p is not None}
    if not ids:
        return 0
    n = 0
    for r in list(_REG):
        h = r()
        if h is not None and id(h.get("pipeline")) in ids:
            h["pipeline"] = None
            n += 1
    _prune()
    if n and log:
        log(f"released {n} downstream pipeline handle(s) held by ComfyUI's node cache (LoRA nodes)")
    return n
