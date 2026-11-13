# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Sweep Checkpoints (2026-11-08)
=========================================
Pick several Krea 2 Turbo checkpoints for ONE sweep: one sweep folder, one contact sheet,
one spreadsheet. Wire its output into the Sweep Plan's `checkpoints` input.

The picker (web/js/krea2_sweep_checkpoints.js) lists every single file the Component
Loader's transformer dropdown sees under models/diffusion_models/krea2/, with check boxes.
The ticked names are stored one per line in the `checkpoints` text widget, so they save
with the workflow and can also be edited by hand. Output order = the order of the pick
list, whatever order they were ticked in.

Turbo only: every checkpoint runs on the Component Loader's base folder and settings,
with only the transformer changed. After a checkpoint sweep all models are unloaded.
"""

import os

from .krea2_component_loader import _list_models


def available_checkpoints() -> list:
    """The pick list: same source and order as the loader's transformer dropdown."""
    return [n for n in _list_models("diffusion_models", "krea2") if n != "none"]


def checkpoint_details() -> list:
    """Pick list with file size and date for the picker window (2026-11-10)."""
    out = []
    try:
        import folder_paths
    except Exception:
        folder_paths = None
    for n in available_checkpoints():
        size, mtime = None, None
        try:
            fp = folder_paths.get_full_path("diffusion_models", n) if folder_paths else None
            if fp and os.path.isfile(fp):
                st = os.stat(fp)
                size, mtime = st.st_size, st.st_mtime
        except Exception:
            pass
        out.append({"name": n, "size": size, "mtime": mtime})
    return out


def _norm(s) -> str:
    return str(s or "").replace("\\", "/").strip().lower()


def parse_checkpoints(text) -> tuple:
    """Ticked text -> (names found, in pick-list order; names not found on disk)."""
    wanted = [ln.strip() for ln in str(text or "").splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
    avail = available_checkpoints()
    by_norm = {_norm(a): a for a in avail}
    found, missing = [], []
    for w in wanted:
        a = by_norm.get(_norm(w))
        if a is None:
            missing.append(w)
        elif a not in found:
            found.append(a)
    rank = {a: i for i, a in enumerate(avail)}
    found.sort(key=lambda a: rank[a])
    return found, missing


class EricKrea2SweepCheckpoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "Off = output nothing: the sweep runs on the loader's own checkpoint as usual."}),
                "checkpoints": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The ticked checkpoints, one per line (use the 'Checkpoints' button to tick "
                               "them). Names are files under models/diffusion_models/krea2/. They run in the "
                               "order of the pick list. Turbo checkpoints only. Lines starting with # are "
                               "ignored."}),
            },
        }

    RETURN_TYPES = ("KREA2_CHECKPOINTS",)
    RETURN_NAMES = ("checkpoints",)
    FUNCTION = "pick"
    CATEGORY = "Eric/Krea2"

    def pick(self, enabled=True, checkpoints=""):
        if not enabled:
            print("[EricKrea2-Sweep] Sweep Checkpoints: off - the sweep uses the loader's checkpoint")
            return (None,)
        found, missing = parse_checkpoints(checkpoints)
        for m in missing:
            print(f"[EricKrea2-Sweep] WARNING: checkpoint '{m}' is not in models/diffusion_models/krea2 "
                  f"- skipped")
        if not found:
            print("[EricKrea2-Sweep] Sweep Checkpoints: nothing ticked - the sweep uses the loader's checkpoint")
            return (None,)
        print(f"[EricKrea2-Sweep] Sweep Checkpoints: {len(found)} checkpoint(s): " + ", ".join(found))
        return ({"names": found},)


NODE_CLASS_MAPPINGS = {"EricKrea2SweepCheckpoints": EricKrea2SweepCheckpoints}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2SweepCheckpoints": "Eric Krea2 Sweep Checkpoints"}
