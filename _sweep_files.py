# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Sweep folder and file names (2026-10-15)
========================================
A sweep's name (its preset, or 'custom', or '<preset>-edited') goes into the folder name and the
file names, so a folder explains itself in Explorer:

    20261015_143000_trigger-test-pilot-20_loxsutopicworldkrea2_v30bf16/
        sweep_manifest_trigger-test-pilot-20.json
        sweep_summary_trigger-test-pilot-20.csv
        sweep_review_trigger-test-pilot-20.xlsx
        sweep_info_trigger-test-pilot-20.txt

Older sweeps keep their old fixed names (sweep_manifest.json ...): find() accepts both, and the
manifest's "files" entry (new sweeps) says which names that sweep uses.

Package-agnostic like _sweep_core: no imports from the rest of Eric_Krea2.
"""

import glob
import os
import re

OLD_NAMES = {"manifest": "sweep_manifest.json", "summary": "sweep_summary.csv",
             "review": "sweep_review.xlsx", "review_prev": "sweep_review.prev.xlsx",
             "info": "sweep_info.txt"}
_PARTS = {"manifest": ("sweep_manifest", ".json"), "summary": ("sweep_summary", ".csv"),
          "review": ("sweep_review", ".xlsx"), "info": ("sweep_info", ".txt")}
NAME_MAX = 40


def slug(text, limit=NAME_MAX):
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return (s[:limit].rstrip("-")) or "custom"


def sweep_name(about):
    """The sweep's short name from the plan's 'about' / meta block."""
    about = about if isinstance(about, dict) else {}
    if about.get("sweep_name"):
        return slug(about["sweep_name"])
    preset = str(about.get("sweep_preset") or "").strip()
    if preset and preset != "custom":
        return slug(preset)
    based = str(about.get("based_on") or "").strip()
    if based:
        return slug(based, NAME_MAX - 7) + "-edited"
    return "custom"


def file_names(name):
    name = slug(name)
    out = {kind: f"{stem}_{name}{ext}" for kind, (stem, ext) in _PARTS.items()}
    out["review_prev"] = f"sweep_review_{name}.prev.xlsx"
    return out


def names_for(manifest):
    """File names a sweep uses: the manifest's own "files" entry (new sweeps), else the old fixed names."""
    files = (manifest or {}).get("files") if isinstance(manifest, dict) else None
    if isinstance(files, dict) and files.get("manifest"):
        out = dict(OLD_NAMES)
        out.update(files)
        return out
    return dict(OLD_NAMES)


def find(sweep_dir, kind="manifest"):
    """Path of a sweep file in a folder (new or old naming), or None."""
    old = os.path.join(sweep_dir, OLD_NAMES[kind])
    if os.path.isfile(old):
        return old
    stem, ext = _PARTS.get(kind, ("", ""))
    if not stem:
        return None
    hits = sorted(p for p in glob.glob(os.path.join(glob.escape(sweep_dir), f"{stem}_*{ext}"))
                  if not p.endswith(".prev" + ext))
    return hits[0] if hits else None


def manifest_path(path):
    """A manifest file path, or a sweep folder -> its manifest (new or old name). Returns the input when nothing
    better is found, so callers keep their own 'not found' message."""
    p = str(path or "").strip().strip('"')
    if p and os.path.isdir(p):
        return find(p, "manifest") or os.path.join(p, OLD_NAMES["manifest"])
    return p
