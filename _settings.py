# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Shared KREA2_SETTINGS helpers
=============================
One schema across the Krea2 nodes so a node's `settings` output, a saved preset,
and a PNG's embedded recipe are all the same shape:

    { "krea2_settings_version": 1, "ultra": {...}, "loader": {...}, "lora": [ {...} ] }

`wrap(section, data)` emits one node's section (pretty JSON). `merge(*strings)`
deep-merges several `settings` strings into one for a Save Image node: dict
sections (`ultra`/`loader`) are merged, `lora` lists are concatenated.

Named-preset helpers (`load_presets`/`list_preset_names`/`apply_named_preset`)
back the per-node preset dropdowns. Presets live in `<section>_presets.json`
(package root) shaped `{ name: { <section>: {...widget values...} } }` and are
written by the server's ★ Save Preset endpoint. NOTE: a preset stores a node's
own WIDGET values (a flat dict), which is distinct from a `settings` *output*
string (where `lora` is a realized list). Different files, different consumers.
"""

import json
import os

KREA2_SETTINGS_VERSION = 1


def wrap(section: str, data) -> str:
    return json.dumps(
        {"krea2_settings_version": KREA2_SETTINGS_VERSION, section: data},
        indent=2, ensure_ascii=False,
    )


def merge(*settings_strings) -> str:
    out = {"krea2_settings_version": KREA2_SETTINGS_VERSION}
    for s in settings_strings:
        if not s or not str(s).strip():
            continue
        try:
            d = json.loads(s)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k == "krea2_settings_version":
                continue
            if k == "lora":
                out.setdefault("lora", []).extend(v if isinstance(v, list) else [v])
            elif isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            else:
                out[k] = v
    return json.dumps(out, indent=2, ensure_ascii=False)


# ── named-preset helpers (per-node preset dropdowns) ─────────────────────────

def _presets_file(section: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"{section}_presets.json")


def load_presets(section: str) -> dict:
    """Tolerant loader for `<section>_presets.json`. Returns {} on any error and
    ignores non-dict entries. Same shape the server's save endpoint writes."""
    try:
        with open(_presets_file(section), "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}
    return {k: v for k, v in d.items() if isinstance(v, dict)} if isinstance(d, dict) else {}


def list_preset_names(section: str) -> list:
    """Combo choices for a `<section>_preset` dropdown: 'custom' + saved names."""
    return ["custom"] + sorted(load_presets(section).keys())


def apply_named_preset(section: str, preset_name: str, values: dict, valid_keys=None) -> int:
    """Override `values` in-place from a saved preset's section dict (headless /
    API-run fallback for the JS that normally writes the panel). Returns the number
    of fields applied. `valid_keys`, if given, restricts which keys may be set."""
    if not preset_name or preset_name == "custom":
        return 0
    entry = load_presets(section).get(preset_name, {})
    fields = entry.get(section, entry) if isinstance(entry, dict) else {}
    if not isinstance(fields, dict):
        return 0
    n = 0
    for k, v in fields.items():
        if valid_keys is not None and k not in valid_keys:
            continue
        values[k] = v
        n += 1
    return n


# ── read a recipe back out of a saved PNG ────────────────────────────────────

def _try_load_settings(v):
    """Parse a KREA2_SETTINGS dict out of one PNG text-chunk value. Handles both a
    chunk that IS the recipe JSON and one that merely CONTAINS it (embedded in a
    larger metadata blob). Returns the dict or None."""
    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", "ignore")
        except Exception:
            return None
    if not isinstance(v, str) or "krea2_settings_version" not in v:
        return None
    try:
        d = json.loads(v)
        if isinstance(d, dict) and "krea2_settings_version" in d:
            return d
    except Exception:
        pass
    # Fallback: extract the enclosing {...} object around the marker.
    idx = v.find("krea2_settings_version")
    start = v.rfind("{", 0, idx)
    if start != -1:
        depth = 0
        for j in range(start, len(v)):
            c = v[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(v[start:j + 1])
                        if isinstance(d, dict) and "krea2_settings_version" in d:
                            return d
                    except Exception:
                        pass
                    break
    return None


def extract_from_png(path: str, key: str = "") -> dict:
    """Read the embedded KREA2_SETTINGS recipe from a PNG's text chunks. By default
    scans every tEXt/iTXt value for the ``krea2_settings_version`` marker (so it
    works no matter which chunk key the Save Image node used); pass ``key`` to read
    one specific chunk. Returns the parsed dict, or {} if none is present."""
    try:
        from PIL import Image
    except Exception:
        return {}
    try:
        img = Image.open(path)
    except Exception:
        return {}
    try:
        meta = dict(getattr(img, "text", None) or {})
    except Exception:
        meta = {}
    if not meta:
        try:
            meta = dict(img.info or {})
        except Exception:
            meta = {}
    if key:
        d = _try_load_settings(meta.get(key))
        if d:
            return d
    for val in meta.values():
        d = _try_load_settings(val)
        if d:
            return d
    return {}


def lora_list_to_slots(entries, max_slots: int = 10) -> dict:
    """Convert a settings 'lora' LIST into MultiLoRA widget slots (a flat dict:
    ``lora_i`` / ``strength_is1/is2/is3`` / ``on_i`` + ``ephemeral``) so an image's
    realized stack can be saved as a selectable MultiLoRA preset."""
    out = {}
    if not isinstance(entries, list):
        return out
    for idx, e in enumerate(entries[:max_slots], start=1):
        if not isinstance(e, dict):
            continue
        out[f"lora_{idx}"] = e.get("lora_name") or e.get("filename") or "none"
        base = e.get("strength", 1.0)
        for stage in ("s1", "s2", "s3"):
            v = e.get(f"weight_{stage}")
            if v is None:            # (0.0 is a legitimate per-stage weight)
                v = base
            try:
                out[f"strength_{idx}{stage}"] = float(v)
            except Exception:
                out[f"strength_{idx}{stage}"] = 1.0
        out[f"on_{idx}"] = True
    for e in entries:
        if isinstance(e, dict) and "ephemeral" in e:
            out["ephemeral"] = bool(e["ephemeral"])
            break
    return out
