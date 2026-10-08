# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 LoRA search index
============================
Backs the 🔍 LoRA search popup (web/js/krea2_lora_stack.js) with one JSON list:
every LoRA in the Krea2 dropdown, joined (by full file path) to

  * the trigger-word cache   data/lora_triggers.json (this package)
  * the LoRA Catalog cards   card_facets table of the LoRA Catalog SQLite
                             (name, role/kind, effect, artists, depicts, facet tags,
                             rating, favorite, nsfw level) - read-only

so the popup can search filenames, folders, triggers AND card text/tags.

Catalog location, first hit wins:
  1. env var ERIC_LORA_CATALOG
  2. video_prompter's config.json  ->  "lora_suggest": {"catalog": ...}
  3. L:/Models/loras/_lora_catalog.sqlite (the LoRA Catalog default)
A missing catalog or trigger cache just yields fewer fields - never an error.
The index is cached until the LoRA list, the catalog or the trigger cache change.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRIGGER_DB = os.path.join(_HERE, "data", "lora_triggers.json")
_DEFAULT_CATALOG = r"L:/Models/loras/_lora_catalog.sqlite"
_EXTS = (".safetensors", ".bin", ".pt", ".pth")
_cache = {"key": None, "payload": None}


def _key(path):
    return os.path.normcase(os.path.abspath(path)).lower()


def catalog_path():
    env = os.environ.get("ERIC_LORA_CATALOG")
    if env:
        return env
    cfg = os.path.join(os.path.dirname(_HERE), "video_prompter", "config.json")
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            sec = (json.load(f) or {}).get("lora_suggest") or {}
        if sec.get("catalog"):
            return str(sec["catalog"])
    except Exception:
        pass
    return _DEFAULT_CATALOG


def _krea2_loras():
    """[(dropdown name, full path)] - same names/order as _lora_utils.get_lora_list('krea2')."""
    import folder_paths
    out = {}
    for root_dir in folder_paths.get_folder_paths("loras"):
        if not os.path.isdir(root_dir):
            continue
        for root, _dirs, files in os.walk(root_dir):
            for fn in files:
                if fn.endswith(_EXTS):
                    full = os.path.join(root, fn)
                    out.setdefault(os.path.relpath(full, root_dir), full)
    names = sorted(out)
    k2 = [n for n in names if n.lower().startswith(("krea2\\", "krea2/"))]
    names = k2 or names                       # mirror get_lora_list's fallback
    return [(n, out[n]) for n in names]


_CIVITAI_DIR = re.compile(r"^\d+-(.+)$")


def _folder_name(rel):
    """Readable name from the Civitai layout {modelId-Model_Name}/{versionId-Version}/file:
    'SW Poetic 5 The Witness - KREA 2 · PART 1'. None when the layout doesn't apply."""
    parts = [p for p in rel.replace("\\", "/").split("/")[:-1]]
    found = []
    for p in reversed(parts):
        m = _CIVITAI_DIR.match(p)
        if not m:
            break
        found.insert(0, re.sub(r"\s+", " ", m.group(1).replace("_", " ")).strip())
        if len(found) == 2:
            break
    return " \u00b7 ".join(found) if found else None


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _load_triggers():
    try:
        with open(_TRIGGER_DB, "r", encoding="utf-8") as f:
            db = (json.load(f) or {}).get("loras") or {}
        return {k.lower(): v for k, v in db.items() if isinstance(v, dict)}
    except Exception:
        return {}


def _jl(v):
    try:
        x = json.loads(v or "[]")
        return x if isinstance(x, list) else []
    except (TypeError, ValueError):
        return []


def _load_cards(cat):
    """{full-path key: card dict} from card_facets, or ({}, note)."""
    if not os.path.exists(cat):
        if cat == _DEFAULT_CATALOG and not os.environ.get("ERIC_LORA_CATALOG"):
            # nothing configured - the catalog is an optional companion tool
            return {}, ("LoRA Catalog not installed (optional) - searching file names, "
                        "folders and trigger words")
        return {}, f"LoRA Catalog not found at {cat}"
    try:
        con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            if "card_facets" not in tables:
                return {}, "catalog has no card_facets table yet (run `run.py facets`)"
            rows = list(con.execute("select * from card_facets"))
        finally:
            con.close()
    except Exception as e:
        return {}, f"catalog unreadable ({type(e).__name__}: {e})"
    cards = {}
    for r in rows:
        cols = r.keys()
        try:
            facets = json.loads(r["facets"] or "{}")
        except ValueError:
            facets = {}
        tags = sorted({str(t).replace("_", " ") for k, v in (facets or {}).items()
                       if k != "role" and isinstance(v, list) for t in v if t})
        off = {str(t) for t in _jl(r["triggers_off"])} if "triggers_off" in cols else set()
        cards[_key(r["path"])] = {
            "name": r["name"] or "",
            "role": r["role"] or "",
            "kind": (r["kind"] or "").replace("_", " "),
            "effect": r["effect"] or "",
            "artists": [str(a) for a in _jl(r["artists"]) if a],
            "depicts": [str(d) for d in _jl(r["depicts"]) if d],
            "triggers": [str(t) for t in _jl(r["triggers"]) if t and str(t) not in off],
            "tags": tags,
            "rating": int(r["rating"]) if "rating" in cols and r["rating"] else 0,
            "fav": bool(r["favorite"]),
            "nsfw": int(r["nsfw"] or 0),
            "excl": bool(r["exclude"]),
        }
    return cards, ""


def build_index():
    loras = _krea2_loras()
    cat = catalog_path()
    key = (len(loras), loras[0][0] if loras else "", loras[-1][0] if loras else "",
           cat, _mtime(cat), _mtime(cat + "-wal"), _mtime(_TRIGGER_DB))
    if _cache["key"] == key:
        return _cache["payload"]
    cards, note = _load_cards(cat)
    trig = _load_triggers()
    entries, n_card = [], 0
    for name, full in loras:
        k = _key(full)
        card = cards.get(k)
        t = trig.get(k) or {}
        triggers = list(card["triggers"]) if card else []
        for w in t.get("trigger_words") or []:
            if w and w not in triggers:
                triggers.append(w)
        rel = name.replace("\\", "/")
        e = {"v": name,
             "n": ((card or {}).get("name") or t.get("name") or _folder_name(rel)
                   or os.path.splitext(os.path.basename(rel))[0]),
             "d": rel.rsplit("/", 1)[0] if "/" in rel else "",
             "t": triggers[:8]}
        if card:
            n_card += 1
            e.update({k2: card[k2] for k2 in ("role", "kind", "effect", "artists", "depicts",
                                              "tags", "rating", "fav", "nsfw", "excl")})
        entries.append(e)
    payload = {"entries": entries, "catalog": cat if cards else None, "carded": n_card,
               "note": note}
    _cache.update(key=key, payload=payload)
    print(f"[EricKrea2-LoRA] search index: {len(entries)} LoRAs, {n_card} with catalog cards"
          + (f" ({note})" if note else ""))
    return payload
