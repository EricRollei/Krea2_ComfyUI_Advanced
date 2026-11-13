# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
LoRA Catalog card lookup for the sweep (2026-10-12)
==================================================
Reads the LoRA Catalog's ``card_facets`` table (the same one the LoRA Suggester and the LoRA
search popup read) for two things the sweep needs per LoRA:

    card_trigger(path)              the card's FIRST active trigger (your priority triggers come
                                    first; triggers you switched off in the card editor are already
                                    left out of card_facets.triggers)
    card_weight(name_or_path, which) weight_typical / weight_low / weight_high (your editor weight
                                    wins - card_facets already carries the edit)
    card_prompt(name_or_path, ckpt) (words, source) for the `recipe` trigger mode (2026-10-15): the
                                    catalog's prompt_text / prompt_by_checkpoint (your pick per
                                    checkpoint, your pick for all, a trigger test's pick, the
                                    recipe default, the first trigger). Empty words with source
                                    'picked' / 'tested' mean "this LoRA needs no words".
    word_options(name_or_path)      the LoRA's named word options (2026-10-20, the catalog's
                                    word_options.py): [{name, text, default, source}]
    find_option(name_or_path, opt)  one of them by name (any case), number (1-based) or unique prefix
    split_option(text)              'name #option @0.8' -> ('name @0.8', 'option') - the sweep / stack
                                    syntax for choosing an option ('#each' = one cell per option)

A LoRA is found by its full path, else by its file name or file stem (so a sweep line written as a
bare name works). Two cards with the same file name: a loras-folder path picks the right one; copies whose
cards agree count as one; otherwise not found (write the full path).
The catalog location is the one _lora_search uses (ERIC_LORA_CATALOG, video_prompter's config,
L:/Models/loras/_lora_catalog.sqlite). A missing catalog just yields "no card" - never an error.
The table is cached until the catalog file (or its -wal) changes.

Author: Eric Hiss (GitHub: EricRollei)
"""

import json
import os
import re
import sqlite3
import threading

from ._lora_search import catalog_path

_LOG = "[EricKrea2-Catalog]"
_lock = threading.RLock()
_cache = {"key": None, "by_path": {}, "by_name": {}}
WEIGHT_TOKENS = {"card": "weight_typical", "card_low": "weight_low", "card_high": "weight_high"}


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _norm(path):
    return os.path.normcase(os.path.abspath(str(path))).lower()


def _json_list(text):
    try:
        value = json.loads(text or "[]")
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _load():
    """(by_path, by_name) dicts of card records, cached until the catalog changes."""
    cat = catalog_path()
    key = (cat, _mtime(cat), _mtime(str(cat) + "-wal"))
    with _lock:
        if _cache["key"] == key:
            return _cache["by_path"], _cache["by_name"]
        by_path, by_name = {}, {}
        if key[1] is not None:
            try:
                con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True, timeout=10)
                con.row_factory = sqlite3.Row
                try:
                    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
                    if "card_facets" in tables:
                        cols = {r[1] for r in con.execute("pragma table_info(card_facets)")}
                        words = [c for c in ("prompt_text", "trigger_status", "prompt_by_checkpoint",
                                             "word_options") if c in cols]
                        for r in con.execute("select path, file_name, triggers, weight_typical, weight_low, "
                                             "weight_high" + "".join(", " + c for c in words) + " from card_facets"):
                            rec = {"path": r["path"], "triggers": [str(t) for t in _json_list(r["triggers"]) if t],
                                   "weight_typical": r["weight_typical"], "weight_low": r["weight_low"],
                                   "weight_high": r["weight_high"],
                                   "prompt_text": r["prompt_text"] if "prompt_text" in words else None,
                                   "trigger_status": r["trigger_status"] if "trigger_status" in words else "",
                                   "prompt_by_checkpoint": r["prompt_by_checkpoint"]
                                   if "prompt_by_checkpoint" in words else None,
                                   "word_options": [o for o in _json_list(r["word_options"]) if isinstance(o, dict)
                                                    and o.get("name")] if "word_options" in words else []}
                            by_path[_norm(r["path"])] = rec
                            fname = str(r["file_name"] or os.path.basename(r["path"])).lower()
                            for name in (fname, os.path.splitext(fname)[0]):
                                by_name.setdefault(name, []).append(rec)
                        # the recipe variants (trigger / official / trigger+phrase / full) for the 'full' trigger
                        # mode (2026-10-16)
                        if "trigger_recipes" in tables:
                            for r in con.execute("select path, variants from trigger_recipes"):
                                rec = by_path.get(_norm(r["path"]))
                                if rec is not None:
                                    rec["variants"] = [v for v in _json_list(r["variants"]) if isinstance(v, dict)]
                    else:
                        print(f"{_LOG} catalog has no card_facets table yet (run `run.py facets`)")
                finally:
                    con.close()
            except Exception as e:
                print(f"{_LOG} catalog unreadable ({type(e).__name__}: {e}) - no card data")
        _cache.update(key=key, by_path=by_path, by_name=by_name)
        return by_path, by_name


def card_for(name_or_path):
    """The card record for a LoRA (full path, file name or stem), or None."""
    if not name_or_path:
        return None
    by_path, by_name = _load()
    rec = by_path.get(_norm(name_or_path))
    if rec is not None:
        return rec
    text = str(name_or_path).strip()
    base = os.path.basename(text).lower()
    hits = by_name.get(base) or by_name.get(os.path.splitext(base)[0]) or []
    if len(hits) > 1:
        # several cards with this file name: a loras-folder path ('Krea2\Treatment\...\x.safetensors')
        # picks the one whose path ends with it; copies whose cards agree count as one
        tail = os.path.normcase(os.path.normpath(text)).lower()
        ending = [h for h in hits if _norm(h["path"]).endswith(os.sep + tail) or _norm(h["path"]) == tail]
        if len(ending) == 1:
            return ending[0]
        same = {(tuple(h["triggers"]), h["weight_typical"], h["weight_low"], h["weight_high"]) for h in hits}
        return hits[0] if len(same) == 1 else None
    return hits[0] if hits else None


def card_trigger(name_or_path):
    """The card's first active trigger, or '' (no card, or a card without triggers)."""
    rec = card_for(name_or_path)
    return rec["triggers"][0] if rec and rec["triggers"] else ""


def card_prompt(name_or_path, checkpoint=""):
    """(words, source) for the recipe trigger mode. source: 'you' / 'test' (a pick for this checkpoint),
    picked | tested | recipe | trigger (the card-wide words), or 'none' when the catalog has nothing - the caller
    then falls back to the catalog trigger."""
    rec = card_for(name_or_path)
    if not rec or (rec.get("prompt_text") is None and not rec.get("prompt_by_checkpoint")):
        return "", "none"
    ck = str(checkpoint or "").strip()
    if ck and rec.get("prompt_by_checkpoint"):
        try:
            by_ck = json.loads(rec["prompt_by_checkpoint"]) or {}
        except (TypeError, ValueError):
            by_ck = {}
        hit = by_ck.get(ck)
        if hit is None:                                   # checkpoint names differ in case/extension only
            low = os.path.splitext(ck)[0].lower()
            hit = next((v for k, v in by_ck.items() if os.path.splitext(k)[0].lower() == low), None)
        if isinstance(hit, dict):
            return str(hit.get("text") or ""), str(hit.get("source") or "you")
    status = str(rec.get("trigger_status") or "")
    if not status:
        return "", "none"
    return str(rec.get("prompt_text") or ""), status


FULL_ORDER = ("full", "trigger+phrase", "official", "trigger")


def card_full(name_or_path):
    """(words, source) for the 'full' trigger mode (2026-10-16): the LoRA's whole recipe - official trained words +
    trigger + style phrase (the recipe's 'full' variant), else trigger + phrase, else the official words, else the
    trigger; no recipe: every active card trigger. Picks are NOT used here (that is 'saved pick').
    ('', 'none') when the catalog has nothing for it."""
    rec = card_for(name_or_path)
    if not rec:
        return "", "none"
    variants = rec.get("variants") or []
    for wanted in FULL_ORDER:
        for v in variants:
            if (v.get("name") == wanted or wanted in (v.get("covers") or [])) and str(v.get("text") or "").strip():
                return str(v["text"]).strip(), f"recipe {wanted}"
    if rec["triggers"]:
        return ", ".join(rec["triggers"]), "card triggers"
    return "", "none"


# -- word options (2026-10-20) ---------------------------------------------------------------------------------------
_OPTION = re.compile(r"\s#\s*([^@|#]*?)\s*(?=@|$)")
EACH = "each"


def split_option(text):
    """'name #option @0.8' or 'name @0.8 #option' -> ('name @0.8', 'option'); no '#option' -> (text, '')."""
    s = str(text or "")
    m = _OPTION.search(s)
    if not m:
        return s.strip(), ""
    rest = (s[:m.start()] + " " + s[m.end():]).strip()
    return re.sub(r"\s+@", " @", rest), m.group(1).strip()


def word_options(name_or_path):
    """The LoRA's word options [{name, text, default, source}], [] when it has none (or no card)."""
    rec = card_for(name_or_path)
    return list(rec.get("word_options") or []) if rec else []


def find_option(name_or_path, wanted):
    """One word option by name (any case), else 1-based number, else a unique name prefix; None when not found."""
    options = word_options(name_or_path)
    w = str(wanted or "").strip().lower()
    if not w or not options:
        return None
    for o in options:
        if str(o.get("name", "")).lower() == w:
            return o
    if w.isdigit() and 1 <= int(w) <= len(options):
        return options[int(w) - 1]
    hits = [o for o in options if str(o.get("name", "")).lower().startswith(w)]
    return hits[0] if len(hits) == 1 else None


def card_weight(name_or_path, which="card"):
    """Weight for a token 'card' | 'card_low' | 'card_high', or None when there is no card value."""
    rec = card_for(name_or_path)
    column = WEIGHT_TOKENS.get(str(which).strip().lower(), "weight_typical")
    if not rec or rec.get(column) is None:
        return None
    try:
        return round(float(rec[column]), 3)
    except (TypeError, ValueError):
        return None
