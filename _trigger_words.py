# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Trigger-word lookup for Krea2 LoRAs
===================================
Many Krea2 LoRAs require trigger words/phrases in the prompt. This module
resolves them once per file and caches the result in a compact local store
(``data/lora_triggers.json``), independent of the AAA_Metadata_System database.

Resolution order (cheapest first; nothing here ever hashes a multi-GB file
unless every other route fails):

  1. Local DB cache, validated against the file's size+mtime signature.
  2. Civitai model-version ID parsed from the on-disk layout used by Eric's
     downloader: ``...\\{versionId}-{VersionName}\\{versionId}_{file}.safetensors``
     -> ``GET /api/v1/model-versions/{id}`` (no hashing needed at all).
  3. SHA256 embedded in the safetensors header (``modelspec.hash_sha256``,
     written by OneTrainer) -> ``GET /api/v1/model-versions/by-hash/{sha}``.
  4. Computed SHA256 of the file (last resort, cached) -> by-hash endpoint.
  5. Offline fallback: kohya/ai-toolkit ``ss_tag_frequency`` header metadata -
     the dataset tag keys, which for single-trigger style LoRAs ARE the
     trigger (e.g. ``{"1_TimBurton": {"TimBurton": 1}}`` -> "TimBurton").

Negative results (file genuinely unknown to Civitai, no usable metadata) are
cached too, so a miss never re-queries on every run. ``force=True`` bypasses
the cache and re-resolves.

Author: Eric Hiss (GitHub: EricRollei)
"""

import hashlib
import json
import os
import re
import struct
import threading
import time

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.join(_MODULE_DIR, "data")
_DB_PATH = os.path.join(_DB_DIR, "lora_triggers.json")
_CIVITAI_TIMEOUT = 10
_LOG = "[EricKrea2-Triggers]"

_lock = threading.RLock()
_db = None   # lazy-loaded process-wide cache of the JSON store


# ----------------------------------------------------------------------
#  Local store
# ----------------------------------------------------------------------

def _load_db() -> dict:
    global _db
    with _lock:
        if _db is not None:
            return _db
        try:
            if os.path.exists(_DB_PATH):
                with open(_DB_PATH, "r", encoding="utf-8") as f:
                    _db = json.load(f)
            else:
                _db = {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"{_LOG} could not read {_DB_PATH} ({e}); starting fresh")
            _db = {}
        if "loras" not in _db:
            _db = {"version": 1, "loras": {}}
        return _db


def _save_db() -> None:
    with _lock:
        try:
            os.makedirs(_DB_DIR, exist_ok=True)
            tmp = _DB_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_db, f, indent=2, ensure_ascii=False)
            os.replace(tmp, _DB_PATH)
        except OSError as e:
            print(f"{_LOG} warning: could not save trigger DB: {e}")


def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return ""


# ----------------------------------------------------------------------
#  Metadata / identity extraction
# ----------------------------------------------------------------------

def _read_st_metadata(path: str) -> dict:
    """Read the __metadata__ dict from a safetensors header (header only -
    never loads tensors). Returns {} on any problem."""
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if n <= 0 or n > 100 * 1024 * 1024:   # sanity cap
                return {}
            header = json.loads(f.read(n))
        md = header.get("__metadata__", {})
        return md if isinstance(md, dict) else {}
    except Exception:
        return {}


def _civitai_id_from_path(path: str) -> "int | None":
    """Parse the Civitai model-version id from Eric's downloader layout:
    parent dir ``{versionId}-{VersionName}`` and/or filename ``{versionId}_...``.
    Prefers agreement between the two; accepts either alone."""
    fname = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    m_dir = re.match(r"^(\d{4,10})-", parent)
    m_file = re.match(r"^(\d{4,10})_", fname)
    if m_dir and m_file:
        return int(m_dir.group(1)) if m_dir.group(1) == m_file.group(1) \
            else int(m_file.group(1))   # filename wins on disagreement
    if m_file:
        return int(m_file.group(1))
    if m_dir:
        return int(m_dir.group(1))
    return None


def _embedded_sha256(md: dict) -> "str | None":
    """OneTrainer writes modelspec.hash_sha256 = '0x<hex>'."""
    v = md.get("modelspec.hash_sha256") or md.get("hash_sha256") or ""
    v = str(v).strip().lower()
    if v.startswith("0x"):
        v = v[2:]
    return v if re.fullmatch(r"[0-9a-f]{64}", v) else None


def _compute_sha256(path: str) -> "str | None":
    """Full-file SHA256 (expensive; only used when no cheaper identity exists).
    The caller caches the result in the DB entry."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        print(f"{_LOG} SHA256 failed for {os.path.basename(path)}: {e}")
        return None


def _ss_tag_triggers(md: dict) -> list:
    """Offline heuristic: kohya/ai-toolkit ``ss_tag_frequency`` dataset-tag keys.
    For single-trigger style LoRAs the tag IS the trigger. Order: most frequent
    first. Returns [] if the field is absent/unparsable."""
    raw = md.get("ss_tag_frequency")
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        counts = {}
        for _dataset, tags in data.items():
            if not isinstance(tags, dict):
                continue
            for tag, n in tags.items():
                tag = str(tag).strip()
                if tag:
                    counts[tag] = counts.get(tag, 0) + (n if isinstance(n, (int, float)) else 1)
        return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    except Exception:
        return []


# ----------------------------------------------------------------------
#  Civitai API
# ----------------------------------------------------------------------

def _civitai_get(url: str) -> "dict | None":
    """GET a Civitai API url. Returns parsed JSON, the string 'not_found' for a
    404 (cacheable negative), or None on transient errors (NOT cached)."""
    try:
        import requests
    except ImportError:
        print(f"{_LOG} 'requests' unavailable; skipping Civitai lookup")
        return None
    try:
        r = requests.get(url, timeout=_CIVITAI_TIMEOUT,
                         headers={"User-Agent": "Eric_Krea2-trigger-lookup"})
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return "not_found"
        print(f"{_LOG} Civitai returned {r.status_code} for {url}")
        return None
    except Exception as e:
        print(f"{_LOG} Civitai request failed: {str(e)[:120]}")
        return None


def _entry_from_version_json(j: dict) -> dict:
    words = [str(w).strip() for w in (j.get("trainedWords") or []) if str(w).strip()]
    model = j.get("model") or {}
    name = " - ".join(x for x in (model.get("name"), j.get("name")) if x)
    return {
        "trigger_words": words,
        "name": name or "",
        "civitai_version_id": j.get("id"),
        "civitai_model_id": j.get("modelId"),
    }


# ----------------------------------------------------------------------
#  Public API
# ----------------------------------------------------------------------

def get_trigger_info(lora_path: str, force: bool = False) -> dict:
    """Resolve trigger words for a LoRA file. Returns the DB entry:
    {trigger_words, name, source, sha256?, civitai_version_id?, not_found?}.
    Cached across runs; ``force=True`` re-resolves (and re-queries Civitai)."""
    path = os.path.normpath(os.path.abspath(lora_path))
    sig = _file_sig(path)
    db = _load_db()
    key = path.lower()

    with _lock:
        cached = db["loras"].get(key)
    if cached and not force and cached.get("sig") == sig:
        return cached

    entry = {"sig": sig, "trigger_words": [], "name": "", "source": "none",
             "fetched_at": time.strftime("%Y-%m-%d")}
    # Carry an already-computed sha256 forward so force never re-hashes.
    if cached and cached.get("sha256") and cached.get("sig") == sig:
        entry["sha256"] = cached["sha256"]

    md = _read_st_metadata(path)

    # -- 1) Civitai by model-version id from the folder layout ---------
    vid = _civitai_id_from_path(path)
    if vid is not None:
        j = _civitai_get(f"https://civitai.com/api/v1/model-versions/{vid}")
        if isinstance(j, dict):
            info = _entry_from_version_json(j)
            if info["trigger_words"]:
                entry.update(info)
                entry["source"] = "civitai_id"
        elif j == "not_found":
            entry["civitai_version_id"] = vid
            entry["id_not_found"] = True

    # -- 2/3) Civitai by SHA256 (embedded, else computed) --------------
    if not entry["trigger_words"]:
        sha = entry.get("sha256") or _embedded_sha256(md)
        sha_src = "embedded" if sha and not entry.get("sha256") else "cached"
        if sha is None:
            sha = _compute_sha256(path)
            sha_src = "computed"
        if sha:
            entry["sha256"] = sha
            j = _civitai_get(
                f"https://civitai.com/api/v1/model-versions/by-hash/{sha}")
            if isinstance(j, dict):
                info = _entry_from_version_json(j)
                if info["trigger_words"]:
                    entry.update(info)
                    entry["source"] = f"civitai_hash_{sha_src}"
            elif j == "not_found":
                entry["hash_not_found"] = True

    # -- 4) Offline fallback: ss_tag_frequency -------------------------
    if not entry["trigger_words"]:
        tags = _ss_tag_triggers(md)
        if tags:
            entry["trigger_words"] = tags
            entry["source"] = "ss_tag_frequency"
            if not entry["name"]:
                entry["name"] = str(md.get("name") or md.get("ss_output_name") or "")

    if not entry["trigger_words"]:
        # Only a *definitive* miss (Civitai said 404, or no identity at all) is
        # cached as negative; transient network errors leave no cache entry so
        # the next run retries.
        definitive = entry.get("id_not_found") or entry.get("hash_not_found")
        if not definitive:
            print(f"{_LOG} could not resolve '{os.path.basename(path)}' "
                  "(network issue?) - will retry next run")
            return entry
        entry["not_found"] = True
        print(f"{_LOG} no trigger words found for '{os.path.basename(path)}' "
              "(cached; use force_refetch to retry)")
    else:
        print(f"{_LOG} '{os.path.basename(path)}' -> "
              f"{entry['trigger_words']} [{entry['source']}]")

    with _lock:
        db["loras"][key] = entry
    _save_db()
    return entry


def get_trigger_words(lora_path: str, force: bool = False) -> list:
    """List of trigger words/phrases for a LoRA file ([] if none found)."""
    return list(get_trigger_info(lora_path, force=force).get("trigger_words", []))


def merge_triggers_into_prompt(prompt: str, triggers: list, mode: str) -> str:
    """Combine trigger words with a prompt. ``mode``: off|prepend|append.
    Triggers already present in the prompt (case-insensitive) are not added
    twice; returns the prompt unchanged for mode 'off' or no triggers."""
    prompt = prompt or ""
    if mode == "off" or not triggers:
        return prompt
    low = prompt.lower()
    missing = [t for t in triggers if t and t.lower() not in low]
    if not missing:
        return prompt
    tail = ", ".join(missing)
    if not prompt.strip():
        return tail
    if mode == "prepend":
        return f"{tail}, {prompt}"
    return f"{prompt}, {tail}"
