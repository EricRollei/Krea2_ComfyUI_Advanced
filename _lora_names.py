# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 LoRA name helpers (2026-10-15)
=========================================
Shared by the Sweep Plan (random / all LoRA picks), Ultra V2's sweep lookup and the Prompt LoRA Tags node.

find_lora(name)          a LoRA by full path, loras-folder path (extension optional), bare file name / stem in
                         any subfolder, or a Civitai download name with its version-id prefix and precision suffix
                         left off. Ambiguous names raise with the candidates; unknown names return None.
find_lora_folder(spec)   a folder by full path, loras-folder path, or a path under loras/Krea2 ('graphic-art').
expand_random_bakeoff(lines)   'random: graphic-art x6' / 'all: graphic-art' -> LoRA lines.
expand_random_stacks(lines)    random / all slots in stack lines -> concrete stacks.

Pick syntax (one bake-off line, or one stack slot):
    random: <source> [xN] [seed=S | seed=new]      N picks (default 1)
    all: <source>                                   every LoRA of the source, once each (2026-10-16)
  <source> is a folder (loras/Krea2 subfolder, loras-folder path or full path), or the LoRA Catalog
  (2026-10-16):
    catalog <words> [tag="..."] [kind=...] [role=...] [arch=Krea2]
  Words and tag= must all appear on the card (name, effect, facets, artists, genre/mood, triggers, folder);
  kind= / role= match the card's kind / role; arch= the shelf (default Krea2). Examples:
    random: catalog tag="art nouveau" x4
    all: catalog kind=style_medium watercolor
  * random, no seed: the same picks every run of the same text (repeatable); seed=S another repeatable draw;
    seed=new a fresh draw every queue. all: always the same list, in name order.
  * Picks are distinct within a line while the source has enough files, and never repeat a LoRA written by hand
    in the same stack line.
Picks are returned as loras-folder paths, so every cell, the manifest, sweep_info and the review workbook name the
exact file.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import sqlite3

_EXTS = (".safetensors", ".bin", ".pt", ".pth")
_by_name: dict = {}
_by_loose: dict = {}
_PICK = re.compile(r"^\s*(random|all)\s*:\s*(.+)$", re.IGNORECASE)
_VERSION_ID = re.compile(r"^\d+_")                                   # Civitai '3317450_...'
_PRECISION = re.compile(r"[_-](bf16|fp16|fp32|fp8\w*|f16|f32)$", re.IGNORECASE)


def loose_key(name: str) -> str:
    """'3317450_shouji_ink_kr2_bf16.safetensors' and 'Shouji_Ink_KR2' -> 'shouji_ink_kr2'."""
    stem = os.path.basename(str(name).replace("\\", "/"))
    if stem.lower().endswith(_EXTS):
        stem = os.path.splitext(stem)[0]
    stem = _PRECISION.sub("", _VERSION_ID.sub("", stem.strip()))
    return re.sub(r"[\s\-]+", "_", stem).lower()


def _roots():
    import folder_paths
    return [r for r in folder_paths.get_folder_paths("loras") if os.path.isdir(r)]


def _rel(path):
    """loras-folder path for a full path (falls back to the full path)."""
    full = os.path.normcase(os.path.abspath(path))
    for root in _roots():
        r = os.path.normcase(os.path.abspath(root))
        if full.startswith(r + os.sep):
            return os.path.relpath(path, root)
    return path


def _index(refresh=False):
    if _by_name and not refresh:
        return _by_name
    _by_name.clear()
    _by_loose.clear()
    for root in _roots():
        for d, _sub, files in os.walk(root):
            for f in files:
                if f.lower().endswith(_EXTS):
                    full = os.path.join(d, f)
                    for key in {f.lower(), os.path.splitext(f)[0].lower()}:
                        _by_name.setdefault(key, []).append(full)
                    _by_loose.setdefault(loose_key(f), []).append(full)
    return _by_name


def find_lora(name: str):
    """Full path of a LoRA, or None. Raises ValueError when a name matches several files."""
    name = str(name or "").strip().strip('"')
    if not name:
        return None
    if os.path.isfile(name):
        return name
    for root in _roots():
        for cand in (name,) + tuple(name + e for e in _EXTS):
            p = os.path.join(root, cand)
            if os.path.isfile(p):
                return p
    key = os.path.basename(name.replace("\\", "/")).lower()
    hits = []
    for refresh in (False, True):            # a file added since the last walk
        hits = _index(refresh).get(key, [])
        if not hits:
            # Civitai download names: version-id prefix and a precision suffix ignored (2026-10-15)
            hits = _by_loose.get(loose_key(name), [])
        if hits:
            break
    if len(hits) > 1:
        raise ValueError(f"LoRA '{name}' matches {len(hits)} files - write its folder path: "
                         + "; ".join(_rel(h) for h in hits[:4]))
    return hits[0] if hits else None


def find_lora_folder(spec: str) -> str:
    spec = str(spec or "").strip().strip('"').rstrip("\\/")
    if os.path.isabs(spec) and os.path.isdir(spec):
        return spec
    for root in _roots():
        for cand in (os.path.join(root, spec), os.path.join(root, "Krea2", spec)):
            if os.path.isdir(cand):
                return cand
    raise ValueError(f"LoRA pick: folder '{spec}' not found (a full path, a loras-folder path, "
                     f"or a folder under loras/Krea2)")


def folder_loras(folder: str) -> list:
    out = []
    for d, _sub, files in os.walk(folder):
        out += [os.path.join(d, f) for f in files if f.lower().endswith(_EXTS)]
    return sorted(out)


# -- the LoRA Catalog as a pick source (2026-10-16) -----------------------------------------------------------------
def _catalog_file():
    try:
        from ._lora_search import catalog_path
        return catalog_path()
    except Exception:
        return os.environ.get("ERIC_LORA_CATALOG") or r"L:/Models/loras/_lora_catalog.sqlite"


def catalog_loras(words, filters) -> list:
    """Full paths of cards matching every word / tag (in the card's text) and kind / role / arch filters."""
    cat = _catalog_file()
    if not os.path.exists(cat):
        raise ValueError(f"LoRA pick: the LoRA Catalog was not found ({cat})")
    con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True, timeout=10)
    try:
        rows = con.execute("select path, name, file_name, rel_dir, arch_folder, role, kind, effect, facets, "
                           "artists, triggers, creator from card_facets").fetchall()
    finally:
        con.close()
    arch = str(filters.get("arch", "Krea2")).lower()
    kind = str(filters.get("kind", "")).lower()
    role = str(filters.get("role", "")).lower()
    needles = [w.lower() for w in words] + [t.lower() for t in filters.get("tags", [])]
    # creator= (2026-10-26): the card's creator / author, matched loosely (Adel_Ai = adel ai = Adel-AI)
    want_creator = str(filters.get("creator", "")).lower().replace("_", " ").replace("-", " ").strip()
    out = []
    for path, name, fname, rel_dir, arch_folder, crole, ckind, effect, facets, artists, triggers, ccreator in rows:
        if arch and arch != "any" and str(arch_folder or "").lower() != arch:
            continue
        if kind and kind not in str(ckind or "").lower():
            continue
        if role and role != str(crole or "").lower():
            continue
        if want_creator and want_creator not in str(ccreator or "").lower().replace("_", " ").replace("-", " "):
            continue
        # the creator is part of the searchable text too (2026-10-26) - the catalog's search box finds it
        blob = " ".join(str(x or "") for x in (name, fname, rel_dir, effect, facets, artists, triggers,
                                               ccreator)).lower()
        blob = blob.replace("_", " ").replace("-", " ")
        if all(n.replace("_", " ").replace("-", " ") in blob for n in needles) and os.path.isfile(path):
            out.append(path)
    return sorted(out)


def parse_pick(text: str):
    """'random: graphic-art x6 seed=3' / 'all: catalog tag="art nouveau"' ->
    {mode, source, n, seed, label} or None when the text is not a pick."""
    m = _PICK.match(str(text or ""))
    if not m:
        return None
    mode = m.group(1).lower()
    try:
        tokens = shlex.split(m.group(2).replace("\\", "/"))
    except ValueError:
        tokens = m.group(2).split()
    per_prompt = False                   # 'per prompt' (2026-10-26): one draw per prompt line, paired
    for j in range(len(tokens) - 1):
        if tokens[j].lower() == "per" and tokens[j + 1].lower() in ("prompt", "prompts"):
            per_prompt, tokens = True, tokens[:j] + tokens[j + 2:]
            break
    if any(t.lower() in ("per_prompt", "per-prompt") for t in tokens):
        per_prompt = True
        tokens = [t for t in tokens if t.lower() not in ("per_prompt", "per-prompt")]
    n, seed, words, filters = 1, "", [], {"tags": []}
    for tok in tokens:
        low = tok.lower()
        if re.fullmatch(r"x\d+", low):
            n = max(1, int(low[1:]))
        elif low.startswith("seed="):
            seed = tok[5:]
        elif "=" in tok and tok.split("=", 1)[0].lower() in ("tag", "kind", "role", "arch", "creator", "author"):
            key, val = tok.split("=", 1)
            if key.lower() == "tag":
                filters["tags"].append(val)
            elif key.lower() in ("creator", "author"):           # 2026-10-26
                filters["creator"] = val
            else:
                filters[key.lower()] = val
        else:
            words.append(tok)
    if words and words[0].lower() == "catalog":
        source = {"type": "catalog", "words": words[1:], "filters": filters}
        label = "catalog " + " ".join(words[1:] + [f'tag="{t}"' for t in filters["tags"]]
                                       + [f"{k}={v}" for k, v in filters.items() if k != "tags"])
    else:
        if not words:
            raise ValueError(f"LoRA pick: no folder in '{text.strip()}' (write 'random: graphic-art x6')")
        source = {"type": "folder", "spec": " ".join(words)}
        label = source["spec"]
    return {"mode": mode, "source": source, "n": n, "seed": seed, "label": label.strip(), "per_prompt": per_prompt}


# old name, kept for callers
def parse_random(text: str):
    p = parse_pick(text)
    if p is None or p["mode"] != "random" or p["source"]["type"] != "folder":
        return None if p is None else (p["source"].get("spec", ""), p["n"], p["seed"])
    return p["source"]["spec"], p["n"], p["seed"]


def pick_pool(pick) -> tuple:
    """(rng key, sorted full paths) for a parsed pick."""
    src = pick["source"]
    if src["type"] == "catalog":
        pool = catalog_loras(src["words"], src["filters"])
        if not pool:
            raise ValueError(f"LoRA pick: no catalog card matches '{pick['label']}'")
        return pick["label"].lower(), pool
    folder = find_lora_folder(src["spec"])
    pool = folder_loras(folder)
    if not pool:
        raise ValueError(f"LoRA pick: no LoRA files in {folder}")
    return os.path.normcase(folder), pool


def _rng(key, where, seed):
    if str(seed).lower() == "new":
        return random.Random()
    return random.Random(f"{key}|{where}|{seed}")


def _draw(pool, n, rng, avoid=()):
    """n picks, distinct while possible, never one of `avoid` if anything else is left."""
    avoid = {os.path.normcase(a) for a in avoid}
    usable = [p for p in pool if os.path.normcase(p) not in avoid] or list(pool)
    out = []
    while len(out) < n:
        batch = list(usable)
        rng.shuffle(batch)
        out += batch[:n - len(out)]
    return out


def is_fresh_draw(text: str) -> bool:
    """True when the text asks for a new draw every run (seed=new)."""
    return "seed=new" in re.sub(r"\s+", "", str(text or "")).lower()


def clean_lines(text: str) -> list:
    return [ln.strip() for ln in str(text or "").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def expand_random_bakeoff(lines: list, notes: list) -> list:
    """Bakeoff lines with every 'random:' / 'all:' line replaced by its LoRAs; duplicates dropped."""
    out, seen = [], set()
    for i, ln in enumerate(lines):
        pick = parse_pick(ln)
        if pick is None:
            picks = [ln]
        else:
            key, pool = pick_pool(pick)
            if pick["mode"] == "all":
                chosen = list(pool)
            else:
                chosen = _draw(pool, min(pick["n"], len(pool)), _rng(key, i, pick["seed"]))
            picks = [_rel(p) for p in chosen]
            notes.append(f"'{ln}' -> {len(picks)}: " + ", ".join(os.path.basename(p) for p in picks)
                         + (f" (only {len(pool)} available)" if pick["mode"] == "random" and pick["n"] > len(pool)
                            else ""))
        for p in picks:
            if p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
    return out


_PER_PROMPT = re.compile(r"\bper[\s_-]+prompts?\b", re.IGNORECASE)


def is_per_prompt(line: str) -> bool:
    return bool(_PER_PROMPT.search(str(line or "")))


def per_prompt_picks(lines: list, n_prompts: int, notes: list):
    """'random: <source> [xN] per prompt [#random]' (2026-10-26): every prompt line gets its OWN draw of N LoRAs,
    paired with that prompt instead of crossed with all of them. Returns [[picks for prompt 1], ...] (loras-folder
    names; '#random' kept on each so the plan draws a word option per pick), or None when no line asks for it."""
    if not any(is_per_prompt(ln) for ln in lines):
        return None
    if len(lines) != 1:
        raise ValueError("LoRA pick: a 'per prompt' line must be the only line in lora_files")
    ln, opt = lines[0], ""
    m = re.search(r"\s#\s*random\b", ln, re.IGNORECASE)
    if m:
        opt, ln = " #random", ln[:m.start()] + ln[m.end():]
    pick = parse_pick(ln)
    if pick is None or pick["mode"] != "random":
        raise ValueError("LoRA pick: 'per prompt' works with a 'random:' line, e.g. "
                         "'random: catalog tag=\"art nouveau\" per prompt'")
    if n_prompts < 1:
        raise ValueError("LoRA pick: 'per prompt' needs prompts - one per line in the prompts field")
    key, pool = pick_pool(pick)
    total = pick["n"] * n_prompts
    draws = _draw(pool, total, _rng(key, 0, pick["seed"]))
    groups = []
    for k in range(n_prompts):
        g = [_rel(p) + opt for p in draws[k * pick["n"]:(k + 1) * pick["n"]]]
        groups.append(g)
        notes.append(f"'{lines[0]}' prompt {k + 1} -> " + ", ".join(os.path.basename(p) for p in
                                                                    draws[k * pick["n"]:(k + 1) * pick["n"]]))
    if total > len(pool):
        notes.append(f"(only {len(pool)} LoRAs match '{pick['label']}' - some prompts share a LoRA)")
    return groups


def per_prompt_stacks(lines: list, n_prompts: int, notes: list):
    """Stacks mode, 'random: <source> xN per prompt [@w] [#random] [| fixed @w ...]' (2026-10-26): every prompt
    line gets ONE stack - N fresh picks from the source (distinct within the stack, never a LoRA written by hand in
    the line) plus the line's fixed LoRAs. Several 'per prompt' slots in the line each add their N. Returns
    [stack line for prompt 1, ...] or None when no line asks for it."""
    if not any(is_per_prompt(ln) for ln in lines):
        return None
    if len(lines) != 1:
        raise ValueError("LoRA pick: a 'per prompt' line must be the only line in lora_files")
    if n_prompts < 1:
        raise ValueError("LoRA pick: 'per prompt' needs prompts - one per line in the prompts field")
    ln = lines[0]
    slots = []
    for part in [p.strip() for p in ln.split("|") if p.strip()]:
        opt = ""
        m = re.search(r"\s#\s*random\b", part, re.IGNORECASE)
        if m:
            opt, part = " #random", (part[:m.start()] + part[m.end():]).strip()
        name, weight = part, None
        if "@" in part:
            name, weight = part.rsplit("@", 1)
            name, weight = name.strip(), weight.strip()
        pick = parse_pick(name)
        # one 'per prompt' anywhere in the line covers every random slot in it (2026-10-26)
        if pick is not None and pick["mode"] != "random":
            raise ValueError("LoRA pick: a 'per prompt' stack line takes 'random:' slots only (not 'all:')")
        slots.append((part + opt, name, weight, pick, opt))
    fixed = [find_lora(s[1]) or s[1] for s in slots if s[3] is None]
    fixed_n = {os.path.normcase(f) for f in fixed}
    draws = {}
    for j, (_part, _name, _w, pick, _opt) in enumerate(slots):
        if pick is None:
            continue
        key, pool = pick_pool(pick)
        usable = [p for p in pool if os.path.normcase(p) not in fixed_n] or list(pool)
        draws[j] = (_draw(usable, pick["n"] * n_prompts, _rng(key, f"pp.{j}", pick["seed"])), usable)
        if pick["n"] * n_prompts > len(usable):
            notes.append(f"(only {len(usable)} LoRAs match '{pick['label']}' - some prompts share a LoRA)")
    out = []
    for k in range(n_prompts):
        used, built, names = set(fixed_n), [], []
        for j, (part, _name, weight, pick, opt) in enumerate(slots):
            if pick is None:
                built.append(part)
                continue
            seq, usable = draws[j]
            n = pick["n"]
            for p in seq[k * n:(k + 1) * n]:
                if os.path.normcase(p) in used:              # never the same file twice in one stack
                    alt = [q for q in usable if os.path.normcase(q) not in used]
                    p = alt[0] if alt else p
                used.add(os.path.normcase(p))
                names.append(os.path.basename(p))
                # no weight written: the plan gives each pick its card weight (as in other stacks)
                built.append((f"{_rel(p)} @{weight}" if weight else _rel(p)) + opt)
        out.append(" | ".join(built))
        notes.append(f"'{ln}' prompt {k + 1} -> stack: " + " + ".join(names))
    return out


_PARTNER = re.compile(r"^\s*partner\s*:\s*(.*)$", re.IGNORECASE)


def _partner_candidates(lead_path: str) -> list:
    """[(path, score, source)] best first for a lead LoRA (2026-10-27): 'your stacks' = positive synergy from the
    LoRA Catalog (markmodel lora_synergy), then 'Civitai' = what other users stack it with (lora_pairs, owned)."""
    cat = _catalog_file()
    if not cat:
        return []
    out, seen = [], set()
    try:
        con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return []
    try:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        if "loras" not in tables:
            return []
        cols = {r[1] for r in con.execute("pragma table_info(loras)")}
        civ = "civitai_version_id" in cols
        row = con.execute(f"select sha256{', civitai_version_id' if civ else ''} from loras "
                          "where lower(path) = lower(?)", (lead_path,)).fetchone()
        if not row:
            return []
        sha, vid = (row[0] or "").lower(), (row[1] if civ else None)

        def add(path, score, source):
            if path and os.path.isfile(path) and os.path.normcase(path) not in seen \
                    and os.path.normcase(path) != os.path.normcase(lead_path):
                seen.add(os.path.normcase(path))
                out.append((path, float(score), source))

        if sha and "lora_synergy" in tables:
            for a, b, syn in con.execute("select a, b, synergy from lora_synergy where (a = ? or b = ?) and synergy > 0 "
                                         "order by synergy desc", (sha, sha)).fetchall():
                other = b if a == sha else a
                pr = con.execute("select path from loras where lower(sha256) = ?", (other,)).fetchone()
                add(pr[0] if pr else None, syn, "your stacks")
        if civ and vid and "lora_pairs" in tables:
            vk = f"v:{int(vid)}"
            for a, b, score in con.execute("select a, b, score from lora_pairs where (a = ? or b = ?) and score > 0 "
                                           "order by score desc limit 30", (vk, vk)).fetchall():
                other = b if a == vk else a
                if other.startswith("v:"):
                    pr = con.execute("select path from loras where civitai_version_id = ?", (int(other[2:]),)).fetchone()
                    add(pr[0] if pr else None, score, "Civitai")
    except sqlite3.Error:
        return out
    finally:
        con.close()
    return out


def resolve_partners(lines: list, notes: list, salt=0) -> list:
    """Stack lines with every 'partner: [best|random] [seed=S] [@w]' slot filled in (2026-10-27): a partner for the
    line's FIRST LoRA - your best-rated stack partner (LoRA Catalog synergy), else what Civitai users stack it with.
    best (default) = the top one; random = any of your positive partners (else one of the top 5 from Civitai).
    Never a LoRA already in the line. No partner known = the slot is dropped, with a note."""
    out = []
    for i, ln in enumerate(lines):
        parts = [p.strip() for p in str(ln).split("|") if p.strip()]
        if not any(_PARTNER.match(p) for p in parts):
            out.append(ln)
            continue
        fixed = [p for p in parts if not _PARTNER.match(p)]
        used, lead = set(), None
        for p in fixed:
            nm = p.split("#", 1)[0]
            nm = nm.rsplit("@", 1)[0] if "@" in nm else nm
            f = find_lora(nm.strip())
            if f:
                used.add(os.path.normcase(f))
                lead = lead or f
        built = list(fixed)
        for p in parts:
            m = _PARTNER.match(p)
            if not m:
                continue
            spec, weight = m.group(1).strip(), None
            if "@" in spec:
                spec, weight = spec.rsplit("@", 1)
                spec, weight = spec.strip(), weight.strip()
            mode, seed = "best", ""
            for t in spec.split():
                if t.lower() in ("best", "random"):
                    mode = t.lower()
                elif t.lower().startswith("seed="):
                    seed = t.split("=", 1)[1]
            if not lead:
                notes.append(f"'{ln}': partner: needs a LoRA earlier in the line to pair with - slot dropped")
                continue
            cands = [c for c in _partner_candidates(lead) if os.path.normcase(c[0]) not in used]
            if not cands:
                notes.append(f"partner for {os.path.basename(lead)}: none known yet (no rated stacks with it, no "
                             f"Civitai harvest) - slot dropped")
                continue
            if mode == "random":
                pool = [c for c in cands if c[2] == "your stacks"] or cands[:5]
                c = _rng(os.path.normcase(lead), f"partner.{salt}.{i}", seed).choice(pool)
            else:
                c = cands[0]
            used.add(os.path.normcase(c[0]))
            built.append(f"{_rel(c[0])} @{weight}" if weight else _rel(c[0]))
            notes.append(f"partner for {os.path.basename(lead)} -> {os.path.basename(c[0])} ({c[2]}, {c[1]:+.2f})")
        out.append(" | ".join(built))
    return out


_CROSS = re.compile(r"\s*cross\s*:\s*", re.IGNORECASE)


def _expand_cross(i: int, ln: str, notes: list) -> list:
    """'cross: random: illustration x4 | random: graphic-arts x4 | treatment_a @0.6' -> every combination:
    each random slot draws its N LoRAs once (distinct), an 'all:' slot offers every LoRA of its source, a fixed
    slot stays as written; 4 x 4 x 1 = 16 stacks. A combination that would hold the same file twice is skipped."""
    import itertools
    parts = [p.strip() for p in ln.split("|") if p.strip()]
    if not parts:
        return []
    slots = []
    for part in parts:
        name, weight = part, None
        if "@" in part:
            name, weight = part.rsplit("@", 1)
            name, weight = name.strip(), weight.strip()
        slots.append((part, name, weight, parse_pick(name)))
    fixed = [find_lora(s[1]) or s[1] for s in slots if s[3] is None]
    fixed_n = {os.path.normcase(f) for f in fixed}
    options, described = [], []
    for j, (part, name, weight, pick) in enumerate(slots):
        if pick is None:
            options.append([(part, os.path.normcase(find_lora(name) or name))])
            continue
        key, pool = pick_pool(pick)
        usable = [p for p in pool if os.path.normcase(p) not in fixed_n] or list(pool)
        if pick["mode"] == "all":
            chosen = usable
        else:
            chosen = _draw(usable, min(pick["n"], len(usable)), _rng(key, f"{i}.{j}.cross", pick["seed"]))
        options.append([((f"{_rel(p)} @{weight}" if weight else _rel(p)), os.path.normcase(p)) for p in chosen])
        described.append(f"{name}: " + ", ".join(os.path.basename(p) for p in chosen))
    out, skipped = [], 0
    for combo in itertools.product(*options):
        files = [f for _t, f in combo]
        if len(set(files)) < len(files):
            skipped += 1
            continue
        out.append(" | ".join(t for t, _f in combo))
    sizes = " x ".join(str(len(o)) for o in options)
    notes.append(f"'cross: {ln}' -> {len(out)} stack(s) ({sizes}" + (f", {skipped} with a LoRA twice skipped" if skipped
                                                                        else "") + "): " + "; ".join(described))
    return out


def expand_random_stacks(lines: list, notes: list) -> list:
    """Stack lines with random / all slots filled in. 'xN' on a random slot repeats the line N times; an 'all:' slot
    repeats it once per LoRA of the source (minus the ones written by hand in the line). The largest count wins;
    the fixed slots stay as written."""
    out = []
    for i, ln in enumerate(lines):
        cross = _CROSS.match(ln)
        if cross:                                  # 'cross:' = every combination of the slots' picks (2026-10-16)
            out += _expand_cross(i, ln[cross.end():], notes)
            continue
        parts = [p.strip() for p in ln.split("|") if p.strip()]
        slots = []
        for part in parts:
            name, weight = part, None
            if "@" in part:
                name, weight = part.rsplit("@", 1)
                name, weight = name.strip(), weight.strip()
            slots.append((part, name, weight, parse_pick(name)))
        if all(s[3] is None for s in slots):
            out.append(ln)
            continue
        fixed = [find_lora(s[1]) or s[1] for s in slots if s[3] is None]
        fixed_n = {os.path.normcase(f) for f in fixed}
        pools, draws, reps = {}, {}, 1
        for j, (_part, _name, _w, pick) in enumerate(slots):
            if pick is None:
                continue
            key, pool = pick_pool(pick)
            pools[j] = pool
            if pick["mode"] == "all":
                usable = [p for p in pool if os.path.normcase(p) not in fixed_n]
                draws[j] = usable or list(pool)
            else:
                draws[j] = None
            reps = max(reps, len(draws[j]) if draws[j] is not None else pick["n"])
        for j, (_part, _name, _w, pick) in enumerate(slots):
            if pick is not None and draws[j] is None:
                key, _pool = pick_pool(pick)
                draws[j] = _draw(pools[j], reps, _rng(key, f"{i}.{j}", pick["seed"]), avoid=fixed)
            elif pick is not None and len(draws[j]) < reps:          # two all-slots of different sizes
                draws[j] = (draws[j] * (reps // len(draws[j]) + 1))[:reps]
        picked = []
        for r in range(reps):
            used, built, names = set(fixed_n), [], []
            for j, (part, _name, weight, pick) in enumerate(slots):
                if pick is None:
                    built.append(part)
                    continue
                pick_path = draws[j][r]
                if os.path.normcase(pick_path) in used:   # two pick slots drew the same file
                    alt = [p for p in pools[j] if os.path.normcase(p) not in used]
                    pick_path = alt[0] if alt else pick_path
                used.add(os.path.normcase(pick_path))
                names.append(os.path.basename(pick_path))
                # no weight written: leave it off, so the plan gives the pick its card weight (2026-10-15)
                built.append(f"{_rel(pick_path)} @{weight}" if weight else _rel(pick_path))
            out.append(" | ".join(built))
            picked.append(", ".join(names))
        notes.append(f"'{ln}' -> {reps} stack(s): " + "; ".join(picked))
    return out
