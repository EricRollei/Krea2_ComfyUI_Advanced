# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Prompt-word merging for a stack of LoRAs in the `recipe` trigger mode (2026-10-16).

Each LoRA brings its LoRA Catalog prompt words ("arackham, pen and ink linework ..., muted sepia palette").
Pasting four of those end to end repeats words, runs long and can contradict itself. `merge_recipes` turns the
stack's words into one tidy phrase list, in four layers:

  1. duplicates: the same phrase twice (case, hyphens and punctuation ignored) is kept once, and a phrase whose
     words all appear in a longer kept phrase ("watercolor" vs "soft watercolor washes") is dropped. Phrases the
     prompt already says are dropped too.
  2. picks: a LoRA whose words are a pick (yours, or a trigger test's, for this checkpoint) keeps all of them -
     no trimming, only exact duplicates go.
  3. order and budget: each LoRA's first phrase (usually its trigger) comes first, in stack order; then phrases
     two or more LoRAs share; then the rest. Unpicked LoRAs share a word budget in proportion to their weight;
     a LoRA always keeps its first phrase plus at least one more.
  4. conflicts: pairs of opposite words from different LoRAs (or LoRA vs prompt) - "muted" vs "vibrant",
     "photorealistic" vs "anime" - are reported, never removed (the choice is yours).

A single LoRA's words pass through untouched apart from layer 1 against the prompt. No widgets: the recipe mode
of the Multi-LoRA Stack node and the sweep's lora_triggers=recipe use it.
"""

from __future__ import annotations

import re

WORD_BUDGET = 60          # words for all unpicked LoRAs together (Krea 2's encoder budget is ~507 tokens)
PICK_SOURCES = {"you", "test"}

_STOP = {"a", "an", "the", "and", "or", "of", "with", "in", "on", "to", "for", "by", "from", "at", "as", "style",
         "art", "look", "like", "into", "over", "under"}

# layer 4: opposite word groups. Two phrases conflict when one holds a word of group A and the other a word of B.
CONFLICTS = [
    ({"muted", "desaturated", "subdued", "pastel", "pastels"}, {"vibrant", "saturated", "vivid", "neon", "garish"}),
    ({"monochrome", "monochromatic", "grayscale", "greyscale", "b&w", "sepia"},
     {"colorful", "colourful", "polychrome", "rainbow", "multicolored"}),
    ({"photorealistic", "photoreal", "photograph", "photographic", "hyperrealistic"},
     {"anime", "cartoon", "cartoonish", "chibi", "manga", "flat"}),
    ({"dark", "moody", "gloomy", "gothic", "noir"}, {"bright", "airy", "cheerful", "sunny", "luminous"}),
    ({"minimalist", "minimal", "sparse", "simple"}, {"ornate", "intricate", "maximalist", "busy", "elaborate"}),
    ({"smooth", "clean", "polished", "glossy"}, {"gritty", "grainy", "grungy", "rough", "distressed"}),
    ({"warm"}, {"cool", "cold", "icy"}),
    ({"soft", "gentle", "delicate"}, {"harsh", "hard-edged", "stark"}),
    ({"flat"}, {"volumetric", "3d", "sculpted", "painterly"}),
    ({"thin", "hairline"}, {"thick", "heavy"}),
    ({"low-contrast", "hazy", "faded"}, {"high-contrast", "chiaroscuro", "punchy"}),
]


def _norm(text: str) -> str:
    t = str(text or "").lower().replace("-", " ").replace("_", " ")
    t = re.sub(r"[^\w@&\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _content(text: str) -> frozenset:
    return frozenset(w for w in _norm(text).split() if w not in _STOP)


def _words(text: str) -> set:
    """Words for the conflict check (hyphen compounds kept as well as split)."""
    low = str(text or "").lower()
    out = set(re.findall(r"[\w@&]+(?:-[\w]+)*", low))
    for w in list(out):
        if "-" in w:
            out.update(w.split("-"))
    if "black and white" in low:
        out.add("b&w")
    if "high contrast" in low:
        out.add("high-contrast")
    if "low contrast" in low:
        out.add("low-contrast")
    return out


def split_phrases(text: str) -> list:
    out = []
    for p in re.split(r"[,;\n]", str(text or "")):
        p = re.sub(r"^(?:and|or|plus)\s+", "", p.strip(), flags=re.I).strip()   # '..., and dramatic lighting'
        if p:
            out.append(p)
    return out


def _in_prompt(phrase: str, prompt_norm: str) -> bool:
    n = _norm(phrase)
    return bool(n) and bool(re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", prompt_norm))


def merge_recipes(items: list, prompt: str = "", budget: int = WORD_BUDGET) -> tuple:
    """items: [{"name", "text", "weight", "source"}] in stack order. Returns (phrases, report) where phrases is the
    merged list to prepend and report = {"dropped": [...], "trimmed": [...], "conflicts": [...], "shared": [...]}."""
    report = {"dropped": [], "trimmed": [], "conflicts": [], "shared": []}
    prompt_norm = _norm(prompt)
    rows = []                                   # one per LoRA: name, weight, picked, phrases
    for it in items or []:
        phrases = split_phrases(it.get("text", ""))
        if not phrases:
            continue
        src = str(it.get("source") or "")
        src = src[src.find("(") + 1:src.rfind(")")] if "(" in src else src
        try:
            w = abs(float(it.get("weight", 1.0) or 0.0))
        except (TypeError, ValueError):
            w = 1.0
        rows.append({"name": str(it.get("name") or "?"), "weight": w or 0.01,
                     "picked": src.strip() in PICK_SOURCES, "phrases": phrases})
    if not rows:
        return [], report

    # layer 1a: exact duplicates and phrases the prompt already says. owners = which LoRAs brought the phrase.
    seen, owners = {}, {}
    for ri, r in enumerate(rows):
        kept = []
        for pi, p in enumerate(r["phrases"]):
            key = _norm(p)
            if not key:
                continue
            if _in_prompt(p, prompt_norm):
                report["dropped"].append(f"'{p}' ({r['name']}): already in the prompt")
                continue
            if key in seen:
                owners[key].add(ri)
                report["dropped"].append(f"'{p}' ({r['name']}): same as {rows[seen[key][0]]['name']}")
                continue
            seen[key] = (ri, pi)
            owners[key] = {ri}
            kept.append(p)
        r["phrases"] = kept
    if len(rows) == 1:
        return list(rows[0]["phrases"]), report

    # layer 1b: a phrase whose words all sit in a longer phrase of ANOTHER LoRA goes (picks are kept). A first
    # phrase (the trigger) only gives way to another LoRA's first phrase ('bradhamel' vs 'bradhamel art style').
    allp = [(ri, pi, p, _content(p)) for ri, r in enumerate(rows) for pi, p in enumerate(r["phrases"])]
    gone = set()
    for ri, pi, p, cw in allp:
        if rows[ri]["picked"] or not cw:
            continue
        for rj, pj, q, cq in allp:
            # same words in another wording ('bradhamel' / 'bradhamel art style'): the earlier LoRA's stays
            if rj == ri or (rj, pj) in gone or not cw <= cq or (len(cq) == len(cw) and rj > ri):
                continue
            if pi == 0 and pj != 0:
                continue
            gone.add((ri, pi))
            owners.setdefault(_norm(q), {rj}).add(ri)
            report["dropped"].append(f"'{p}' ({rows[ri]['name']}): covered by '{q}' ({rows[rj]['name']})")
            break
    for ri, r in enumerate(rows):
        r["phrases"] = [p for pi, p in enumerate(r["phrases"]) if (ri, pi) not in gone]

    shared = {k for k, o in owners.items() if len(o) > 1}

    # layer 3: budget for the unpicked LoRAs, by weight
    free = [r for r in rows if not r["picked"]]
    total_w = sum(r["weight"] for r in free) or 1.0
    for r in free:
        allow = max(1, round(budget * r["weight"] / total_w))
        out, used = [], 0
        for pi, p in enumerate(r["phrases"]):
            n = len(p.split())
            if pi < 2 or _norm(p) in shared or used + n <= allow:
                out.append(p)
                used += n
            else:
                report["trimmed"].append(f"'{p}' ({r['name']})")
        r["phrases"] = out

    heads = [r["phrases"][0] for r in rows if r["phrases"]]
    rest = [p for r in rows for p in r["phrases"][1:]]
    shared_first = [p for p in rest if _norm(p) in shared]
    report["shared"] = list(shared_first)
    merged = heads + shared_first + [p for p in rest if _norm(p) not in shared]

    # layer 4: conflicts between LoRAs, and LoRA vs prompt
    where = []
    for r in rows:
        for p in r["phrases"]:
            where.append((r["name"], p, _words(p)))
    pw = _words(prompt)
    found = set()
    for a, b in CONFLICTS:
        for n1, p1, w1 in where:
            if not (w1 & a):
                continue
            for n2, p2, w2 in where:
                if n2 != n1 and (w2 & b):
                    found.add(f"'{p1}' ({n1}) vs '{p2}' ({n2})")
        for side, other in ((a, b), (b, a)):
            for n1, p1, w1 in where:
                if (w1 & side) and (pw & other):
                    found.add(f"'{p1}' ({n1}) vs the prompt ({', '.join(sorted(pw & other))})")
    report["conflicts"] = sorted(found)
    return merged, report


def report_lines(report: dict) -> list:
    out = []
    if report.get("conflicts"):
        out.append("possible conflicts: " + "; ".join(report["conflicts"]))
    if report.get("dropped"):
        out.append(f"{len(report['dropped'])} repeated phrase(s) dropped")
    if report.get("trimmed"):
        out.append(f"{len(report['trimmed'])} phrase(s) trimmed for length: " + "; ".join(report["trimmed"][:6])
                   + (" ..." if len(report["trimmed"]) > 6 else ""))
    return out
