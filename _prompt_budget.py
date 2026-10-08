# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 prompt token budget
==============================
Krea 2 encodes a FIXED 512-token text budget; the chat template's 5-token suffix leaves 507 for
the prompt, and diffusers truncates anything beyond that SILENTLY - from the end, which is where
appended LoRA trigger words sit. This module counts prompt tokens exactly as the pipeline does
and shortens the prompt BODY so trigger words always survive.

Trim strategy (prose prompts lead with the subject and end with medium / style / framing, so
both ends matter): drop whole sentences from just before the last one; only if that is not
enough, cut the end at the last sentence / clause / word that fits.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import re

MAX_SEQUENCE_LENGTH = 512


def prompt_counter(pipe):
    """(count_fn, budget) for this pipeline's tokenizer/template, or (None, 0) if unavailable.
    count_fn(text) = tokens the PROMPT occupies (template prefix excluded)."""
    try:
        tok = pipe.tokenizer
        prefix = pipe.prompt_template_encode_prefix
        start = int(pipe.prompt_template_encode_start_idx)
        budget = MAX_SEQUENCE_LENGTH - int(pipe.prompt_template_encode_num_suffix_tokens)
    except Exception:
        return None, 0

    def count(text):
        return len(tok(prefix + (text or ""), truncation=False).input_ids) - start
    return count, budget


def _drop_before_last(text, budget, count):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    keep, removed = list(sentences), []
    while len(keep) >= 3:
        removed.insert(0, keep.pop(-2))
        cand = " ".join(keep)
        if count(cand) <= budget:
            return cand, " ".join(removed)
    return None, ""


def _cut_end(text, budget, count):
    words = list(re.finditer(r"\S+", text))
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:words[mid - 1].end()]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    if lo <= 0:
        return "", text
    end = words[lo - 1].end()
    cut = end
    for pattern in (r"[.!?](?=\s|$)", r"[,;:](?=\s|$)"):
        hits = [m.end() for m in re.finditer(pattern, text[:end])]
        if hits and hits[-1] >= end * 0.6:
            cut = hits[-1]
            break
    body = text[:cut].rstrip()
    if body and body[-1] in ",;:":
        body = body[:-1].rstrip() + "."
    return body, text[cut:].strip()


def fit_body(body, budget, count):
    """Shorten `body` to <= budget tokens. Returns (body, dropped_text)."""
    if count(body) <= budget:
        return body, ""
    out, dropped = _drop_before_last(body, budget, count)
    if out is not None:
        return out, dropped
    return _cut_end(body, budget, count)


def join_triggers(body, tail, mode):
    body = (body or "").strip()
    if not body:
        return tail
    if mode == "prepend":
        return f"{tail}, {body}"
    body = body.rstrip(",;:").rstrip()
    return f"{body} {tail}" if body[-1:] in ".!?" else f"{body}, {tail}"


def merge_within_budget(prompt, tail, mode, pipe, log_prefix="[EricKrea2-Triggers]"):
    """Merge `tail` (trigger words) into `prompt` (prepend/append) so the result fits Krea 2's
    prompt budget, shortening the prompt body - never the triggers. Falls back to a plain merge
    when no tokenizer is available."""
    merged = join_triggers(prompt, tail, mode)
    count, budget = prompt_counter(pipe) if pipe is not None else (None, 0)
    if count is None:
        return merged
    try:
        n = count(merged)
        if n <= budget:
            return merged
        room = budget - (n - count(prompt))           # tokens left for the body
        for _ in range(3):                            # joint tokenization can differ by a token or two
            body, dropped = fit_body(prompt, max(0, room), count)
            out = join_triggers(body, tail, mode)
            m = count(out)
            if m <= budget:
                print(f"{log_prefix} prompt + triggers was {n} tokens (Krea 2 reads {budget}): shortened the "
                      f"prompt to {m} so the trigger words survive. Removed: \"{dropped[:100]}"
                      f"{'...' if len(dropped) > 100 else ''}\"")
                return out
            room -= (m - budget)
        print(f"{log_prefix} WARNING: prompt + triggers is {n} tokens and could not be fitted to {budget}")
        return merged
    except Exception as e:
        print(f"{log_prefix} token budget check skipped ({type(e).__name__}: {e})")
        return merged
