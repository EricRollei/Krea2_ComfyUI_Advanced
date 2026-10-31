# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Eric_Krea2 / _prompt_lines.py
#
# The Sweep Plan's `prompts` field (2026-10-26): besides typed prompts (one per line), a line can
# pull prompts from a text file - the same wording as the LoRA picks:
#
#   file: D:\prompts\portraits.txt              every prompt in the file (also: all:)
#   random: D:\prompts\portraits.txt x20        20 different prompts drawn at random
#   random: prompts\portraits.txt x20 seed=7    another repeatable draw; seed=new = fresh every queue
#   A lighthouse at dawn, oil on canvas         typed prompts still work, mixed freely
#
# Files: one prompt per line; blank lines and lines starting with # are skipped. A relative path is
# taken from the ComfyUI folder. A line counts as a file line only when its path ends in .txt or
# names an existing file - anything else is an ordinary prompt. The ACTUAL prompt text goes into
# every cell, so results stay reproducible even if the file changes later.

from __future__ import annotations

import os
import random
import re

_DIRECTIVE = re.compile(r"^\s*(file|all|random)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _base_dir() -> str:
    try:
        import folder_paths                       # ComfyUI
        return folder_paths.base_path
    except Exception:
        return os.getcwd()


def _resolve(path: str) -> str:
    p = os.path.expandvars(os.path.expanduser(path.strip().strip('"').strip("'")))
    if not os.path.isabs(p):
        p = os.path.join(_base_dir(), p)
    return os.path.normpath(p)


def _parse(line: str):
    """(mode, full_path, n, seed) for a file line, or None for an ordinary prompt."""
    m = _DIRECTIVE.match(line)
    if not m:
        return None
    mode, toks = m.group(1).lower(), m.group(2).split()
    n, seed = None, ""
    while toks:
        t = toks[-1]
        if re.fullmatch(r"[xX]\d+", t):
            n = int(t[1:])
        elif t.lower().startswith("seed="):
            seed = t.split("=", 1)[1]
        else:
            break
        toks.pop()
    path = " ".join(toks)
    full = _resolve(path)
    if not os.path.isfile(full):
        if path.strip().strip('"').lower().endswith(".txt"):
            raise ValueError(f"SweepPlan prompts: file not found: {full}")
        return None                               # e.g. a prompt that starts with "random: ..."
    return mode, full, n, seed


def _read(full: str) -> list:
    with open(full, encoding="utf-8-sig", errors="replace") as f:
        out = [ln.strip() for ln in f]
    out = [ln for ln in out if ln and not ln.startswith("#")]
    if not out:
        raise ValueError(f"SweepPlan prompts: no prompts in {full} (blank and # lines are skipped)")
    return out


def expand_prompt_lines(text, notes=None) -> list:
    """The prompts field -> the list of prompt texts (file lines expanded, duplicates dropped)."""
    out = []
    for i, line in enumerate(str(text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        spec = _parse(line)
        if spec is None:
            out.append(line)
            continue
        mode, full, n, seed = spec
        lines = _read(full)
        name = os.path.basename(full)
        if mode in ("file", "all"):
            got = lines
            msg = f"prompts: '{name}' -> all {len(lines)} prompts"
        else:
            n = n or 1
            rng = random.Random() if str(seed).lower() == "new" else random.Random(f"{full}|{i}|{seed}")
            if n >= len(lines):
                got = list(lines)
                msg = f"prompts: '{name}' x{n} -> only {len(lines)} in the file, using all"
            else:
                got = rng.sample(lines, n)
                msg = f"prompts: '{name}' x{n} -> {n} of {len(lines)} drawn" + (" (seed=new)" if
                                                                               str(seed).lower() == "new" else "")
        out += got
        print(f"[EricKrea2-Sweep] {msg}")
        if notes is not None:
            notes.append(msg)
    return list(dict.fromkeys(out))


def files_signature(text) -> str:
    """Changes when a prompt file used by the field is edited - so the plan is rebuilt."""
    sig = []
    for line in str(text or "").splitlines():
        try:
            spec = _parse(line.strip())
        except ValueError:
            continue
        if spec:
            try:
                sig.append(f"{spec[1]}:{os.path.getmtime(spec[1])}")
            except OSError:
                pass
    return "|".join(sig)
