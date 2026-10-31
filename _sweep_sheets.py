# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Share sheets for a finished sweep (2026-10-16): pictures to post (Civitai, Reddit ...) that explain themselves.

    strips   one picture per prompt (or per value of another axis): the prompt and the settings every image shares
             at the top, each image with its LoRAs underneath - short model name, weight, creator, Civitai model id.
             Images at `tile_px` (768 by default) so people can pixel-peep; a strip wraps into more rows instead of
             growing past `max_px` (6000 px, e.g. LinkedIn's limit) and splits into parts if it would grow taller.
    singles  every cell at full size on a border: checkpoint / sampler / schedule / steps / seed on top, the LoRAs
             (name, weight, creator, model id) at the bottom. No prompt, so the picture stays a picture.

Written to <sweep folder>/share/ as JPEG (quality 92). Reads only the sweep folder (manifest + cell PNGs) and the LoRA
Catalog (names, creators, Civitai ids; your stars / notes from the Sweeps page when `ratings` asks for them), so it
works on any finished sweep, again and again, with other settings. Self-contained on purpose (PIL, json, sqlite3
only): the LoRA Catalog's Sweeps page loads this file directly.

    make_sheets(sweep_dir, mode="strips", background="black", group_by="auto", tile_px=768, max_px=6000,
                ratings="none", catalog=None) -> {"ok", "files", "notes"}
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import textwrap
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CATALOG = r"L:\Models\loras\_lora_catalog.sqlite"
FONT_DIRS = [r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]
LATIN = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
LATIN_BOLD = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
CJK = ["msyh.ttc", "msyh.ttf", "YuGothM.ttc", "simsun.ttc"]
SYMBOL = ["seguisym.ttf", "DejaVuSans.ttf"]
THEMES = {"black": {"bg": (12, 12, 12), "fg": (236, 236, 232), "muted": (150, 150, 145), "star": (232, 182, 40)},
          "white": {"bg": (255, 255, 255), "fg": (24, 24, 22), "muted": (110, 110, 104), "star": (196, 140, 0)}}
NOISE = re.compile(r"[\[\(]?\b(krea\s*-?\s*2|kr2|krea2|for krea( 2)?|lora|style lora)\b[\]\)]?", re.I)


# -- fonts ---------------------------------------------------------------------------------------------------------
_FONT_CACHE: Dict[Tuple[str, int], Any] = {}


def _font(kind: str, size: int):
    from PIL import ImageFont
    key = (kind, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = {"bold": LATIN_BOLD, "cjk": CJK, "symbol": SYMBOL}.get(kind, LATIN)
    font = None
    for d in FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except OSError:
                    continue
        if font:
            break
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _font_for(text: str, size: int, bold: bool = False):
    if any(ord(ch) >= 0x2E80 for ch in text):
        return _font("cjk", size)
    if any(0x2600 <= ord(ch) <= 0x27BF for ch in text):
        return _font("symbol", size)
    return _font("bold" if bold else "regular", size)


def _width(draw, text: str, font) -> int:
    return int(draw.textlength(text, font=font))


def _fit(draw, text: str, font, width: int) -> str:
    """One line, cut with an ellipsis to fit `width` px."""
    if _width(draw, text, font) <= width:
        return text
    while text and _width(draw, text + "...", font) > width:
        text = text[:-1]
    return text.rstrip(" ,;-") + "..."


def _wrap(draw, text: str, font, width: int, max_lines: int) -> List[str]:
    words, lines, cur = str(text or "").split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if _width(draw, trial, font) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = w
        if len(lines) == max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    rest = " ".join(words)
    shown = " ".join(lines)
    if len(shown) < len(rest) and lines:
        lines[-1] = _fit(draw, lines[-1] + " ...", font, width)
    return lines


# -- sweep + catalog facts -----------------------------------------------------------------------------------------
def _manifest(sweep_dir: str) -> Dict[str, Any]:
    paths = sorted(glob.glob(os.path.join(sweep_dir, "sweep_manifest*.json")))
    if not paths:
        raise ValueError(f"no sweep manifest in {sweep_dir}")
    with open(paths[0], "r", encoding="utf-8") as handle:
        return json.load(handle)


def _catalog_path(catalog: Optional[str]) -> str:
    if catalog and os.path.isfile(catalog):
        return catalog
    env = os.environ.get("ERIC_LORA_CATALOG", "")
    if env and os.path.isfile(env):
        return env
    return DEFAULT_CATALOG if os.path.isfile(DEFAULT_CATALOG) else ""


def short_name(name: str, limit: int = 34) -> str:
    """'Arthur Rackham, Pen and Ink and Watercolor illustrations [Krea2]' -> 'Arthur Rackham, Pen and Ink and ...'"""
    n = NOISE.sub(" ", str(name or ""))
    n = re.sub(r"\s*[|/]\s*(style)?\s*[|/]?\s*$", "", n)
    n = re.sub(r"[\s_]+", " ", n).strip(" -|,_")
    n = re.sub(r"\s*\|\s*", " | ", n)
    if len(n) > limit:
        n = n[:limit].rsplit(" ", 1)[0].rstrip(" ,;-|") + "..."
    return n or str(name or "")


class Facts:
    """LoRA names / creators / Civitai ids and your marks, from the LoRA Catalog (empty when there is none)."""

    def __init__(self, catalog: Optional[str], sweep_id: str):
        self.by_path: Dict[str, Dict[str, Any]] = {}
        self.by_file: Dict[str, Dict[str, Any]] = {}
        self.marks: Dict[str, Dict[str, Any]] = {}
        path = _catalog_path(catalog)
        if not path:
            return
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
            con.row_factory = sqlite3.Row
            tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            creators = {}
            if "card_facets" in tables:
                creators = {r["path"]: (r["creator"] or "", r["name"] or "")
                            for r in con.execute("select path, creator, name from card_facets")}
            for r in con.execute("select path, file_name, stem, model_name, version_name, civitai_model_id, "
                                 "civitai_version_id from loras"):
                cr, nm = creators.get(r["path"], ("", ""))
                rec = {"name": r["model_name"] or nm or r["stem"], "version": r["version_name"] or "",
                       "model_id": r["civitai_model_id"], "version_id": r["civitai_version_id"], "creator": cr,
                       "stem": r["stem"]}
                self.by_path[os.path.normcase(r["path"])] = rec
                self.by_file.setdefault(str(r["file_name"]).lower(), rec)
            if "stacks" in tables:
                cols = {c[1] for c in con.execute("pragma table_info(stacks)")}
                sel = "cell_file, verdict, notes" + (", stars" if "stars" in cols else "")
                for r in con.execute(f"select {sel} from stacks where sweep_id = ?", (sweep_id,)):
                    self.marks[r["cell_file"]] = dict(r)
            con.close()
        except sqlite3.Error:
            pass

    def lora(self, path: str) -> Dict[str, Any]:
        rec = self.by_path.get(os.path.normcase(str(path or ""))) or \
            self.by_file.get(os.path.basename(str(path or "")).lower())
        stem = os.path.splitext(os.path.basename(str(path or "")))[0]
        return rec or {"name": stem, "version": "", "model_id": None, "version_id": None, "creator": "", "stem": stem}


def _lora_lines(entries: List[Dict[str, Any]], facts: Facts) -> List[Tuple[str, str]]:
    """[(name @weight, 'by creator - civitai 123456'), ...] for the cell's swept LoRAs (panel LoRAs marked)."""
    out = []
    for e in entries:
        info = facts.lora(e.get("path") or e.get("filename") or "")
        ws = [e.get("weight_s1", e.get("strength")), e.get("weight_s2"), e.get("weight_s3")]
        ws = [float(w) for w in ws if w is not None]
        wt = f"{ws[0]:g}" if ws and all(abs(w - ws[0]) < 1e-6 for w in ws) else "/".join(f"{w:g}" for w in ws)
        name = short_name(info["name"])
        panel = "" if str(e.get("adapter_name", "")).startswith("sweep") else " (panel)"
        credit = " - ".join(x for x in [f"by {info['creator']}" if info["creator"] else "",
                                        f"civitai {info['model_id']}" if info["model_id"] else ""] if x)
        out.append((f"{name} @{wt}{panel}", credit))
    return out


def _cell_info(sweep_dir: str, cell: Dict[str, Any]):
    from PIL import Image
    path = os.path.join(sweep_dir, cell["file"])
    img = Image.open(path)
    try:
        chunk = json.loads(img.info.get("krea2_settings") or "{}")
    except ValueError:
        chunk = {}
    return img.convert("RGB"), chunk


def _settings_of(cell: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, str]:
    r = dict(chunk.get("ultra") or {})
    r.update(cell.get("resolved") or {})
    out = {}
    if r.get("s1_sampler"):
        out["sampler"] = str(r["s1_sampler"])
    if r.get("s1_schedule"):
        out["schedule"] = str(r["s1_schedule"])
    sig = chunk.get("sigmas") or {}
    if isinstance(sig, dict) and isinstance(sig.get("s1"), dict) and (cell.get("overrides") or {}).get("sigma_profile"):
        out["schedule"] = "profile " + str(cell["overrides"]["sigma_profile"])
    if r.get("s1_steps"):
        out["steps"] = f"{r['s1_steps']} steps"
    if cell.get("seed") is not None:
        out["seed"] = f"seed {cell['seed']}"
    if r.get("s1_cfg") not in (None, "", 0, 0.0):
        try:
            if float(r["s1_cfg"]) > 1.0:
                out["cfg"] = f"cfg {float(r['s1_cfg']):g}"
        except (TypeError, ValueError):
            pass
    up2 = r.get("upscale_to_stage2")
    try:
        out["stages"] = "3 stages" if float(r.get("upscale_to_stage3") or 0) > 0 and float(up2 or 0) > 0 else \
            "2 stages" if float(up2 or 0) > 0 else "1 stage"
    except (TypeError, ValueError):
        pass
    return out


def _stars(n: Optional[int]) -> str:
    return ("★" * int(n) + "☆" * (5 - int(n))) if n else ""


# -- layout --------------------------------------------------------------------------------------------------------
def _groups(cells: List[Dict[str, Any]], group_by: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """[(group title value, cells)] in plan order."""
    prompts = {str((c.get("overrides") or {}).get("prompt", "")) for c in cells}
    key = group_by
    if key == "auto":
        key = "prompt" if len(prompts) > 1 else "none"
    if key == "none":
        return [("", cells)]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for c in cells:
        over = c.get("overrides") or {}
        if key == "prompt":
            v = str(over.get("prompt", ""))
        elif key == "lora":
            v = str(over.get("lora_stack") or over.get("lora_file") or ("no LoRA" if over.get("lora_baseline") else ""))
        elif key == "seed":
            v = str(c.get("seed"))
        else:
            v = str(next((val for k, val in over.items() if k.endswith(key)), ""))
        out.setdefault(v, []).append(c)
    return list(out.items())


def _grid(n: int, tile_w: int, gap: int, margin: int, max_px: int) -> Tuple[int, int]:
    """(columns, rows) for n images: as few rows as fit `max_px` wide, the images spread evenly over the rows."""
    cols_max = max(1, (max_px - 2 * margin + gap) // (tile_w + gap))
    rows = -(-n // cols_max)
    cols = -(-n // rows)
    return cols, rows


def make_sheets(sweep_dir: str, mode: str = "strips", background: str = "black", group_by: str = "auto",
                tile_px: int = 768, max_px: int = 6000, ratings: str = "none", catalog: Optional[str] = None,
                log=print) -> Dict[str, Any]:
    from PIL import Image, ImageDraw
    if mode not in ("strips", "singles", "both", "strips + singles"):
        return {"ok": True, "files": [], "notes": ["share sheets off"]}
    manifest = _manifest(sweep_dir)
    sweep_id = os.path.basename(os.path.normpath(sweep_dir))
    facts = Facts(catalog, sweep_id)
    theme = THEMES.get(background, THEMES["black"])
    out_dir = os.path.join(sweep_dir, "share")
    os.makedirs(out_dir, exist_ok=True)
    cells = [c for c in manifest.get("cells") or [] if c.get("file")]
    if not cells:
        return {"ok": False, "error": "no finished cells in this sweep"}
    checkpoint = str(manifest.get("checkpoint") or "")
    lora_cfg = (manifest.get("plan") or {}).get("lora") or {}
    words = str(lora_cfg.get("triggers") or "")
    words = {"off": "none", "catalog": "trigger", "recipe": "saved pick"}.get(words, words)
    created = str(manifest.get("created") or "")[:16].replace("T", " ")
    title = str((manifest.get("about") or {}).get("sweep_name") or manifest.get("sweep_name") or sweep_id)
    files, notes = [], []

    info = {}
    for c in cells:
        img, chunk = _cell_info(sweep_dir, c)
        info[c["file"]] = (img, chunk)

    def label_for(c) -> Dict[str, Any]:
        img, chunk = info[c["file"]]
        over = c.get("overrides") or {}
        entries = list(chunk.get("lora") or [])
        lines = _lora_lines(entries, facts) if entries else []
        mark = facts.marks.get(c["file"]) or {}
        return {"loras": lines, "baseline": bool(over.get("lora_baseline")) or not entries,
                "settings": _settings_of(c, chunk), "stars": mark.get("stars"), "verdict": mark.get("verdict"),
                "note": mark.get("notes") or ""}

    labels = {c["file"]: label_for(c) for c in cells}
    show_stars = ratings in ("stars", "stars + notes")
    show_notes = ratings == "stars + notes"

    # ---- strips -----------------------------------------------------------------------------------------------
    if mode in ("strips", "both", "strips + singles"):
        gap, margin = 16, 32
        f_title = 30
        f_label = 22
        for gi, (gval, gcells) in enumerate(_groups(cells, group_by), 1):
            # what every image in the strip shares goes on top, what differs goes under each image
            setts = [labels[c["file"]]["settings"] for c in gcells]
            keys = list(dict.fromkeys(k for s in setts for k in s))
            common = [k for k in keys if all(s.get(k) == setts[0].get(k) for s in setts)]
            varying = [k for k in keys if k not in common]
            img0 = info[gcells[0]["file"]][0]
            tile_w = tile_px
            tile_h = int(round(tile_px * max(info[c["file"]][0].height / info[c["file"]][0].width for c in gcells)))
            n_lora_lines = max([len(labels[c["file"]]["loras"]) or 1 for c in gcells] or [1])
            under = n_lora_lines * 2 + (1 if varying else 0) + (1 if show_stars else 0) + (1 if show_notes else 0)
            label_h = under * (f_label + 8) + 14
            cols, rows = _grid(len(gcells), tile_w, gap, margin, max_px)
            canvas_w = margin * 2 + cols * tile_w + (cols - 1) * gap
            probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
            body_w = canvas_w - 2 * margin
            head = []
            if gval and group_by in ("auto", "prompt"):
                head = [("bold", l) for l in _wrap(probe, gval, _font("bold", f_title), body_w, 3)]
            elif gval:
                head = [("bold", _fit(probe, f"{group_by}: {gval}", _font("bold", f_title), body_w))]
            meta = " - ".join([f"Krea 2 checkpoint: {checkpoint}"] + [setts[0][k] for k in common] +
                              ([f"LoRA words: {words}"] if words else []))
            head.append(("muted", _fit(probe, meta, _font("regular", f_label), body_w)))
            head_h = margin + sum((f_title + 10) if s == "bold" else (f_label + 10) for s, _ in head) + 12
            foot_h = f_label + 2 * 14
            row_h = tile_h + label_h + gap
            rows_fit = max(1, (max_px - head_h - foot_h) // row_h)
            parts = [gcells[i:i + rows_fit * cols] for i in range(0, len(gcells), rows_fit * cols)]
            for pi, part in enumerate(parts, 1):
                prow = -(-len(part) // cols)
                canvas_h = head_h + prow * row_h + foot_h
                canvas = Image.new("RGB", (canvas_w, canvas_h), theme["bg"])
                d = ImageDraw.Draw(canvas)
                y = margin
                for style, line in head:
                    size = f_title if style == "bold" else f_label
                    d.text((margin, y), line, font=_font_for(line, size, style == "bold"),
                           fill=theme["fg"] if style == "bold" else theme["muted"])
                    y += size + 10
                y = head_h
                for i, c in enumerate(part):
                    r, k = divmod(i, cols)
                    x0 = margin + k * (tile_w + gap)
                    y0 = y + r * row_h
                    img = info[c["file"]][0].copy()
                    img.thumbnail((tile_w, tile_h), Image.LANCZOS)
                    canvas.paste(img, (x0 + (tile_w - img.width) // 2, y0 + (tile_h - img.height) // 2))
                    lab = labels[c["file"]]
                    ly = y0 + tile_h + 8
                    lines: List[Tuple[str, str]] = []
                    if lab["baseline"] and not lab["loras"]:
                        lines.append(("fg", "no LoRA (reference)"))
                    for name, credit in lab["loras"]:
                        lines.append(("fg", name))
                        if credit:
                            lines.append(("muted", credit))
                    if varying:
                        lines.append(("muted", " - ".join(lab["settings"].get(v, "") for v in varying)))
                    if show_stars and (lab["stars"] or lab["verdict"]):
                        lines.append(("star", _stars(lab["stars"]) or lab["verdict"]))
                    if show_notes and lab["note"]:
                        lines.append(("muted", lab["note"]))
                    for style, line in lines:
                        f = _font_for(line, f_label, style == "fg")
                        d.text((x0, ly), _fit(d, line, f, tile_w), font=f, fill=theme[style] if style in theme
                               else theme["fg"])
                        ly += f_label + 8
                foot = f"sweep: {title} - {created}" + (f" - part {pi}/{len(parts)}" if len(parts) > 1 else "") + \
                       " - LoRAs by their creators, see civitai.com/models/<id>"
                d.text((margin, canvas_h - foot_h + 10), _fit(d, foot, _font("regular", f_label - 4), body_w),
                       font=_font("regular", f_label - 4), fill=theme["muted"])
                slug = re.sub(r"[^\w]+", "_", gval)[:40].strip("_") if gval else "all"
                name = f"strip_{gi:03d}_{slug}" + (f"_part{pi}" if len(parts) > 1 else "") + ".jpg"
                canvas.save(os.path.join(out_dir, name), quality=92, optimize=True)
                files.append(name)
        log(f"[sweep sheets] {len(files)} strip(s) in {out_dir}")

    # ---- singles ----------------------------------------------------------------------------------------------
    if mode in ("singles", "both", "strips + singles"):
        n0 = len(files)
        for c in cells:
            img = info[c["file"]][0]
            lab = labels[c["file"]]
            size = max(18, img.width // 64)
            pad = size
            top = " - ".join([checkpoint] + [lab["settings"][k] for k in ("sampler", "schedule", "steps", "cfg",
                                                                           "stages", "seed") if k in lab["settings"]])
            bottom: List[Tuple[str, str]] = []
            if not lab["loras"]:
                bottom.append(("fg", "no LoRA"))
            for name, credit in lab["loras"]:
                bottom.append(("fg", name + (f"  ({credit})" if credit else "")))
            if show_stars and (lab["stars"] or lab["verdict"]):
                bottom.append(("star", _stars(lab["stars"]) or lab["verdict"]))
            if show_notes and lab["note"]:
                bottom.append(("muted", lab["note"]))
            top_h = size + 2 * pad
            bot_h = len(bottom) * (size + pad // 2) + 2 * pad
            canvas = Image.new("RGB", (img.width + 2 * pad, img.height + top_h + bot_h), theme["bg"])
            canvas.paste(img, (pad, top_h))
            d = ImageDraw.Draw(canvas)
            d.text((pad, pad), _fit(d, top, _font("regular", size), img.width), font=_font("regular", size),
                   fill=theme["muted"])
            y = top_h + img.height + pad
            for style, line in bottom:
                f = _font_for(line, size, style == "fg")
                d.text((pad, y), _fit(d, line, f, img.width), font=f, fill=theme.get(style, theme["fg"]))
                y += size + pad // 2
            m = re.match(r"(cell_\d+)", os.path.basename(c["file"]))
            cell_id = m.group(1) if m else f"cell_{int(c.get('index', 0)):03d}"
            who = lab["loras"][0][0].split(" @")[0] if lab["loras"] else "no_lora"
            if len(lab["loras"]) > 1:
                who += f"_plus{len(lab['loras']) - 1}"
            slug = re.sub(r"[^\w]+", "_", who).strip("_")[:40] or "lora"
            name = f"single_{cell_id}_{slug}.jpg"
            canvas.save(os.path.join(out_dir, name), quality=92, optimize=True)
            files.append(name)
        log(f"[sweep sheets] {len(files) - n0} labelled single(s) in {out_dir}")
    return {"ok": True, "files": files, "folder": out_dir, "notes": notes}
