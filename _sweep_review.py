# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Sweep folder extras (2026-10-15)
================================
Two files written next to a finished sweep's manifest, so a sweep folder still explains itself months later:

    sweep_info.txt      what was run and why: preset name, preset notes, run notes, checkpoint, date, cell count,
                        what varied, and the panel settings every cell started from
    sweep_review.xlsx   a review workbook: one row per cell with a thumbnail, what varied, empty columns for your
                        own ratings (rating, fine detail, render quality, contrast, composition, cohesiveness, notes)
                        and the automatic metrics at the right; an "About" sheet carries sweep_info.txt

Both are built from sweep_manifest.json and the cell PNGs alone, so older sweeps can be backfilled:

    python _sweep_review.py <sweep folder> [<sweep folder> ...]

Never raises into a sweep: write_all() logs and carries on. The workbook needs openpyxl (skipped without it).

Author: Eric Hiss (GitHub: EricRollei)
"""

import io
import json
import os
import sys

try:
    from . import _sweep_files as _sf
except ImportError:                       # run as a script: python _sweep_review.py <folder>
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _sweep_files as _sf

RATING_COLS = ["rating", "fine detail", "render quality", "contrast", "composition", "cohesiveness", "notes"]
METRIC_COLS = [("gen_time_s", None), ("sharp_ten", "sharpness_tenengrad"), ("noise", "noise_sigma"),
               ("clip_pct", "clip_fraction"), ("uniqueness", "uniqueness"), ("splotch", "flat_chroma_var"),
               ("sharp_lap", "sharpness_laplacian")]
# scores.csv (Eric Sweep Folder Scorer, Eric-image-classification): UniPercept's three scores first, by name
# the _sd columns (2026-10-16): how unsure UniPercept is about that image's score (spread of its logits over
# the 0-100 score codes, in score points). Low = a confident score; high = take that score with salt.
SCORE_LABELS = {"unip.iaa": "aesthetics (IAA)", "unip.iaa_sd": "aesthetics doubt (IAA sd)",
                "unip.iqa": "quality (IQA)", "unip.iqa_sd": "quality doubt (IQA sd)",
                "unip.ista": "structure & texture (ISTA)", "unip.ista_sd": "structure doubt (ISTA sd)"}
THUMB_PX = 160
LONG_TEXT = 60          # characters: a swept column with longer values gets a wide, wrapped column


def _load(sweep_dir, manifest=None):
    if manifest is not None:
        return manifest
    path = _sf.find(sweep_dir, "manifest")
    if not path:
        raise FileNotFoundError(f"no sweep manifest in {sweep_dir}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stack_text(stack):
    """A LoRA stack as one line per LoRA: 'file stem @w' or '@s1/s2/s3' (2026-10-15)."""
    lines = []
    for e in stack or []:
        if not isinstance(e, dict):
            continue
        name = os.path.splitext(str(e.get("filename") or os.path.basename(str(e.get("path", "")))))[0]
        try:
            ws = [float(e.get(k, e.get("strength", 1.0))) for k in ("weight_s1", "weight_s2", "weight_s3")]
        except (TypeError, ValueError):
            ws = [1.0, 1.0, 1.0]
        w = f"{ws[0]:g}" if ws[0] == ws[1] == ws[2] else "/".join(f"{x:g}" for x in ws)
        lines.append(f"{name} @{w}")
    return "\n".join(lines)


def _cell_loras(sweep_dir, cell):
    """The LoRAs a cell actually ran: the manifest's 'loras' (sweeps since 2026-10-15), else the 'lora' section
    of the cell PNG's krea2_settings chunk (older sweeps)."""
    if cell.get("loras") is not None:
        return str(cell.get("loras") or "")
    if not cell.get("file"):
        return ""
    try:
        from PIL import Image
        with Image.open(os.path.join(sweep_dir, cell["file"])) as im:
            raw = im.info.get("krea2_settings") or ""
        lora = json.loads(raw).get("lora") if raw else None
        if isinstance(lora, str):
            lora = json.loads(lora)
        if isinstance(lora, dict):
            lora = lora.get("lora") or lora.get("entries") or []
        return stack_text(lora if isinstance(lora, list) else [])
    except Exception:
        return ""


def _about(manifest):
    about = manifest.get("about") or (manifest.get("plan") or {}).get("meta") or {}
    return about if isinstance(about, dict) else {}


def _duration(seconds):
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "?"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _swept_keys(cells):
    return sorted({k for c in cells for k in (c.get("overrides") or {})})


def info_text(manifest):
    about = _about(manifest)
    plan = manifest.get("plan") or {}
    cells = manifest.get("cells") or []
    ok = len([c for c in cells if c.get("file")])
    lora = plan.get("lora") or {}
    seed = next((c.get("seed") for c in cells if c.get("seed") is not None), "")
    preset = about.get("sweep_preset") or ""
    if not about:
        preset_text = "not recorded (this sweep ran before presets and notes were saved, 2026-10-15)"
    elif preset and preset != "custom":
        preset_text = preset
    elif about.get("based_on"):
        edited = about.get("edited_fields") or []
        preset_text = (f"custom, based on preset '{about['based_on']}'"
                       + (f" (edited: {', '.join(edited)})" if edited else " (no fields changed)"))
    else:
        preset_text = "custom (the panel values, no named preset)"
    lines = [
        f"Sweep         {manifest.get('sweep_id', '')}",
        f"Preset        {preset_text}",
        f"Preset notes  {(about.get('preset_notes') or '-').strip()}",
        f"Run notes     {(about.get('run_notes') or '-').strip()}",
        "",
        f"Checkpoint    {manifest.get('checkpoint', '')}",
        f"Created       {manifest.get('created', '')}   (took {_duration(manifest.get('total_s'))})",
        f"Cells         {ok}/{manifest.get('n_planned', len(cells))} rendered"
        + ("   INCOMPLETE (interrupted)" if manifest.get("incomplete") else ""),
        f"Mode          {plan.get('mode', '')}, stage {plan.get('stage_scope', '')}, seed policy "
        f"{plan.get('seed_policy', '')}" + (f", seed {seed}" if seed != "" else "")
        + (", QUICK-SCREEN" if plan.get("quick_screen") else ""),
        f"LoRA          " + (f"{lora.get('mode', '')}, triggers {lora.get('triggers', '')}"
                             + (", panel stack kept" if lora.get("keep_panel_stack") else "") if lora else "none"),
        f"What varies   {', '.join(_swept_keys(cells)) or '-'}",
    ] + (["Random LoRAs  " + "\n              ".join(str(about["random_loras"]).splitlines())]
         if about.get("random_loras") else []) + [
        f"Files         contact_sheet.png, {_sf.names_for(manifest)['manifest']} (everything), "
        f"{_sf.names_for(manifest)['summary']}, {_sf.names_for(manifest)['review']},",
        "              cell_*.png (each PNG carries its own full recipe)",
        "",
        "Panel settings every cell started from (a cell's own values override these):",
    ]
    base = manifest.get("base") or {}
    for key in sorted(base):
        value = base[key]
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        lines.append(f"  {key} = {text if len(text) <= 300 else text[:297] + '...'}")
    return "\n".join(lines) + "\n"


def write_info(sweep_dir, manifest=None):
    manifest = _load(sweep_dir, manifest)
    path = os.path.join(sweep_dir, _sf.names_for(manifest)["info"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(info_text(manifest))
    return path


def _cell_value(key, value):
    if key == "lora_file" and isinstance(value, str):
        return os.path.basename(value)
    if key == "lora_stack" and isinstance(value, str):
        # random picks are written as folder paths: show each LoRA by its file stem, one per line
        parts = []
        for part in value.split("|"):
            name, _at, weight = part.strip().rpartition("@")
            if not _at:
                name, weight = part.strip(), ""
            stem = os.path.splitext(os.path.basename(name.strip().replace("\\", "/")))[0]
            parts.append(stem + (f" @{weight.strip()}" if weight else ""))
        return "\n".join(parts)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _metric(cell, field):
    if field is None:
        return cell.get("time_s", "")
    m = cell.get("metrics") or {}
    if field not in m:
        return ""
    value = m[field]
    if field == "clip_fraction":
        return round(100.0 * float(value), 2)
    if field == "noise_sigma":
        return round(float(value), 1)
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return value


def _scores(sweep_dir):
    """({cell file name: {column: value}}, [score columns in display order]) from scores.csv, or ({}, [])."""
    import csv
    path = os.path.join(sweep_dir, "scores.csv")
    if not os.path.isfile(path):
        return {}, []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        key = next((f for f in fields if f.lower() in ("file", "filename", "name", "image")), fields[0] if fields else "")
        rows = {os.path.basename(str(r.get(key, ""))).strip(): r for r in reader}
    columns = [f for f in fields if f != key]
    columns.sort(key=lambda f: (f not in SCORE_LABELS, list(SCORE_LABELS).index(f) if f in SCORE_LABELS else 0))
    out = {}
    for name, row in rows.items():
        values = {}
        for col in columns:
            try:
                values[col] = round(float(row.get(col, "")), 2)
            except (TypeError, ValueError):
                values[col] = row.get(col, "") or ""
        out[name] = values
    return out, columns


def _previous_ratings(path):
    """{cell index: {rating column: value}} from an existing review workbook, so a rebuild keeps what you typed."""
    if not os.path.isfile(path):
        return {}
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb["Review"] if "Review" in wb.sheetnames else wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except Exception:
        return {}
    if not rows:
        return {}
    header = [str(h or "") for h in rows[0]]
    if "cell" not in header:
        return {}
    at = {name: header.index(name) for name in RATING_COLS if name in header}
    kept = {}
    for row in rows[1:]:
        values = {name: row[i] for name, i in at.items() if i < len(row) and row[i] not in (None, "")}
        if values:
            kept[row[header.index("cell")]] = values
    return kept


def _setting_key(over, keys):
    """One text key per setting combination (prompt and seed excluded) - the "By setting" join."""
    return " | ".join(f"{k}={_cell_value(k, over.get(k, ''))}" for k in keys)


def _add_by_setting(wb, headers, cells, keys):
    """'By setting' sheet (2026-10-24): one row per setting combination, averaged over seeds and
    prompts. Live Excel formulas, so the averages follow the ratings typed into Review. Skipped when
    every setting appears only once (nothing to average)."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    if not keys or "setting id" not in headers:
        return
    groups = {}
    for c in cells:
        groups.setdefault(_setting_key(c.get("overrides") or {}, keys), []).append(c)
    if all(len(v) == 1 for v in groups.values()):
        return
    last = len(cells) + 1
    col = {h: get_column_letter(i) for i, h in enumerate(headers, 1)}

    def rng(h):
        return f"Review!${col[h]}$2:${col[h]}${last}"

    rate = [r for r in RATING_COLS if r != "notes" and r in col]
    metrics = [h for h in ("clip_pct", "sharp_ten", "uniqueness") if h in col]
    head = (list(keys) + ["cells", "rated"] + [f"avg {r}" for r in rate]
            + (["min rating", "max rating"] if "rating" in col else [])
            + [f"avg {m}" for m in metrics] + ["setting id", "setting"])
    sh = wb.create_sheet("By setting")
    sh.append(head)
    grey = PatternFill("solid", fgColor="DDDDDD")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for i, name in enumerate(head, 1):
        cl = sh.cell(row=1, column=i)
        cl.font = Font(bold=True)
        cl.fill = yellow if name.startswith(("avg ", "min ", "max ", "rated")) else grey
        cl.alignment = Alignment(wrap_text=True, vertical="center")
    # Excel's COUNTIF/AVERAGEIFS criteria fail past 255 characters, so the join uses a short id
    # (S01, S02 ... in sorted-key order, matching the Review sheet's "setting id" column).
    sk = rng("setting id")
    key_col = get_column_letter(len(head) - 1)
    for r, (key, members) in enumerate(sorted(groups.items()), 2):
        over = members[0].get("overrides") or {}
        kc = f"${key_col}{r}"
        row = [_cell_value(k, over.get(k, "")) for k in keys]
        row.append(f"=COUNTIF({sk},{kc})")
        row.append(f'=COUNTIFS({sk},{kc},{rng("rating")},">=0")' if "rating" in col else "")
        row += [f'=IFERROR(AVERAGEIFS({rng(x)},{sk},{kc}),"")' for x in rate]
        if "rating" in col:
            rr = rng("rating")
            row.append(f'=IF(COUNTIFS({sk},{kc},{rr},">=0")=0,"",_xlfn.MINIFS({rr},{sk},{kc}))')
            row.append(f'=IF(COUNTIFS({sk},{kc},{rr},">=0")=0,"",_xlfn.MAXIFS({rr},{sk},{kc}))')
        row += [f'=IFERROR(AVERAGEIFS({rng(m)},{sk},{kc}),"")' for m in metrics]
        row += [f"S{r - 1:02d}", key]
        sh.append(row)
    for i, name in enumerate(head, 1):
        sh.column_dimensions[get_column_letter(i)].width = 80 if name == "setting" else (
            12 if i > len(keys) else 14)
    sh.freeze_panes = sh.cell(row=2, column=len(keys) + 1)
    sh.auto_filter.ref = f"A1:{get_column_letter(len(head))}{len(groups) + 1}"


def write_review(sweep_dir, manifest=None, thumb_px=THUMB_PX):
    """sweep_review.xlsx; None when openpyxl is missing. A rebuild (e.g. after scoring) keeps the ratings and notes
    already typed into the old workbook, and leaves the old file as sweep_review.prev.xlsx."""
    try:
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        return None
    from PIL import Image

    manifest = _load(sweep_dir, manifest)
    cells = sorted(manifest.get("cells") or [], key=lambda c: c.get("index", 0))
    swept = _swept_keys(cells)
    group_keys = [k for k in swept if k not in ("prompt", "_seed", "seed")]   # By setting (2026-10-24)
    setting_ids = ({k: f"S{i:02d}" for i, k in enumerate(
        sorted({_setting_key(c.get("overrides") or {}, group_keys) for c in cells}), 1)}
        if group_keys else {})
    names = _sf.names_for(manifest)
    path = os.path.join(sweep_dir, names["review"])
    kept = _previous_ratings(path)
    scores, score_cols = _scores(sweep_dir)
    score_heads = [SCORE_LABELS.get(c, c) for c in score_cols]
    # trajectory-probe summaries (2026-10-22): scalar probe_* fields, after the image metrics
    probe_cols = sorted({k for c in cells for k, v in c.items()
                         if str(k).startswith("probe_") and isinstance(v, (int, float))})
    probe_heads = [k[len("probe_"):].replace("_", " ") for k in probe_cols]
    # every cell's actual LoRA stack, swept or not (2026-10-15: a prompt sweep showed no LoRAs at all)
    loras = {c.get("index"): _cell_loras(sweep_dir, c) for c in cells}
    lora_head = ["LoRAs used"] if any(loras.values()) else []
    headers = (["cell", "image"] + swept + lora_head + RATING_COLS + score_heads
               + [name for name, _f in METRIC_COLS] + probe_heads + ["file", "error"]
               + (["setting id"] if group_keys else []))

    wb = Workbook()
    ws = wb.active
    ws.title = "Review"
    bold = Font(bold=True)
    grey = PatternFill("solid", fgColor="DDDDDD")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    ws.append(headers)
    for col, name in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col)
        cell.font = bold
        cell.fill = yellow if name in RATING_COLS else grey
        cell.alignment = Alignment(wrap_text=True, vertical="center")

    long_cols = {k for k in swept if any(len(str(_cell_value(k, (c.get("overrides") or {}).get(k, ""))))
                                         > LONG_TEXT for c in cells)}
    row_height = thumb_px * 0.75 + 6          # points; a pixel is 0.75 pt
    for r, c in enumerate(cells, 2):
        over = c.get("overrides") or {}
        mine = kept.get(c.get("index", ""), {})
        graded = scores.get(os.path.basename(str(c.get("file", ""))), {})
        values = ([c.get("index", ""), ""] + [_cell_value(k, over.get(k, "")) for k in swept]
                  + [loras.get(c.get("index"), "") for _h in lora_head]
                  + [mine.get(name, "") for name in RATING_COLS] + [graded.get(col, "") for col in score_cols]
                  + [_metric(c, f) for _n, f in METRIC_COLS] + [c.get(k, "") for k in probe_cols]
                  + [c.get("file", ""), c.get("error", "")]
                  + ([setting_ids.get(_setting_key(over, group_keys), "")] if group_keys else []))
        ws.append(values)
        ws.row_dimensions[r].height = row_height
        for col in range(1, len(headers) + 1):
            ws.cell(row=r, column=col).alignment = Alignment(wrap_text=True, vertical="top")
        image_path = os.path.join(sweep_dir, c["file"]) if c.get("file") else ""
        if image_path and os.path.isfile(image_path):
            try:
                with Image.open(image_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb_px, thumb_px))
                    buffer = io.BytesIO()
                    im.save(buffer, format="JPEG", quality=85)
                buffer.seek(0)
                picture = XLImage(buffer)
                ws.add_image(picture, f"B{r}")
            except Exception:
                pass

    widths = {"cell": 6, "image": thumb_px / 7 + 2, "file": 30, "error": 20, "LoRAs used": 45, "setting id": 10}
    widths.update({head: 13 for head in score_heads})
    for col, name in enumerate(headers, 1):
        if name in widths:
            width = widths[name]
        elif name in long_cols:
            width = 60
        elif name in RATING_COLS:
            width = 30 if name == "notes" else 11
        elif name in swept:
            width = min(40, max(10, max((len(str(_cell_value(name, (c.get("overrides") or {}).get(name, ""))))
                                         for c in cells), default=8) + 2))
        else:
            width = 10
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = "C2"

    _add_by_setting(wb, headers, cells, group_keys)

    about = wb.create_sheet("About")
    for line in info_text(manifest).splitlines():
        about.append([line])
    about.column_dimensions["A"].width = 140

    if os.path.isfile(path):
        previous = os.path.join(sweep_dir, names["review_prev"])
        try:
            if os.path.isfile(previous):
                os.remove(previous)
            os.replace(path, previous)
        except OSError as error:      # open in Excel: Windows locks it - say so instead of losing anything
            raise RuntimeError(f"{names['review']} is open or locked ({error}); close it and rebuild") from error
    wb.save(path)
    return path


def write_all(sweep_dir, manifest=None, log=print):
    """Both files; logs instead of raising (a sweep must never fail over its extras)."""
    try:
        manifest = _load(sweep_dir, manifest)
    except Exception as error:
        log(f"[sweep] sweep_info / review: manifest unreadable ({error})")
        return
    try:
        write_info(sweep_dir, manifest)
    except Exception as error:
        log(f"[sweep] sweep_info.txt not written: {error}")
    try:
        if write_review(sweep_dir, manifest) is None:
            log("[sweep] sweep_review.xlsx skipped: openpyxl is not installed")
    except Exception as error:
        log(f"[sweep] sweep_review.xlsx not written: {error}")


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        write_all(folder)
        print(f"done: {folder}")
