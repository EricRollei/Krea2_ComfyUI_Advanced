# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Sweep engine (package-agnostic)
===============================
Runs a plan of settings combos through a host generator, scores each result with
cheap technical metrics (CPU, milliseconds), writes metadata-stamped cell PNGs +
a labeled contact sheet + a JSON manifest, and survives interrupts/failures.

Deliberately knows NOTHING about Krea2 (or any package): the host supplies
callables. Porting to another node set (qwen, ...) = new plan node + adapter,
this file unchanged.

Host contract (all images are numpy float32 HxWx3 in 0..1):
    run_one(overrides: dict) -> np.ndarray          # generate one cell
        (or (np.ndarray, dict) - the dict is merged into that cell's
        manifest record, e.g. {"s1_reused": True})
    resolve_values(overrides: dict) -> dict         # full resolved recipe (manifest)
    settings_string(resolved: dict) -> str          # recipe JSON for the PNG chunk
Optional:
    interrupt_check() -> None (raises interrupt_exc) # called between cells
    interrupt_exc: BaseException subclass            # finalize-then-reraise
    progress(i, n) -> None
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import traceback

import numpy as np

SWEEP_MANIFEST_VERSION = 1

METRIC_KEYS = ("sharpness_laplacian", "sharpness_tenengrad", "noise_sigma",
               "flat_chroma_var", "clip_fraction")
# Direction for "best-first" sorting per metric (True = higher is better).
METRIC_HIGHER_BETTER = {"sharpness_laplacian": True, "sharpness_tenengrad": True,
                        "noise_sigma": False, "flat_chroma_var": False,
                        "clip_fraction": False, "time_s": False}


# ── metrics (numpy only; no torch / skimage / model loads) ───────────────────

def _luma(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])


def _conv3(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """3x3 valid convolution via shifted slices (fast enough, no scipy)."""
    h, w = x.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            kv = k[dy, dx]
            if kv != 0.0:
                out += kv * x[dy:h - 2 + dy, dx:w - 2 + dx]
    return out


_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
_SOBX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBY = _SOBX.T.copy()


def compute_metrics(img: np.ndarray) -> dict:
    """Technical-quality metrics on a float32 HxWx3 image in 0..1.

    sharpness_laplacian : variance of the Laplacian of luma (classic focus
                          measure; oversharpening halos inflate it).
    sharpness_tenengrad : mean squared Sobel gradient magnitude (rewards real
                          edge energy; less halo-sensitive than the Laplacian,
                          so the PAIR disagreeing flags oversharpening).
    noise_sigma         : robust noise std estimate from the finest Haar
                          diagonal band (median|HH|/0.6745) - texture-resistant.
    flat_chroma_var     : chroma variance restricted to low-gradient regions -
                          the splotch/residue detector (color mottle in what
                          should be flat areas).
    clip_fraction       : fraction of channel samples at 0/255 (blowout/crush).
    Metrics are computed on a <=1MP downsample-by-striding for speed EXCEPT the
    noise estimate, which uses the native-res crop center (decimation aliases
    noise). Values are comparable WITHIN a sweep (same content), not across
    prompts.
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    # noise on native pixels (center crop up to 1024^2 for speed)
    ch, cw = min(h, 1024), min(w, 1024)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    ly_n = _luma(img[y0:y0 + ch, x0:x0 + cw]) * 255.0
    d = (ly_n[0::2, 0::2][:ch // 2, :cw // 2] - ly_n[0::2, 1::2][:ch // 2, :cw // 2]
         - ly_n[1::2, 0::2][:ch // 2, :cw // 2] + ly_n[1::2, 1::2][:ch // 2, :cw // 2]) * 0.5
    noise_sigma = float(np.median(np.abs(d)) / 0.6745)

    stride = max(1, int(math.ceil(math.sqrt((h * w) / 1_048_576))))
    s = img[::stride, ::stride]
    ly = _luma(s) * 255.0
    lap = _conv3(ly, _LAP)
    gx, gy = _conv3(ly, _SOBX), _conv3(ly, _SOBY)
    gm = gx * gx + gy * gy
    sharp_lap = float(lap.var())
    sharp_ten = float(gm.mean())

    thr = np.percentile(gm, 25.0)
    flat = gm <= thr
    chroma = (s.max(axis=-1) - s.min(axis=-1))[1:-1, 1:-1] * 255.0
    flat_chroma_var = float(chroma[flat].var()) if flat.any() else 0.0

    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    clip_fraction = float(((u8 <= 1) | (u8 >= 254)).mean())

    return {"sharpness_laplacian": round(sharp_lap, 2),
            "sharpness_tenengrad": round(sharp_ten, 2),
            "noise_sigma": round(noise_sigma, 3),
            "flat_chroma_var": round(flat_chroma_var, 2),
            "clip_fraction": round(clip_fraction, 5)}


# ── grid expansion (generic helper for plan nodes) ───────────────────────────

def expand_grid(axes: dict) -> list:
    """Cartesian product of {axis_name: [ {override-dict}, ... ]} entries.
    Each axis value is already a partial override dict (the plan node does the
    key mapping); axes with an empty list are skipped. Returns merged dicts in
    a stable (first axis slowest) order."""
    combos = [dict()]
    for name, values in axes.items():
        if not values:
            continue
        combos = [dict(c, **v) for c in combos for v in values]
    return combos


def diff_label(overrides: dict) -> str:
    """Human caption for a cell: the swept keys only, compactly."""
    parts = []
    for k in sorted(overrides.keys()):
        if k.startswith("_"):
            continue
        v = overrides[k]
        if isinstance(v, float):
            v = f"{v:g}"
        parts.append(f"{k}={v}")
    return "  ".join(parts) if parts else "(base)"


# ── contact sheet ────────────────────────────────────────────────────────────

def _load_font(px: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "seguisb.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(text: str, draw, font, max_w: int) -> list:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines[:5]


def render_contact_sheet(cells: list, tile_px: int, header: str) -> np.ndarray:
    """cells: [{image: np HxWx3 float 0..1 | None, caption: str, note: str}].
    Returns the sheet as float32 HxWx3 in 0..1."""
    from PIL import Image, ImageDraw
    n = max(1, len(cells))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    font = _load_font(max(14, tile_px // 24))
    hfont = _load_font(max(16, tile_px // 20))
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    lh = max(17, tile_px // 20)

    tiles = []
    for c in cells:
        im = c.get("image")
        if im is None:
            t = Image.new("RGB", (tile_px, int(tile_px * 0.6)), (40, 24, 24))
            d = ImageDraw.Draw(t)
            for i, ln in enumerate(_wrap("FAILED: " + c.get("note", ""), probe, font, tile_px - 12)):
                d.text((6, 6 + i * lh), ln, fill=(255, 120, 120), font=font)
        else:
            u8 = np.clip(np.asarray(im, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            t = Image.fromarray(u8)
            t.thumbnail((tile_px, tile_px * 4), Image.LANCZOS)
        cap_lines = _wrap(c.get("caption", ""), probe, font, tile_px - 8)
        note = c.get("note", "")
        tiles.append((t, cap_lines, note))

    cap_h = (max(len(cl) for _, cl, _ in tiles) + 1) * lh + 10
    tile_h = max(t.height for t, _, _ in tiles)
    head_h = lh + 14
    W = cols * (tile_px + 8) + 8
    H = head_h + rows * (tile_h + cap_h + 8) + 8
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), header, fill=(230, 230, 230), font=hfont)
    for i, (t, cap_lines, note) in enumerate(tiles):
        r, cc = divmod(i, cols)
        x = 8 + cc * (tile_px + 8) + (tile_px - t.width) // 2
        y = head_h + 8 + r * (tile_h + cap_h + 8)
        sheet.paste(t, (x, y))
        ty = y + tile_h + 4
        tx = 8 + cc * (tile_px + 8)
        for j, ln in enumerate(cap_lines):
            d.text((tx + 2, ty + j * lh), ln, fill=(210, 210, 210), font=font)
        if note:
            d.text((tx + 2, ty + len(cap_lines) * lh), note,
                   fill=(150, 200, 150), font=font)
    return np.asarray(sheet, dtype=np.float32) / 255.0


# ── main loop ────────────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")[:48] or "model"


def _fname_tag(overrides: dict) -> str:
    """Compact value-only tag for cell filenames: '_deis_3m_beta57_10'.
    Keys are dropped (they repeat per sweep); values keep sorted-key order so
    filenames align with captions/manifest. Capped for filesystem sanity."""
    vals = [f"{v:g}" if isinstance(v, float) else str(v)
            for k, v in sorted(overrides.items()) if not str(k).startswith("_")]
    tag = _slug("_".join(vals))[:48].rstrip("_.")
    return f"_{tag}" if tag else ""


def run_sweep(plan: dict, *, run_one, resolve_values, settings_string,
              out_root: str, checkpoint_name: str, base_seed: int,
              interrupt_check=None, interrupt_exc=None, progress=None,
              log=print) -> dict:
    """Execute the plan. Returns
        {cells, results, sheet, manifest_path, sweep_dir, incomplete}
    where cells is aligned with results (None image for failed cells).
    On interrupt: files + manifest are finalized (incomplete=True), then the
    interrupt is re-raised so the queue stops normally - disk artifacts survive."""
    combos = list(plan.get("combos", []))
    labels = plan.get("labels") or [diff_label(c) for c in combos]
    n = len(combos)
    sweep_id = time.strftime("%Y%m%d_%H%M%S") + "_" + _slug(checkpoint_name)
    sweep_dir = os.path.join(out_root, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)

    quick = bool(plan.get("quick_screen"))
    screen_mp = float(plan.get("screen_megapixels", 1.0))
    seeds = plan.get("seeds") or []
    score = bool(plan.get("score_metrics", True))

    results, cells, incomplete = [], [], False
    interrupted = None
    t_sweep = time.time()
    for i, (ov, lab) in enumerate(zip(combos, labels)):
        if interrupt_check is not None:
            try:
                interrupt_check()
            except BaseException as e:  # finalize below, then re-raise
                interrupted, incomplete = e, True
                log(f"[sweep] interrupted before cell {i}/{n}; finalizing partial results")
                break
        eff = dict(ov)
        if quick:
            eff["upscale_to_stage2"] = 0.0
            eff["upscale_to_stage3"] = 0.0
            if "s1_megapixels" not in ov:  # a swept megapixels axis wins
                eff["s1_megapixels"] = screen_mp
        seed = int(seeds[i]) if i < len(seeds) else int(base_seed)
        if plan.get("seed_policy") == "seeds_axis" and "_seed" in ov:
            seed = int(eff.pop("_seed"))
        eff.pop("_coerced_noise", None)
        eff["seed"] = seed
        coerced = bool(ov.get("_coerced_noise"))
        log(f"[sweep {i + 1}/{n}] {lab}" + ("  (noise->white for lcm)" if coerced else ""))
        if progress is not None:
            progress(i, n)
        rec = {"index": i, "overrides": {k: v for k, v in ov.items() if not k.startswith('_')},
               "label": lab, "seed": seed, "coerced_noise_white": coerced,
               "quick_screen": quick}
        t0 = time.time()
        try:
            img = run_one(eff)
            if isinstance(img, tuple):
                img, _extra = img
                if isinstance(_extra, dict):
                    rec.update(_extra)
            img = np.asarray(img, dtype=np.float32)
            rec["time_s"] = round(time.time() - t0, 2)
            rec["resolved"] = resolve_values(eff)
            if score:
                rec["metrics"] = compute_metrics(img)
            fname = f"cell_{i:03d}{_fname_tag(rec['overrides'])}.png"
            _save_png(os.path.join(sweep_dir, fname), img,
                      settings_string(rec["resolved"]), json.dumps(rec["overrides"]))
            rec["file"] = fname
            cells.append(img)
        except BaseException as e:
            if interrupt_exc is not None and isinstance(e, interrupt_exc):
                interrupted, incomplete = e, True
                rec["error"] = "interrupted mid-cell"
                results.append(rec)
                cells.append(None)
                log(f"[sweep] interrupted during cell {i}; finalizing partial results")
                break
            rec["time_s"] = round(time.time() - t0, 2)
            rec["error"] = f"{type(e).__name__}: {e}"
            log(f"[sweep {i + 1}/{n}] FAILED: {rec['error']}")
            traceback.print_exc()
            cells.append(None)
        results.append(rec)

    # order for the sheet
    sort_by = plan.get("sort_by", "plan_order")
    order = list(range(len(results)))
    if sort_by != "plan_order" and results:
        hb = METRIC_HIGHER_BETTER.get(sort_by, True)

        def keyf(j):
            r = results[j]
            v = r.get("metrics", {}).get(sort_by, r.get(sort_by))
            if v is None:
                return math.inf
            return -v if hb else v
        order = sorted(order, key=keyf)

    sheet_cells = []
    for j in order:
        r = results[j]
        note = ""
        if "metrics" in r:
            m = r["metrics"]
            note = (f"{r.get('time_s', 0):.1f}s | sharp {m['sharpness_laplacian']:.0f}/"
                    f"{m['sharpness_tenengrad']:.0f}  noise {m['noise_sigma']:.1f}  "
                    f"splotch {m['flat_chroma_var']:.0f}")
        elif "time_s" in r:
            note = f"{r['time_s']:.1f}s"
        sheet_cells.append({"image": cells[j] if j < len(cells) else None,
                            "caption": f"#{r['index']}  {r['label']}",
                            "note": note or r.get("error", "")})
    header = (f"{sweep_id}   {len([r for r in results if 'file' in r])}/{n} cells   "
              f"sort: {sort_by}" + ("   QUICK-SCREEN" if quick else "")
              + ("   INCOMPLETE" if incomplete else ""))
    sheet = render_contact_sheet(sheet_cells, int(plan.get("tile_px", 448)), header) \
        if sheet_cells else np.zeros((64, 64, 3), np.float32)
    _save_png(os.path.join(sweep_dir, "contact_sheet.png"), sheet, None, None)

    manifest = {"sweep_manifest_version": SWEEP_MANIFEST_VERSION,
                "sweep_id": sweep_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint": checkpoint_name, "incomplete": incomplete,
                "total_s": round(time.time() - t_sweep, 1),
                "plan": {k: v for k, v in plan.items() if k != "combos"},
                "n_planned": n, "sort_by": sort_by, "order": order,
                "base": resolve_values({}), "cells": results}
    manifest_path = os.path.join(sweep_dir, "sweep_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    # Human-readable summary table (opens straight into a spreadsheet). One row
    # per cell: what varied, per-cell generation time, rounded scores.
    swept_cols = sorted({k for r in results for k in r.get("overrides", {})})
    with open(os.path.join(sweep_dir, "sweep_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "file", "gen_time_s"] + swept_cols
                   + ["sharp_lap", "sharp_ten", "noise", "splotch", "clip_pct",
                      "s1_reused", "error"])
        for r in results:
            m = r.get("metrics") or {}
            w.writerow([r["index"], r.get("file", ""), r.get("time_s", "")]
                       + [r.get("overrides", {}).get(k, "") for k in swept_cols]
                       + [round(m["sharpness_laplacian"]) if m else "",
                          round(m["sharpness_tenengrad"]) if m else "",
                          round(m.get("noise_sigma", 0), 1) if m else "",
                          round(m.get("flat_chroma_var", 0)) if m else "",
                          round(100 * m.get("clip_fraction", 0), 2) if m else "",
                          r.get("s1_reused", ""), r.get("error", "")])
    log(f"[sweep] {sweep_dir}: {len([r for r in results if 'file' in r])}/{n} ok, "
        f"manifest + contact sheet written ({manifest['total_s']}s)")
    if interrupted is not None:
        raise interrupted
    return {"cells": cells, "results": results, "sheet": sheet,
            "manifest_path": manifest_path, "sweep_dir": sweep_dir,
            "incomplete": incomplete}


def _save_png(path: str, img: np.ndarray, settings_str, overrides_json):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    u8 = np.clip(np.asarray(img, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    meta = PngInfo()
    if settings_str:
        meta.add_text("krea2_settings", settings_str)  # readable by Settings From Image
    if overrides_json:
        meta.add_text("sweep_cell", overrides_json)
    Image.fromarray(u8).save(path, pnginfo=meta)
