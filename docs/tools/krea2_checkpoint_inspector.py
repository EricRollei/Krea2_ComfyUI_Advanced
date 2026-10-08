# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Krea2 checkpoint inspector (read-only, CPU, headers only)
=========================================================
Scans Krea2 transformer checkpoints (.safetensors / .gguf) and reports, per file:

  * key layout       comfy (model.diffusion_model.*) | bare | diffusers
  * quant formats    per-layer formats from comfy_quant markers + file metadata,
                     or inferred from dtypes / *_scale companions for legacy files
  * shape check      every transformer weight vs a reference bf16 checkpoint
  * NOW verdict      will the CURRENT component loader load it correctly?
  * FIX verdict      will it load after the planned comfy_kitchen dequant dispatch?

Nothing is loaded to the GPU and no weight data is read for safetensors - only the
JSON header and the tiny '.comfy_quant' marker tensors. For GGUF, the smallest
tensor of each quant type is test-dequantized to prove the numpy path handles it.

Usage (embedded python):
  python krea2_checkpoint_inspector.py [folder_or_file ...] [--ref REF.safetensors]
                                       [--csv OUT.csv]
Defaults: folder A:/Models/diffusion_models/Krea2, ref krea2_turbo_bf16.safetensors
inside it, csv next to this script.

Author: Eric Hiss (GitHub: EricRollei)
"""

import argparse
import csv
import json
import os
import struct
import sys
from collections import Counter

DEFAULT_DIR = r"A:/Models/diffusion_models/Krea2"
DEFAULT_REF = "krea2_turbo_bf16.safetensors"

# Formats the CURRENT loader dequantizes correctly (per-tensor / per-row scale,
# optional ConvRot un-rotation). Everything in KITCHEN_FORMATS is handled by the
# planned comfy_kitchen dispatch (mirrors comfy/ops.py + quant_ops.QUANT_ALGOS).
NOW_FORMATS = {
    "plain",
    "fp8_bare",
    "fp8_scaled",
    "int8_scaled",
    "float8_e4m3fn",
    "float8_e5m2",
    "int8_tensorwise",
}
KITCHEN_FORMATS = NOW_FORMATS | {
    "mxfp8",
    "nvfp4",
    "convrot_w4a4",
    "asym_w4a8_int8",
    "w6a8_int8",
}
AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".comfy_quant",
    ".pre_quant_scale",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
)


def strip_prefix(name):
    for pre in ("model.diffusion_model.", "diffusion_model.", "model."):
        if name.startswith(pre):
            return name[len(pre) :]
    return name


def read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def layer_confs(path, hdr):
    """{module (prefix-stripped): conf} from file metadata + comfy_quant markers."""
    confs = {}
    meta = hdr.get("__metadata__") or {}
    qm = meta.get("_quantization_metadata")
    if qm:
        try:
            for k, c in (json.loads(qm).get("layers") or {}).items():
                if isinstance(c, dict):
                    confs[strip_prefix(k)] = c
        except Exception:
            pass
    markers = [k for k in hdr if k.endswith(".comfy_quant")]
    if markers:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            for k in markers:
                try:
                    v = f.get_tensor(k).flatten()
                    if str(v.dtype) != "torch.uint8":
                        v = (
                            v.float()
                            .round()
                            .clamp(0, 255)
                            .to(dtype=__import__("torch").uint8)
                        )
                    c = json.loads(bytes(v.tolist()).decode("utf-8"))
                    if isinstance(c, dict):
                        confs.setdefault(strip_prefix(k[: -len(".comfy_quant")]), c)
                except Exception:
                    pass
    return confs


def ref_shapes(ref_path):
    hdr = read_st_header(ref_path)
    return {
        strip_prefix(k): tuple(v["shape"])
        for k, v in hdr.items()
        if k != "__metadata__"
    }


def classify_weight(key, info, hdr, conf):
    """-> (format, scale_shape or None)"""
    dt = info["dtype"]
    scale = hdr.get(key + "_scale")
    sshape = tuple(scale["shape"]) if scale else None
    if conf and conf.get("format"):
        return conf["format"], sshape
    if dt in ("BF16", "F16", "F32"):
        return "plain", sshape
    if dt.startswith("F8"):
        return ("fp8_scaled" if scale else "fp8_bare"), sshape
    if dt == "I8":
        return ("int8_scaled" if scale else "int8_noscale"), sshape
    return f"unknown({dt})", sshape


def inspect_safetensors(path, ref):
    hdr = read_st_header(path)
    keys = [k for k in hdr if k != "__metadata__"]
    if any(k.startswith("model.diffusion_model.") for k in keys):
        layout = "comfy"
    elif any(k.startswith("transformer_blocks.") for k in keys):
        layout = "diffusers"
    else:
        layout = "bare"
    confs = layer_confs(path, hdr)
    fmts = Counter()
    now_fail, fix_fail, notes = [], [], []

    weights = {}
    for k in keys:
        if k.endswith(AUX_SUFFIXES):
            continue
        weights[strip_prefix(k)] = (k, hdr[k])

    if layout == "diffusers":
        notes.append("diffusers layout - shape check skipped (remap passes it through)")
    else:
        missing = [k for k in ref if k not in weights]
        if missing:
            m = f"{len(missing)} missing key(s) e.g. {missing[:3]}"
            now_fail.append(m)
            fix_fail.append(m)
        extra = [k for k in weights if k not in ref]
        if extra:
            notes.append(f"{len(extra)} extra key(s) e.g. {extra[:3]}")

    for sk, (k, info) in weights.items():
        if not k.endswith(".weight") and info["dtype"] in ("BF16", "F16", "F32"):
            continue
        conf = confs.get(sk[: -len(".weight")]) if sk.endswith(".weight") else None
        fmt, sshape = classify_weight(k, info, hdr, conf)
        fmts[fmt] += 1
        wshape = tuple(info["shape"])
        if fmt not in NOW_FORMATS:
            now_fail.append(f"{fmt} unsupported ({sk})")
        elif sshape is not None:
            ok = (
                len(sshape) == 0
                or sshape == (1,)
                or (len(wshape) == 2 and sshape == (wshape[0], 1))
            )
            if not ok:
                now_fail.append(
                    f"{fmt} scale {sshape} not per-tensor/row for {wshape} ({sk})"
                )
        if fmt == "int8_noscale":
            now_fail.append(f"int8 without scale ({sk})")
            fix_fail.append(f"int8 without scale ({sk})")
        if fmt not in KITCHEN_FORMATS and fmt != "int8_noscale":
            fix_fail.append(f"{fmt} unknown to comfy_kitchen ({sk})")
        if conf and (conf.get("convrot") or (conf.get("params") or {}).get("convrot")):
            gs = int(
                conf.get(
                    "convrot_groupsize",
                    (conf.get("params") or {}).get("convrot_groupsize", 256),
                )
            )
            if len(wshape) == 2 and wshape[1] % gs:
                notes.append(f"ConvRot gs {gs} !| in-dim {wshape[1]} ({sk})")
        ref_shape = ref.get(sk)
        packed = fmt in ("nvfp4", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8")
        if (
            layout != "diffusers"
            and ref_shape is not None
            and wshape != ref_shape
            and not packed
        ):
            m = f"shape {wshape} != ref {ref_shape} ({sk})"
            now_fail.append(m)
            fix_fail.append(m)

    meta = hdr.get("__metadata__") or {}
    conv = meta.get("converted_by") or meta.get("quant_format") or ""
    return layout, fmts, now_fail, fix_fail, notes, conv


def inspect_gguf(path, ref):
    import numpy as np
    import gguf
    from gguf.quants import dequantize

    r = gguf.GGUFReader(path)
    arch = None
    try:
        fld = r.get_field("general.architecture")
        if fld is not None:
            arch = str(bytes(fld.parts[fld.data[-1]]), encoding="utf-8")
    except Exception:
        pass
    fmts = Counter()
    smallest = {}
    names = {}
    for t in r.tensors:
        tn = t.tensor_type.name
        fmts[f"gguf_{tn}"] += 1
        names[strip_prefix(t.name)] = tuple(int(d) for d in reversed(t.shape))
        if tn not in ("F16", "F32") and (
            tn not in smallest or t.n_elements < smallest[tn].n_elements
        ):
            smallest[tn] = t
    now_fail, notes = [], [f"arch={arch}"]
    for tn, t in smallest.items():
        try:
            arr = dequantize(t.data, t.tensor_type)
            if not np.isfinite(np.asarray(arr, dtype=np.float32)).all():
                now_fail.append(f"{tn} dequant produced non-finite values")
        except Exception as e:
            now_fail.append(f"{tn} numpy dequant failed: {type(e).__name__}: {e}")
    missing = [k for k in ref if k not in names]
    # comfy stores per-block modulation flat; ref is also comfy-bare so names line up
    if missing:
        now_fail.append(f"{len(missing)} missing key(s) e.g. {missing[:3]}")
    bad = [k for k, s in names.items() if k in ref and s != ref[k]]
    if bad:
        now_fail.append(
            f"{len(bad)} shape mismatch(es) e.g. {bad[0]} {names[bad[0]]} != {ref[bad[0]]}"
        )
    return "gguf", fmts, now_fail, list(now_fail), notes, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[DEFAULT_DIR])
    ap.add_argument("--ref", default=None)
    ap.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "krea2_inspector_report.csv"
        ),
    )
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                files += [
                    os.path.join(root, f)
                    for f in fs
                    if f.endswith((".safetensors", ".gguf"))
                ]
        else:
            files.append(p)
    ref_path = a.ref or os.path.join(
        a.paths[0] if os.path.isdir(a.paths[0]) else DEFAULT_DIR, DEFAULT_REF
    )
    ref = {k: v for k, v in ref_shapes(ref_path).items()}
    print(f"reference: {os.path.basename(ref_path)} ({len(ref)} keys)\n")

    rows = []
    for path in sorted(files, key=lambda x: os.path.basename(x).lower()):
        name = os.path.basename(path)
        try:
            if path.endswith(".gguf"):
                res = inspect_gguf(path, ref)
            else:
                res = inspect_safetensors(path, ref)
            layout, fmts, now_fail, fix_fail, notes, conv = res
            now = "OK" if not now_fail else "FAIL"
            fix = "OK" if not fix_fail else "FAIL"
            why = (now_fail[0] if now_fail else "") + (
                f" (+{len(now_fail) - 1} more)" if len(now_fail) > 1 else ""
            )
        except Exception as e:
            layout, fmts, notes, conv = "?", Counter(), [], ""
            now = fix = "ERROR"
            why = f"{type(e).__name__}: {e}"
        fstr = ", ".join(f"{k}x{v}" for k, v in fmts.most_common())
        gb = os.path.getsize(path) / 1e9
        rows.append(
            [name, f"{gb:.1f}", layout, fstr, now, fix, why, "; ".join(notes[:3]), conv]
        )
        print(f"{now:5} -> {fix:5} {gb:5.1f}GB {layout:9} {name}")
        print(f"      formats: {fstr}")
        if why:
            print(f"      why: {why}")
        for n in notes[:3]:
            print(f"      note: {n}")

    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "GB",
                "layout",
                "formats",
                "now",
                "after_fix",
                "why",
                "notes",
                "converter",
            ]
        )
        w.writerows(rows)
    print(
        f"\n{sum(r[4] == 'OK' for r in rows)}/{len(rows)} load correctly now; "
        f"{sum(r[5] == 'OK' for r in rows)}/{len(rows)} after the planned fix. CSV: {a.csv}"
    )


if __name__ == "__main__":
    sys.exit(main())
