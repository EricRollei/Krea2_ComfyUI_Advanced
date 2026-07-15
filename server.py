# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric_Krea2 - server routes
==========================
Registers ComfyUI (aiohttp) endpoints used by the ★ Save Preset buttons and the
preset dropdowns:

    POST /eric_krea2/save_preset
    body: { "name": str, "section": "ultra"|"loader"|"lora", "data": {field: value} }

    GET  /eric_krea2/get_presets?section=ultra|loader|lora

Each section has its own library file in the package root
(``ultra_presets.json`` / ``loader_presets.json`` / ``lora_presets.json``),
shaped ``{ presetName: { <section>: {...fields...} } }``. Per-run values
(seed / prompt) and any ``*_preset`` control widget are stripped so a saved
preset stays a reusable recipe. Registering on import is a no-op if the ComfyUI
server isn't available (e.g. when the package is imported standalone).
"""

from __future__ import annotations

import json
import os
import re

# Preset sections -> their JSON library file (package root). Each file holds
# { presetName: { <section>: { ...field: value... } } }. One file per section
# preserves the original ultra_presets.json shape and back-compat.
_ALLOWED_SECTIONS = ("ultra", "loader", "lora", "sigmas", "apply_lora", "decode_vae")

# Never persist per-run values into a reusable recipe. Any "*_preset" control
# widget (ultra_preset / loader_preset / lora_preset ...) is stripped too.
# control_after_generate is ComfyUI's seed-control pseudo-widget, not a recipe field.
_STRIP_KEYS = {"seed", "prompt", "negative_prompt", "control_after_generate"}


def _presets_path(section: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"{section}_presets.json")


def _strip_run_keys(data: dict) -> dict:
    return {k: v for k, v in data.items()
            if k not in _STRIP_KEYS and not str(k).endswith("_preset")}


def _sanitize_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name or "").strip())
    # Reject path separators / control chars; keep it a plain library key.
    if not name or name in (".", "..") or re.search(r"[\\/\x00-\x1f]", name):
        return ""
    return name[:128]


def _load_raw(section: str) -> dict:
    try:
        with open(_presets_path(section), "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        # Corrupt file: don't clobber it silently; surface via empty + caller error.
        raise


def _save_preset(name: str, section: str, data: dict) -> dict:
    clean = _sanitize_name(name)
    if not clean:
        raise ValueError("invalid preset name")
    if section not in _ALLOWED_SECTIONS:
        raise ValueError(f"unsupported section '{section}' "
                         f"(allowed: {', '.join(_ALLOWED_SECTIONS)})")
    if not isinstance(data, dict):
        raise ValueError("data must be an object")

    fields = _strip_run_keys(data)

    lib = _load_raw(section)
    lib[clean] = {section: fields}
    path = _presets_path(section)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return {"ok": True, "name": clean, "count": len(fields), "section": section}


def _save_image_presets(image: str, name: str) -> dict:
    """Extract the KREA2_SETTINGS recipe embedded in an input-dir PNG and save each
    present section (loader / ultra / lora) into its preset library under `name`.
    The 'lora' LIST is converted to MultiLoRA widget slots. Returns which sections
    were saved (the image -> named-preset bridge behind the ★ button)."""
    clean = _sanitize_name(name)
    if not clean:
        raise ValueError("invalid preset name")

    from . import _settings
    try:
        import folder_paths
        path = folder_paths.get_annotated_filepath(image)
    except Exception as e:
        raise ValueError(f"cannot resolve image '{image}': {e}")
    if not path or not os.path.isfile(path):
        raise ValueError(f"image not found: {image}")

    blob = _settings.extract_from_png(path)
    if not isinstance(blob, dict) or not blob:
        raise ValueError("no KREA2_SETTINGS recipe embedded in that image")

    saved = []
    if isinstance(blob.get("loader"), dict):
        _save_preset(clean, "loader", blob["loader"])
        saved.append("loader")
    if isinstance(blob.get("ultra"), dict):
        _save_preset(clean, "ultra", blob["ultra"])
        saved.append("ultra")
    if "lora" in blob:
        slots = _settings.lora_list_to_slots(blob["lora"])
        if slots:
            _save_preset(clean, "lora", slots)
            saved.append("lora")

    if not saved:
        raise ValueError("recipe had no loader / ultra / lora sections to save")
    return {"ok": True, "name": clean, "saved": saved}


def _preview_sigmas(body):
    """Realize the Sigmas node's per-stage shapes for the on-node curve plot.

    Uses the exact same ``_sigmas.build_sigmas`` the Ultra node runs, so what the
    graph draws is what will actually sample. Input is the node's widget values
    (``s1_curve`` / ``s1_detail_bias`` / ``s1_rho`` / ``s1_alpha`` / ``s1_beta`` /
    ``s1_enable`` ...) plus optional per-stage run info read from the downstream
    Ultra node (``s1_steps`` / ``s1_start`` / ``s1_end``). Output is a realized
    descending sigma list per stage AT ITS REAL STEP COUNT, plus the executed
    ``[start, end]`` window so the plot can highlight the portion that runs.
    """
    from ._sigmas import build_sigmas

    def _clampi(v, lo, hi, default):
        try:
            v = int(v)
        except Exception:
            return default
        return max(lo, min(hi, v))

    g_steps = _clampi(body.get("steps", 24), 4, 200, 24)
    stages = {}
    for n in ("s1", "s2", "s3"):
        curve = str(body.get(f"{n}_curve", "linear"))
        bias = float(body.get(f"{n}_detail_bias", 0.0) or 0.0)
        rho_raw = body.get(f"{n}_rho", 0.0)
        rho = float(rho_raw) if rho_raw else None      # 0 => auto (from detail_bias)
        alpha = float(body.get(f"{n}_alpha", 0.5) or 0.5)
        beta = float(body.get(f"{n}_beta", 0.7) or 0.7)
        enabled = bool(body.get(f"{n}_enable", True))

        # Per-stage run window (from the connected Ultra node, if any). end<=0 is the
        # s1 "run to the end" sentinel; clamp everything into the realized schedule.
        steps = _clampi(body.get(f"{n}_steps", g_steps), 1, 200, g_steps)
        start = _clampi(body.get(f"{n}_start", 0), 0, max(0, steps - 1), 0)
        end_raw = _clampi(body.get(f"{n}_end", steps), 0, steps, steps)
        end = steps if end_raw <= 0 else max(start + 1, end_raw)

        try:
            sig = build_sigmas(steps, 1.0, curve, bias=bias, rho=rho, alpha=alpha, beta=beta)
        except Exception as e:  # pragma: no cover - a bad curve shouldn't kill the UI
            sig, curve = [], f"{curve} (err: {e})"
        stages[n] = {"enabled": enabled, "curve": curve, "sigmas": sig,
                     "steps": steps, "start": start, "end": end}
    return {"ok": True, "steps": g_steps, "stages": stages}


def _resolution_chain(body):
    """Compute the real S1->S2->S3->decode resolution chain for the Ultra node.

    Mirrors the exact sizing math the Ultra node runs (``_dims_from_ratio_mp``,
    ``_scaled_dims``, ``_round16`` + the upscale-VAE overrides) so the on-node
    readout matches what actually generates - the whole point is to expose the
    footgun where a connected upscale VAE silently ignores upscale_to_stage2/3
    (forced 4x area) or cascades to a huge final size. ``upscale_vae_connected``
    tells us whether the VAE paths can fire at all; when false we force disabled.
    Never fatal.
    """
    from .nodes.krea2_multistage_ultra import (
        _dims_from_ratio_mp, _scaled_dims, _round16, _UPSCALE_VAE_MODE_FLAGS)

    def _f(key, default):
        try:
            return float(body.get(key, default))
        except Exception:
            return float(default)

    aspect = str(body.get("aspect_ratio", "1:1 square"))
    s1_mp = _f("s1_megapixels", 3.0)
    width = int(_f("width", 0))
    height = int(_f("height", 0))
    up2 = _f("upscale_to_stage2", 0.0)
    up3 = _f("upscale_to_stage3", 0.0)
    connected = bool(body.get("upscale_vae_connected", False))
    mode = str(body.get("upscale_vae_mode", "disabled"))
    s1s2 = bool(body.get("s1_s2_upscale_vae", False))
    if not connected:
        mode, s1s2 = "disabled", False

    s2s3_vae, s2s3_down, final_vae, final_down = _UPSCALE_VAE_MODE_FLAGS.get(
        mode, (False, False, False, False))

    if width > 0 and height > 0:
        s1_w, s1_h = _round16(width), _round16(height)
    else:
        s1_w, s1_h = _dims_from_ratio_mp(aspect, s1_mp)

    do_s2 = up2 > 0
    do_s3 = do_s2 and up3 > 0
    use_s1s2_vae = s1s2 and do_s2

    def _mp(w, h):
        return round(w * h / 1e6, 2)

    chain = [{"stage": "S1", "w": s1_w, "h": s1_h, "mp": _mp(s1_w, s1_h), "via": "generate"}]

    if do_s2:
        if use_s1s2_vae:
            s2_w, s2_h = _round16(s1_w * 2), _round16(s1_h * 2)
            via2 = "VAE 2x (4x area; upscale_to_stage2 ignored)"
        else:
            s2_w, s2_h = _scaled_dims(s1_w, s1_h, up2)
            via2 = f"bislerp {up2:g}x area"
        chain.append({"stage": "S2", "w": s2_w, "h": s2_h, "mp": _mp(s2_w, s2_h), "via": via2})
    else:
        s2_w, s2_h = s1_w, s1_h

    if do_s3:
        use_inter = s2s3_vae  # needs 3 stages, which do_s3 guarantees
        if use_inter and s2s3_down:
            s3_w, s3_h = _scaled_dims(s2_w, s2_h, up3)
            via3 = f"VAE 2x -> downsample to {up3:g}x area"
        elif use_inter:
            s3_w, s3_h = _round16(s2_w * 2), _round16(s2_h * 2)
            via3 = "VAE 2x (4x area; upscale_to_stage3 ignored)"
        else:
            s3_w, s3_h = _scaled_dims(s2_w, s2_h, up3)
            via3 = f"bislerp {up3:g}x area"
        chain.append({"stage": "S3", "w": s3_w, "h": s3_h, "mp": _mp(s3_w, s3_h), "via": via3})
        last_w, last_h = s3_w, s3_h
    else:
        last_w, last_h = s2_w, s2_h

    if final_vae and final_down:
        dec = {"w": last_w, "h": last_h, "mp": _mp(last_w, last_h),
               "via": "VAE 2x decode -> downsample to native (supersample)"}
    elif final_vae:
        dec = {"w": last_w * 2, "h": last_h * 2, "mp": _mp(last_w * 2, last_h * 2),
               "via": "VAE 2x decode"}
    else:
        dec = {"w": last_w, "h": last_h, "mp": _mp(last_w, last_h), "via": "standard decode"}

    return {"ok": True, "chain": chain, "decode": dec, "mode": mode,
            "s1_s2_upscale_vae": use_s1s2_vae, "connected": connected}


def register_routes():
    """Attach the save-preset route to ComfyUI's PromptServer, if present."""
    try:
        from server import PromptServer  # ComfyUI
        from aiohttp import web
    except Exception:
        return False

    routes = PromptServer.instance.routes

    @routes.post("/eric_krea2/save_preset")
    async def _save_preset_route(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            result = _save_preset(
                body.get("name", ""),
                body.get("section", "ultra"),
                body.get("data", {}),
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:  # pragma: no cover
            return web.json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)
        return web.json_response(result)

    @routes.get("/eric_krea2/get_presets")
    async def _get_presets_route(request):
        # Returns the whole <section>_presets.json library so the JS side can apply
        # a selected preset's values into the node's own widgets on selection
        # (fixes: dropdown showing a name but the panel/generate() silently
        # disagreeing about what will actually run).
        section = request.rel_url.query.get("section", "ultra")
        if section not in _ALLOWED_SECTIONS:
            return web.json_response(
                {"ok": False, "error": f"unsupported section '{section}' "
                                       f"(allowed: {', '.join(_ALLOWED_SECTIONS)})"}, status=400)
        try:
            lib = _load_raw(section)
        except Exception as e:  # pragma: no cover
            return web.json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)
        return web.json_response({"ok": True, "presets": lib, "section": section})

    @routes.post("/eric_krea2/save_image_presets")
    async def _save_image_presets_route(request):
        # ★ Save recipe -> presets: extract the recipe embedded in a saved image and
        # drop each section into its preset library under the given name.
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            result = _save_image_presets(body.get("image", ""), body.get("name", ""))
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:  # pragma: no cover
            return web.json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)
        return web.json_response(result)

    @routes.post("/eric_krea2/preview_sigmas")
    async def _preview_sigmas_route(request):
        # Realize the Sigmas node's per-stage curves for the on-node plot (same math
        # the Ultra node samples with). Never fatal: a bad shape returns empty sigmas.
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            result = _preview_sigmas(body)
        except Exception as e:  # pragma: no cover
            return web.json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)
        return web.json_response(result)

    @routes.post("/eric_krea2/resolution_chain")
    async def _resolution_chain_route(request):
        # Realize the Ultra node's S1->S2->S3->decode resolution chain for the on-node
        # readout (same sizing math generate() runs, incl. upscale-VAE overrides).
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            result = _resolution_chain(body)
        except Exception as e:  # pragma: no cover
            return web.json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)
        return web.json_response(result)

    print("[Eric_Krea2] registered POST /eric_krea2/save_preset, "
          "POST /eric_krea2/save_image_presets, GET /eric_krea2/get_presets, "
          "POST /eric_krea2/preview_sigmas, POST /eric_krea2/resolution_chain")
    return True
