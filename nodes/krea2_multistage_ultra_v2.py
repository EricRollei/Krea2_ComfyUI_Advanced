# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Multi-Stage Ultra V2 (presets)
=========================================
A thin wrapper around :class:`EricKrea2MultistageUltra` that adds the shared
``KREA2_SETTINGS`` recipe system (see Krea2_Settings_Preset_Spec):

  * ``settings`` output - pretty-printed JSON of this node's ``ultra`` section,
    wrapped in the top-level schema. Feed it into a Save Image node to embed the
    recipe in the PNG, or into a merge node. The same serializer backs saved
    presets, so a PNG's recipe and a saved preset are interchangeable.
  * ``ultra_preset`` dropdown - load a named recipe from ``ultra_presets.json``
    and override the panel values for that run (partial presets supported).
  * ★ Save Preset button (JS) + ``/eric_krea2/save_preset`` endpoint write the
    current widget values into ``ultra_presets.json`` under a name.

The heavy multistage engine is inherited unchanged; nothing in the shipping v1
node is disturbed. The authoritative ``ultra`` field list is derived live from
the inherited ``INPUT_TYPES`` (never hardcoded), minus per-run / non-serializable
inputs (pipeline, VAE connections, prompt, seed).
"""

from __future__ import annotations

import json
import os

from .krea2_multistage_ultra import EricKrea2MultistageUltra

# ComfyUI primitive widget types that serialize cleanly to JSON.
_PRIMITIVE_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}

# Keys excluded from the recipe: object connections (not JSON), the preset
# control itself, and per-run values (prompt/seed are not part of a reusable
# recipe - this also keeps the metadata string == saved preset content).
_SETTINGS_EXCLUDE = {
    "krea2_pipeline", "ultra_preset",
    "prompt", "negative_prompt", "seed",
    "upscale_vae", "decode_vae",
}

KREA2_SETTINGS_VERSION = 1


def _ultra_presets_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "ultra_presets.json")


def _load_ultra_presets() -> dict:
    """Tolerant loader mirroring ``_load_cond_presets``: returns ``{}`` on any
    error and ignores non-dict entries."""
    try:
        with open(_ultra_presets_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if isinstance(v, dict)} if isinstance(d, dict) else {}
    except Exception:
        return {}


class EricKrea2MultistageUltraV2(EricKrea2MultistageUltra):
    """Ultra multistage generate + save/load presets + a settings metadata output."""

    @classmethod
    def INPUT_TYPES(cls):
        # Start from the inherited schema so we always track field changes, then
        # inject the ultra_preset dropdown at the top of `optional`.
        base = EricKrea2MultistageUltra.INPUT_TYPES()
        preset_names = ["custom"] + sorted(_load_ultra_presets().keys())
        preset_widget = {
            "ultra_preset": (preset_names, {"default": "custom",
                "tooltip": "Load a saved recipe from ultra_presets.json and override the panel "
                           "values for this run (partial presets only override the fields they "
                           "contain). 'custom' = use the panel as-is. Use the ★ Save Preset button "
                           "to add one; the dropdown refreshes after a graph reload."}),
        }
        merged_optional = {}
        merged_optional.update(preset_widget)
        merged_optional.update(base.get("optional", {}))
        # Sweep input appended LAST (2026-07-24, spec_sweep_system_v2_final): a
        # connection-type socket (no widget value), so existing saved workflows'
        # positional widget_values are untouched either way.
        merged_optional["sweep"] = ("KREA2_SWEEP", {
            "tooltip": "Optional sweep plan (from Eric Krea2 Sweep Plan). When connected, this "
                       "run executes EVERY combo in the plan inside one queue item (model + "
                       "LoRA state stay loaded), saves metadata-stamped cell PNGs + a labeled "
                       "contact sheet + sweep_manifest.json under output/sweeps/<id>/, and the "
                       "image output becomes the batch of cells. Panel values are the base "
                       "recipe; each combo overrides only its swept keys."})
        base["optional"] = merged_optional
        return base

    # Append `settings` as the LAST output so existing wired outputs keep index.
    # Sweep outputs appended AFTER settings (2026-07-24) for the same reason:
    # every pre-existing output keeps its index. Non-sweep runs emit a tiny
    # black sweep_sheet and an empty manifest path.
    RETURN_TYPES = EricKrea2MultistageUltra.RETURN_TYPES + ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = EricKrea2MultistageUltra.RETURN_NAMES + ("settings", "sweep_sheet", "sweep_manifest")
    FUNCTION = "generate"
    CATEGORY = "Eric/Krea2"

    # ── settings schema ────────────────────────────────────────────────────
    @classmethod
    def ultra_setting_keys(cls):
        """The serializable ``ultra`` recipe keys, derived live from INPUT_TYPES."""
        keys = []
        it = cls.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, spec in it.get(section, {}).items():
                if name in _SETTINGS_EXCLUDE:
                    continue
                t = spec[0] if isinstance(spec, (tuple, list)) and spec else spec
                if isinstance(t, list):            # combo widget
                    keys.append(name)
                elif isinstance(t, str) and t in _PRIMITIVE_TYPES:
                    keys.append(name)
        return keys

    @classmethod
    def serialize_settings(cls, values: dict) -> str:
        """Serialize the ``ultra`` section from a values dict (pretty JSON).
        The same function backs both the metadata output and saved presets."""
        keys = cls.ultra_setting_keys()
        ultra = {k: values[k] for k in keys if k in values}
        blob = {"krea2_settings_version": KREA2_SETTINGS_VERSION, "ultra": ultra}
        return json.dumps(blob, indent=2, ensure_ascii=False)

    # ── run ────────────────────────────────────────────────────────────────
    def generate(self, **kwargs):
        # 1) Apply a named preset (before anything reads the values).
        preset = str(kwargs.pop("ultra_preset", "custom") or "custom")
        if preset and preset != "custom":
            entry = _load_ultra_presets().get(preset, {})
            ultra = entry.get("ultra", entry) if isinstance(entry, dict) else {}
            valid = set(self.ultra_setting_keys())
            applied = 0
            for k, v in (ultra or {}).items():
                if k in valid:
                    kwargs[k] = v
                    applied += 1
            print(f"[EricKrea2-MS] ultra_preset '{preset}': applied {applied} field(s).")

        # 2) Build the settings string from the (possibly overridden) values,
        #    BEFORE the base pops cond_* out of kwargs.
        settings = self.serialize_settings(kwargs)

        # 2b) If a sigma-shape bundle is wired in, fold its enabled stages into the
        #     recipe as a "sigmas" section (so the master image's metadata carries
        #     the authored curves alongside the ultra fields).
        sig_bundle = kwargs.get("sigmas")
        if isinstance(sig_bundle, dict):
            section = {k: v for k, v in sig_bundle.items()
                       if k in ("s1", "s2", "s3") and isinstance(v, dict) and v.get("enabled", True)}
            if section:
                from .. import _settings
                settings = _settings.merge(settings, _settings.wrap("sigmas", section))

        # 3) Sweep branch (2026-07-24): if a plan is wired, execute every combo
        #    inside THIS queue item. Otherwise run the single inherited engine
        #    pass exactly as before.
        sweep = kwargs.pop("sweep", None)
        if sweep is None:
            result = super().generate(**kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            import torch
            return result + (settings, torch.zeros(1, 8, 8, 3), "")
        return self._run_sweep(sweep, kwargs, settings)

    # ── sweep execution ────────────────────────────────────────────────────
    def _run_sweep(self, plan, kwargs, base_settings):
        """Loop the plan through the inherited generate() (full per-combo
        setup/teardown: LoRA realize + ephemeral clear, rebalance, ref installs
        - correctness by construction, same as queueing N runs by hand; the
        ~seconds of per-cell install overhead is accepted for v1 and noted in
        the dev doc). Engine + files + metrics live in _sweep_core (portable)."""
        import inspect
        import os
        import torch
        from .. import _sweep_core
        from .krea2_multistage_ultra import EricKrea2MultistageUltra as _Base

        if not isinstance(plan, dict) or not plan.get("combos"):
            raise ValueError("Ultra V2 sweep: the wired plan is empty/invalid")

        # Authoritative defaults, derived live from the engine signature (never
        # hardcoded - same philosophy as ultra_setting_keys).
        sig = inspect.signature(_Base._generate_inner)
        defaults = {k: p.default for k, p in sig.parameters.items()
                    if p.default is not inspect.Parameter.empty}
        valid = set(self.ultra_setting_keys())

        pipe_dict = kwargs.get("krea2_pipeline") or {}
        ck_name = ""
        for key in ("transformer_path", "model_name", "model_path"):
            v = pipe_dict.get(key)
            if v:
                ck_name = os.path.splitext(os.path.basename(str(v)))[0]
                break
        ck_name = ck_name or "krea2"

        try:
            import folder_paths
            base_out = folder_paths.get_output_directory()
        except Exception:
            base_out = os.path.join(os.getcwd(), "output")
        custom = str(plan.get("output_folder") or "").strip().strip('"')
        if custom:
            out_root = custom if os.path.isabs(custom) else os.path.join(base_out, custom)
        else:
            out_root = os.path.join(base_out, "sweeps")

        # Upstream provenance sections for every cell's metadata: the loader
        # recipe (carried in the pipeline dict since 2026-07-24) and the LoRA
        # stack. Merged under their own section names alongside "ultra", same
        # schema the sidecar/save path uses - so a sweep cell PNG is as
        # self-documenting as a normal save.
        extra_sections = {}
        ls = pipe_dict.get("loader_settings")
        if ls:
            try:
                extra_sections.update({k: v for k, v in json.loads(ls).items()
                                       if k != "krea2_settings_version"})
            except Exception:
                pass
        lstack = pipe_dict.get("lora_stack")
        if lstack:
            try:
                extra_sections["lora"] = json.loads(json.dumps(lstack, default=str))
            except Exception:
                pass
        # A wired KREA2_SIGMAS bundle rides through every cell and its enabled
        # stages OVERRIDE the schedule widget (and any swept sX_schedule).
        # Record it in each cell's chunk so bundle-run cells stay standalone-
        # reproducible, and warn when a schedule axis collides with it.
        try:
            _base_doc = json.loads(base_settings)
            if "sigmas" in _base_doc:
                extra_sections["sigmas"] = _base_doc["sigmas"]
                _enabled = set(_base_doc["sigmas"].keys())
                _collide = sorted({k for c in plan.get("combos", [])
                                   for k in c if k.endswith("_schedule")
                                   and k.split("_")[0] in _enabled})
                if _collide:
                    print(f"[EricKrea2-MS] [sweep] WARNING: the wired sigmas bundle "
                          f"overrides {', '.join(_collide)} - those swept schedule "
                          f"values are INERT on the bundle's enabled stage(s); cells "
                          f"differing only there will produce identical images.")
        except Exception:
            pass

        def _settings_with_sections(resolved):
            s = self.serialize_settings(resolved)
            if extra_sections:
                try:
                    doc = json.loads(s)
                    for k, v in extra_sections.items():
                        doc.setdefault(k, v)
                    s = json.dumps(doc, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            return s
        interrupt_check = interrupt_exc = None
        try:
            import comfy.model_management as _mm
            interrupt_check = _mm.throw_exception_if_processing_interrupted
            interrupt_exc = _mm.InterruptProcessingException
        except Exception:
            pass
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(len(plan["combos"]))
        except Exception:
            pass

        last = {"res": None}

        # v2 cell context: per-cell truth for chunk metadata (lora stack /
        # sigmas bundle / prompt actually used). run_one fills it before the
        # engine call; _settings_with_sections reads it right after (strictly
        # sequential within one cell). Reserved non-widget keys (lora_file,
        # lora_strength, lora_baseline, lora_target, prompt, sigma_profile)
        # are popped here - they are plan vocabulary, not engine kwargs.
        cell_ctx = {}
        lora_cfg = plan.get("lora") or {}
        base_stack = list(pipe_dict.get("lora_stack") or [])
        _sigma_node = None

        def _resolve_lora_entry(name_or_path, strength):
            from .._lora_utils import get_lora_full_path
            from .krea2_lora import _sanitize_adapter_name
            import os as _os
            path = None
            if _os.path.isfile(str(name_or_path)):
                path = str(name_or_path)
            else:
                path = get_lora_full_path(str(name_or_path))
            if not path:
                raise ValueError(f"sweep lora: '{name_or_path}' not found (loras folder or full path)")
            s = float(strength)
            return {"path": path, "filename": _os.path.basename(path),
                    "lora_name": str(name_or_path),
                    "adapter_name": "sweep_" + _sanitize_adapter_name(path),
                    "strength": s, "weight_s1": s, "weight_s2": s, "weight_s3": s,
                    "ephemeral": True}

        def _ladder_stack(target, strength):
            import copy as _copy
            st = _copy.deepcopy(base_stack)
            if not st:
                raise ValueError("sweep lora ladder: the panel stack is empty")
            idx = None
            t = str(target).strip()
            if t.isdigit():
                i = int(t) - 1
                if 0 <= i < len(st):
                    idx = i
            if idx is None:
                low = t.lower()
                for i, e in enumerate(st):
                    if (low in str(e.get("lora_name", "")).lower()
                            or low in str(e.get("filename", "")).lower()):
                        idx = i
                        break
            if idx is None:
                raise ValueError(f"sweep lora ladder: target '{target}' matches no stack entry")
            s = float(strength)
            st[idx].update({"strength": s, "weight_s1": s, "weight_s2": s, "weight_s3": s})
            return st

        def _apply_v2_keys(ck, overrides):
            """Pop reserved keys, patch the kwargs copy, record cell truth."""
            cell_ctx.clear()
            lf = ck.pop("lora_file", None)
            ls = ck.pop("lora_strength", None)
            lb = ck.pop("lora_baseline", None)
            lt = ck.pop("lora_target", None)
            pr = ck.pop("prompt", None) if "prompt" in overrides else None
            spf = ck.pop("sigma_profile", None)
            if pr is not None:
                ck["prompt"] = pr
                ck["prompt_conditioning"] = None  # force re-encode for this cell
                cell_ctx["sweep_prompt"] = pr
            if spf is not None:
                nonlocal _sigma_node
                if _sigma_node is None:
                    from .krea2_sigmas import EricKrea2Sigmas
                    _sigma_node = EricKrea2Sigmas()
                bundle = _sigma_node.build(sigmas_preset=str(spf))[0]
                ck["sigmas"] = bundle
                cell_ctx["sigmas"] = {k2: v2 for k2, v2 in bundle.items()
                                      if k2 in ("s1", "s2", "s3") and
                                      isinstance(v2, dict) and v2.get("enabled", True)}
            if lora_cfg and (lb or lf is not None or (lt is not None and ls is not None)):
                if lb:
                    # baseline: ladder -> panel stack as declared; bakeoff ->
                    # no LoRA, unless keep_panel_stack (then the panel stack
                    # alone IS the reference the swept LoRAs layer onto).
                    stack = (list(base_stack)
                             if (lora_cfg.get("mode") == "strength_ladder"
                                 or lora_cfg.get("keep_panel_stack")) else [])
                elif lora_cfg.get("mode") == "bakeoff":
                    entry = _resolve_lora_entry(lf, ls)
                    stack = (list(base_stack) + [entry]
                             if lora_cfg.get("keep_panel_stack") else [entry])
                else:  # strength_ladder
                    stack = _ladder_stack(lt, ls)
                pd = dict(pipe_dict)
                pd["lora_stack"] = stack
                ck["krea2_pipeline"] = pd
                cell_ctx["lora"] = stack
            return ck

        def _settings_with_cell(resolved):
            s = _settings_with_sections(resolved)
            if cell_ctx:
                try:
                    doc = json.loads(s)
                    for k, v in cell_ctx.items():
                        doc[k] = v  # cell truth OVERRIDES base sections
                    s = json.dumps(doc, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            return s

        # S1 latent reuse (sweep v1.1): capture the post-S1 latent from the
        # first cell whose overrides leave Stage 1 invariant, feed it to later
        # invariant cells via the engine's private _s1_reuse kwarg. Conditions:
        # fixed-seed policy, no quick_screen (S1-only runs have nothing to
        # hand off), seed_mode != same_all_stages (generator-sharing - see the
        # engine-side guard), and the cell's override keys strictly s2_*/s3_*
        # (anything global - distilled_shift, upscale_to_stage2, shift_mu_s1,
        # cfg via s1 - changes what S1 produces and disqualifies that cell).
        reuse_ok_base = (str(kwargs.get("seed_mode", "offset_per_stage")) != "same_all_stages"
                         and plan.get("seed_policy", "fixed") == "fixed"
                         and not plan.get("quick_screen"))
        s1_cache = {"lat": None}

        def _s1_invariant(overrides):
            # `seed` is injected into every cell by the engine loop; under the
            # fixed policy (required by reuse_ok_base) it is constant, so it
            # does not vary Stage 1 and is ignored here.
            return all(str(k).startswith(("s2_", "s3_"))
                       for k in overrides
                       if not str(k).startswith("_") and k != "seed")

        def run_one(overrides):
            ck = dict(kwargs)
            ck.update(overrides)
            ck = _apply_v2_keys(ck, overrides)
            reused = False
            if (s1_cache["lat"] is not None and reuse_ok_base
                    and _s1_invariant(overrides)):
                ck["_s1_reuse"] = s1_cache["lat"]
                reused = True
            res = _Base.generate(self, **ck)  # inherited engine, full wrapper
            last["res"] = res
            if (s1_cache["lat"] is None and reuse_ok_base
                    and _s1_invariant(overrides)):
                cand = res[4]  # stage1_latent output
                if isinstance(cand, dict) and cand.get("packed") is not None:
                    s1_cache["lat"] = cand
            return res[0][0].detach().float().cpu().numpy(), {"s1_reused": reused}

        def resolve_values(overrides):
            merged = dict(defaults)
            for k, v in kwargs.items():
                if k in valid or k == "seed":
                    merged[k] = v
            for k, v in overrides.items():
                if not str(k).startswith("_"):
                    merged[k] = v
            return {k: merged[k] for k in sorted(merged) if k in valid or k == "seed"}

        out = _sweep_core.run_sweep(
            plan, run_one=run_one, resolve_values=resolve_values,
            settings_string=_settings_with_cell,
            out_root=out_root, checkpoint_name=ck_name,
            base_seed=int(kwargs.get("seed", 0) or 0),
            interrupt_check=interrupt_check, interrupt_exc=interrupt_exc,
            progress=(lambda i, n: pbar.update_absolute(i)) if pbar else None,
            log=lambda *a: print("[EricKrea2-MS]", *a))

        ok = [c for c in out["cells"] if c is not None]
        if ok:
            hs = [c.shape[0] for c in ok]
            ws = [c.shape[1] for c in ok]
            H, W = max(hs), max(ws)
            if min(hs) != H or min(ws) != W:
                # Mixed cell sizes (megapixels axis): comfy batches need uniform
                # dims, so center-pad smaller cells with black. Files, metrics,
                # and the sheet all use each cell's native output; only this
                # batch view is padded.
                padded = []
                for c in ok:
                    t = torch.zeros(H, W, 3)
                    y0 = (H - c.shape[0]) // 2
                    x0 = (W - c.shape[1]) // 2
                    t[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = torch.from_numpy(c)
                    padded.append(t)
                batch = torch.stack(padded, dim=0)
            else:
                batch = torch.stack([torch.from_numpy(c) for c in ok], dim=0)
        else:
            batch = torch.zeros(1, 8, 8, 3)
        sheet = torch.from_numpy(out["sheet"]).unsqueeze(0)
        if last["res"] is not None:
            _, latent, s1i, s2i, s1l, s2l = last["res"][:6]
        else:
            latent, s1l, s2l = {}, {}, {}
            s1i = s2i = torch.zeros(1, 1, 1, 3)
        # The `settings` STRING is one value for the whole batch, so in sweep
        # mode it cannot carry a per-image recipe (those live in each cell PNG's
        # krea2_settings chunk and in the manifest). Make it self-describing
        # instead: base recipe + a "sweep" section mapping batch index ->
        # overrides, so sidecars written by a downstream Save node are honest
        # about being sweep cells rather than silently claiming panel values.
        try:
            ok_cells = [r for r in out["results"] if "file" in r]
            base_doc = json.loads(base_settings)
            for _k, _v in extra_sections.items():
                base_doc.setdefault(_k, _v)
            base_doc["sweep"] = {
                "sweep_id": os.path.basename(out["sweep_dir"]),
                "manifest": out["manifest_path"],
                "note": ("IMAGE output is the batch of sweep cells; the ultra section "
                         "above is the BASE recipe only. Per-item overrides listed in "
                         "cells[]; full per-cell recipes are embedded in each "
                         "cell_NNN.png and in the manifest."),
                "cells": [{"batch_index": bi, "cell_index": r["index"],
                           "overrides": r["overrides"]}
                          for bi, r in enumerate(ok_cells)],
            }
            sweep_settings = json.dumps(base_doc, indent=2, ensure_ascii=False)
        except Exception:
            sweep_settings = base_settings
        return (batch, latent, s1i, s2i, s1l, s2l, sweep_settings, sheet,
                out["manifest_path"])


NODE_CLASS_MAPPINGS = {"EricKrea2MultistageUltraV2": EricKrea2MultistageUltraV2}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2MultistageUltraV2": "Eric Krea2 Multi-Stage Ultra V2 (presets)"}
