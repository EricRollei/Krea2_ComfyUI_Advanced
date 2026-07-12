# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 LoRA utilities (vendored)
====================================
Model-agnostic LoRA loading/normalization helpers, ported verbatim from the
Eric_Qwen_Edit LoRA node so Eric_Krea2 stays self-contained (no cross-package
import). Handles standard LoRA, LoKR (Kronecker) and LoHa (Hadamard) adapters,
multi-prefix key normalization (transformer. / diffusion_model. /
model.diffusion_model. / model. / unet.), kohya lora_down/up -> diffusers
lora_A/B renaming, PEFT injection with a direct weight-merge fallback, and a
compatibility diagnostic. All functions target `pipe.transformer` generically.

Krea2's diffusers pipeline provides Krea2LoraLoaderMixin (pipe) +
PeftAdapterMixin (transformer), so the clean PEFT path is the norm; the
direct-merge path is the rare fallback for adapters PEFT rejects.

Source author: Eric Hiss (GitHub: EricRollei). Vendored for Eric_Krea2.
"""
import os
import re
import folder_paths
import torch
from typing import Tuple, List


def get_lora_list(prefix: str = None) -> List[str]:
    """Get list of available LoRA files from ComfyUI's loras folder.

    If ``prefix`` is given (e.g. "krea2"), only LoRAs living under that
    top-level subfolder are returned (case-insensitive, matches either path
    separator). If nothing matches the prefix, the full list is returned so the
    node never ends up with an empty dropdown.
    """
    loras = []
    
    # Get ComfyUI's standard loras folder
    lora_paths = folder_paths.get_folder_paths("loras")
    
    for lora_dir in lora_paths:
        if not os.path.isdir(lora_dir):
            continue
        
        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(lora_dir):
            for file in files:
                if file.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                    # Get relative path from lora_dir
                    rel_path = os.path.relpath(os.path.join(root, file), lora_dir)
                    loras.append(rel_path)
    
    # Sort alphabetically
    loras.sort()

    if prefix:
        pl = prefix.lower().rstrip("\\/")
        filtered = [x for x in loras
                    if x.lower().startswith(pl + "\\") or x.lower().startswith(pl + "/")]
        if filtered:
            loras = filtered
    
    # Add "none" option at the start
    return ["none"] + loras


def get_lora_full_path(lora_name: str) -> str:
    """Get the full path for a LoRA file."""
    if lora_name == "none" or not lora_name:
        return None
    
    lora_paths = folder_paths.get_folder_paths("loras")
    
    for lora_dir in lora_paths:
        full_path = os.path.join(lora_dir, lora_name)
        if os.path.exists(full_path):
            return full_path
    
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Adapter format helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_state_dict(path: str) -> dict:
    """Load a state dict from a safetensors, .bin, .pt, or .pth file.

    Raises a clear ValueError when the file is not actually a weights file
    (e.g. a zipped ComfyUI node/workflow bundle that was handed a .bin
    extension), so the failure is actionable instead of cryptic."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        import zipfile
        if zipfile.is_zipfile(path):
            try:
                names = zipfile.ZipFile(path).namelist()
            except Exception:
                names = []
            is_torch = any(n.endswith("data.pkl") or n.endswith("/version")
                           or n == "version" for n in names)
            if names and not is_torch:
                exts = sorted({os.path.splitext(n)[1].lower()
                               for n in names if not n.endswith("/")})
                raise ValueError(
                    f"'{os.path.basename(path)}' is not a LoRA/weights file - it is "
                    f"a ZIP bundle (contents: {exts}). This is a packaged ComfyUI "
                    "node/workflow (e.g. a conditioning-rebalance trick), not model "
                    "weights, so it cannot be loaded as a LoRA. Install its node "
                    "separately instead."
                ) from None
        raise ValueError(
            f"Could not read '{os.path.basename(path)}' as a weights file "
            f"(torch.load failed: {str(e)[:120]}). If this is a trusted legacy "
            "pickle LoRA, re-save it as .safetensors."
        ) from None
    if not isinstance(sd, dict):
        raise ValueError(
            f"'{os.path.basename(path)}' did not contain a state dict "
            f"(got {type(sd).__name__}); it is not a loadable LoRA."
        )
    return sd


def _adapter_module_path(key: str) -> str:
    """Extract the module path from an adapter state-dict key.

    Strips adapter-specific suffixes like ``.lokr_w1``, ``.lora_A.weight``,
    ``.alpha``, etc. so we get the bare module path that should correspond
    to a ``nn.Module`` inside the transformer.
    """
    _SUFFIXES = (
        ".lokr_w1", ".lokr_w2", ".lokr_t2",
        ".lora_A.weight", ".lora_A.default.weight",
        ".lora_B.weight", ".lora_B.default.weight",
        ".lora_down.weight", ".lora_up.weight",
        ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
        ".alpha", ".diff", ".diff_b",
    )
    for sfx in _SUFFIXES:
        idx = key.find(sfx)
        if idx >= 0:
            return key[:idx]
    # Fallback: strip last dotted component
    return key.rsplit(".", 1)[0] if "." in key else key


def _normalize_keys(state_dict: dict, model=None) -> dict:
    """Strip component prefixes from state-dict keys.

    Many non-diffusers LoRA files bake in a component prefix such as
    ``transformer.``, ``diffusion_model.``, or ``model.diffusion_model.``.
    Diffusers expects keys relative to the transformer module itself.

    If *model* is provided (the ``nn.Module`` the adapter will target),
    the function **auto-detects** which prefix to strip by comparing the
    adapter module paths from the state dict against the model's
    ``named_modules()``.  This makes it robust regardless of training
    tool or checkpoint convention.
    """
    # Filter out text-encoder keys first
    filtered = {}
    for k, v in state_dict.items():
        if k.startswith(("text_encoder.", "text_encoder_2.")):
            continue
        filtered[k] = v

    if not filtered:
        return filtered

    # ── Smart mode: match against actual model modules ───────────────
    if model is not None:
        model_names = {name for name, _ in model.named_modules() if name}
        sd_module_paths = {_adapter_module_path(k) for k in filtered}

        # Already matches — no stripping needed
        if sd_module_paths & model_names:
            return filtered

        # Try known prefixes (most common first)
        _KNOWN = [
            "transformer.",
            "diffusion_model.",
            "model.diffusion_model.",
            "model.",
        ]
        for pfx in _KNOWN:
            stripped = {p[len(pfx):] for p in sd_module_paths
                        if p.startswith(pfx)}
            if stripped & model_names:
                cleaned = {}
                for k, v in filtered.items():
                    cleaned[k[len(pfx):] if k.startswith(pfx) else k] = v
                return cleaned

        # Auto-detect: find ANY prefix that produces matches
        for sd_path in list(sd_module_paths)[:20]:
            for m_path in list(model_names)[:50]:
                if sd_path.endswith(m_path) and len(sd_path) > len(m_path):
                    pfx = sd_path[: -len(m_path)]
                    hit = sum(1 for p in sd_module_paths
                              if p.startswith(pfx)
                              and p[len(pfx):] in model_names)
                    if hit > len(sd_module_paths) * 0.3:
                        print(f"[LoRA] Auto-detected key prefix: '{pfx}'")
                        cleaned = {}
                        for k, v in filtered.items():
                            cleaned[k[len(pfx):]
                                    if k.startswith(pfx)
                                    else k] = v
                        return cleaned

        # Nothing matched — warn and return as-is
        print("[LoRA] WARNING: Could not match state-dict keys to model "
              "modules after trying known prefixes.")
        print(f"[LoRA]   state-dict paths (sample): "
              f"{sorted(sd_module_paths)[:3]}")
        print(f"[LoRA]   model module paths (sample): "
              f"{sorted(model_names)[:5]}")
        return filtered

    # ── Simple mode (no model): strip 'transformer.' if present ──────
    prefix = "transformer."
    if any(k.startswith(prefix) for k in filtered):
        cleaned = {}
        for k, v in filtered.items():
            cleaned[k[len(prefix):] if k.startswith(prefix) else k] = v
        return cleaned
    return filtered


def _detect_adapter_type(state_dict: dict) -> str:
    """Detect adapter format from state dict keys.

    Returns one of: ``"lokr"``, ``"loha"``, ``"lora"``, ``"unknown"``.
    """
    keys = set(state_dict.keys())
    has_lokr = any("lokr_w1" in k or "lokr_w2" in k for k in keys)
    has_loha = any("hada_w1_a" in k or "hada_w2_a" in k for k in keys)
    has_lora = any("lora_A" in k or "lora_B" in k for k in keys)
    has_lora_alt = any("lora_down" in k or "lora_up" in k for k in keys)

    if has_lokr:
        return "lokr"
    if has_loha:
        return "loha"
    if has_lora or has_lora_alt:
        return "lora"
    return "unknown"


def _load_lokr_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]",
                       weight: float = 1.0) -> None:
    """Inject a LoKR (Kronecker) adapter via PEFT.

    Diffusers' ``load_lora_weights`` only understands standard LoRA format.
    For LoKR we bypass the pipeline and use PEFT's ``inject_adapter_in_model``
    + ``set_peft_model_state_dict`` directly on the transformer.

    If PEFT injection fails (e.g. due to unresolvable key mismatch), falls
    back to **direct weight merging**: computes ``kron(w1, w2) * scale`` and
    adds the deltas straight to the transformer parameters.

    Scaling convention: for full (non-decomposed) LoKR the effective delta is
    ``kron(w1, w2) * (alpha / r)``.  We set ``alpha = r`` in the config so
    that the base scaling is 1.0 (matching ComfyUI / LyCORIS convention).
    ``set_adapters()`` then multiplies by the user-supplied weight.
    """
    try:
        _load_lokr_adapter_peft(pipe, state_dict, adapter_name, log_prefix)
    except (ValueError, RuntimeError) as peft_err:
        print(f"{log_prefix} PEFT injection failed: {peft_err}")
        print(f"{log_prefix} Falling back to direct weight merge...")
        _load_lokr_adapter_direct(pipe, state_dict, adapter_name, log_prefix,
                                  weight=weight)


def _load_lokr_adapter_peft(pipe, state_dict: dict, adapter_name: str,
                            log_prefix: str = "[LoRA]") -> None:
    """Inject LoKR adapter via PEFT's ``inject_adapter_in_model``."""
    from peft import LoKrConfig, inject_adapter_in_model, set_peft_model_state_dict

    transformer = pipe.transformer

    # Use a very large r so that PEFT creates full (non-decomposed) w1/w2
    # matrices.  alpha = r gives unit scaling (1.0).
    cfg_r = 100000
    config = LoKrConfig(
        r=cfg_r,
        alpha=float(cfg_r),
        decompose_both=False,
        decompose_factor=-1,       # default factorization (near square-root)
        target_modules=["_dummy"],  # will be overridden by state_dict keys
    )

    # Inject adapter layers (structure only, random weights)
    inject_adapter_in_model(config, transformer,
                            adapter_name=adapter_name,
                            state_dict=state_dict)

    # Load actual trained weights from state dict
    incompatible = set_peft_model_state_dict(transformer, state_dict,
                                             adapter_name=adapter_name)

    # Mark PEFT as active so the pipeline's set_adapters() works
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    # Log incompatible keys (alpha entries are expected here)
    if incompatible:
        unexpected = getattr(incompatible, "unexpected_keys", [])
        missing = getattr(incompatible, "missing_keys", [])
        if unexpected:
            # Filter out alpha keys which are expected to be left over
            non_alpha = [k for k in unexpected if not k.endswith(".alpha")]
            if non_alpha:
                print(f"{log_prefix} LoKR unexpected keys: {non_alpha[:5]}...")
        if missing:
            print(f"{log_prefix} LoKR missing keys: {missing[:5]}...")

    print(f"{log_prefix} LoKR adapter loaded successfully via PEFT injection")


def _load_lokr_adapter_direct(pipe, state_dict: dict, adapter_name: str,
                              log_prefix: str = "[LoRA]",
                              weight: float = 1.0) -> None:
    """Apply LoKR adapter by directly merging weights into the transformer.

    Computes ``kron(w1, w2) * (alpha / r) * weight`` and adds the delta to
    each target parameter.  When *alpha* is absent from the checkpoint the
    scale defaults to ``1.0`` (weights are assumed pre-scaled).

    This is the same approach ComfyUI's native weight patcher uses.
    The adapter is registered in ``peft_config`` so ``set_adapters()``
    can still find it, but per-stage weight adjustment is limited to a
    single re-merge at changed weight.
    """
    import math, re

    transformer = pipe.transformer
    model_sd = dict(transformer.named_parameters())

    # Group state dict keys by module path
    modules: dict[str, dict] = {}  # module_path -> {"lokr_w1": ..., ...}
    for k, v in state_dict.items():
        path = _adapter_module_path(k)
        # Extract the param name (e.g., "lokr_w1", "alpha")
        param_name = k[len(path) + 1:]  # +1 for the dot
        modules.setdefault(path, {})[param_name] = v

    applied = 0
    skipped = 0
    for mod_path, params in modules.items():
        w1 = params.get("lokr_w1")
        w2 = params.get("lokr_w2")
        if w1 is None or w2 is None:
            skipped += 1
            continue

        # Compute scaling factor.
        # When alpha IS stored, use alpha / r (LyCORIS convention).
        # When alpha is NOT stored, assume weights are pre-scaled → 1.0.
        alpha = params.get("alpha")
        if alpha is not None:
            alpha_val = alpha.item()
            r_val = min(w1.shape) if w1.ndim >= 2 else 1
            scale = (alpha_val / r_val) * weight if r_val > 0 else weight
        else:
            scale = weight

        # Compute delta = kron(w1, w2) * scale
        w1f = w1.float()
        w2f = w2.float()
        delta = torch.kron(w1f, w2f) * scale

        # Match to model param.  LoKR targets ".weight" by default.
        target_key = mod_path + ".weight"
        if target_key not in model_sd:
            # Try without .weight suffix
            target_key = mod_path
        if target_key not in model_sd:
            skipped += 1
            continue

        param = model_sd[target_key]
        if delta.shape != param.shape:
            # Try reshaping
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                print(f"{log_prefix} Shape mismatch for {mod_path}: "
                      f"delta {delta.shape} vs param {param.shape}, skipping")
                skipped += 1
                continue

        # Store original weights for potential unloading
        backup_key = f"_lokr_backup_{adapter_name}"
        if not hasattr(transformer, backup_key):
            setattr(transformer, backup_key, {})
        backup = getattr(transformer, backup_key)
        if target_key not in backup:
            backup[target_key] = param.data.clone()

        # Apply delta
        param.data.add_(delta.to(dtype=param.dtype, device=param.device))
        applied += 1

    # Register in peft_config for set_adapters() discovery
    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    # Minimal config marker so get_list_adapters() finds this adapter
    transformer.peft_config[adapter_name] = {
        "_type": "lokr_direct",
        "_applied_modules": applied,
        "_weight": weight,
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoKR direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    if skipped > applied:
        sample_keys = list(state_dict.keys())[:5]
        sample_modules = sorted(model_sd.keys())[:5]
        print(f"{log_prefix} WARNING: Many modules skipped. "
              f"State-dict keys (sample): {sample_keys}")
        print(f"{log_prefix}   Model params (sample): {sample_modules}")


def _load_loha_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]",
                       weight: float = 1.0) -> None:
    """Inject a LoHa (Hadamard) adapter via PEFT.

    LoHa always uses decomposed weights (w1_a/w1_b, w2_a/w2_b), so the
    rank ``r`` is directly available from the weight shapes.

    Falls back to direct weight merge if PEFT injection fails.
    """
    try:
        _load_loha_adapter_peft(pipe, state_dict, adapter_name, log_prefix)
    except (ValueError, RuntimeError) as peft_err:
        print(f"{log_prefix} PEFT injection failed for LoHa: {peft_err}")
        print(f"{log_prefix} Falling back to direct weight merge...")
        _load_loha_adapter_direct(pipe, state_dict, adapter_name, log_prefix,
                                  weight=weight)


def _load_loha_adapter_peft(pipe, state_dict: dict, adapter_name: str,
                            log_prefix: str = "[LoRA]") -> None:
    """Inject LoHa adapter via PEFT's ``inject_adapter_in_model``."""
    from peft import LoHaConfig, inject_adapter_in_model, set_peft_model_state_dict

    transformer = pipe.transformer

    # Determine rank from the first w1_a tensor shape
    r_val = None
    alpha_val = None
    for key, val in state_dict.items():
        if ".hada_w1_a" in key and val.ndim >= 2:
            r_val = val.shape[1]  # (out_dim, rank)
            break
        if ".hada_w1_b" in key and val.ndim >= 2:
            r_val = val.shape[0]  # (rank, in_dim)
            break
    if r_val is None:
        r_val = 8  # fallback default

    # Try to read alpha from state dict
    for key, val in state_dict.items():
        if key.endswith(".alpha") and val.numel() == 1:
            alpha_val = val.item()
            break
    if alpha_val is None:
        alpha_val = float(r_val)

    config = LoHaConfig(
        r=r_val,
        alpha=alpha_val,
        target_modules=["_dummy"],  # overridden by state_dict keys
    )

    inject_adapter_in_model(config, transformer,
                            adapter_name=adapter_name,
                            state_dict=state_dict)
    incompatible = set_peft_model_state_dict(transformer, state_dict,
                                             adapter_name=adapter_name)

    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    if incompatible:
        unexpected = getattr(incompatible, "unexpected_keys", [])
        non_alpha = [k for k in unexpected if not k.endswith(".alpha")]
        if non_alpha:
            print(f"{log_prefix} LoHa unexpected keys: {non_alpha[:5]}...")

    print(f"{log_prefix} LoHa adapter loaded successfully via PEFT injection")


def _load_loha_adapter_direct(pipe, state_dict: dict, adapter_name: str,
                              log_prefix: str = "[LoRA]",
                              weight: float = 1.0) -> None:
    """Apply LoHa adapter by directly merging weights into the transformer.

    LoHa delta = ``(w1_a @ w1_b) * (w2_a @ w2_b) * (alpha / r) * weight``.
    When *alpha* is absent, scale defaults to ``1.0``.
    """
    transformer = pipe.transformer
    model_sd = dict(transformer.named_parameters())

    # Group state dict keys by module path
    modules: dict[str, dict] = {}
    for k, v in state_dict.items():
        path = _adapter_module_path(k)
        param_name = k[len(path) + 1:]
        modules.setdefault(path, {})[param_name] = v

    applied = 0
    skipped = 0
    for mod_path, params in modules.items():
        w1_a = params.get("hada_w1_a")
        w1_b = params.get("hada_w1_b")
        w2_a = params.get("hada_w2_a")
        w2_b = params.get("hada_w2_b")
        if w1_a is None or w1_b is None or w2_a is None or w2_b is None:
            skipped += 1
            continue

        # Scaling: alpha / r when alpha is present, else 1.0
        alpha = params.get("alpha")
        if alpha is not None:
            alpha_val = alpha.item()
            r_val = w1_b.shape[0] if w1_b.ndim >= 2 else 1
            scale = (alpha_val / r_val) * weight if r_val > 0 else weight
        else:
            scale = weight

        # delta = (w1_a @ w1_b) * (w2_a @ w2_b) * scale
        w1 = (w1_a.float() @ w1_b.float())
        w2 = (w2_a.float() @ w2_b.float())
        delta = w1 * w2 * scale

        target_key = mod_path + ".weight"
        if target_key not in model_sd:
            target_key = mod_path
        if target_key not in model_sd:
            skipped += 1
            continue

        param = model_sd[target_key]
        if delta.shape != param.shape:
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                print(f"{log_prefix} LoHa shape mismatch for {mod_path}: "
                      f"delta {delta.shape} vs param {param.shape}, skipping")
                skipped += 1
                continue

        backup_key = f"_loha_backup_{adapter_name}"
        if not hasattr(transformer, backup_key):
            setattr(transformer, backup_key, {})
        backup = getattr(transformer, backup_key)
        if target_key not in backup:
            backup[target_key] = param.data.clone()

        param.data.add_(delta.to(dtype=param.dtype, device=param.device))
        applied += 1

    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    transformer.peft_config[adapter_name] = {
        "_type": "loha_direct",
        "_applied_modules": applied,
        "_weight": weight,
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoHa direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    if skipped > applied:
        sample_keys = list(state_dict.keys())[:5]
        sample_modules = sorted(model_sd.keys())[:5]
        print(f"{log_prefix} WARNING: Many modules skipped. "
              f"State-dict keys (sample): {sample_keys}")
        print(f"{log_prefix}   Model params (sample): {sample_modules}")


def _rename_lora_down_up(state_dict: dict) -> dict:
    """Rename ``lora_down`` / ``lora_up`` keys to ``lora_A`` / ``lora_B``.

    Some training tools (kohya_ss, diffusers-native) use ``lora_down``/
    ``lora_up`` instead of the PEFT-standard ``lora_A``/``lora_B``.
    This normalises them so the rest of the loading pipeline only needs
    to handle one convention.
    """
    if not any("lora_down" in k or "lora_up" in k for k in state_dict):
        return state_dict

    renamed = {}
    count = 0
    for k, v in state_dict.items():
        new_k = k.replace(".lora_down.weight", ".lora_A.weight") \
                 .replace(".lora_up.weight", ".lora_B.weight")
        if new_k != k:
            count += 1
        renamed[new_k] = v

    if count:
        print(f"[LoRA] Renamed {count} lora_down/lora_up keys "
              f"to lora_A/lora_B")
    return renamed


def _live_lora_layer_count(transformer, adapter_name: str = None) -> int:
    """Count PEFT tuner layers currently holding LoRA weights on the transformer.

    If *adapter_name* is given, count only layers that reference that adapter.
    Used to detect the diffusers silent no-op: ``pipe.load_lora_weights()``
    returns WITHOUT error even when it matches zero modules (it only prints a
    'No LoRA keys associated ...' warning), so a return value is not proof the
    LoRA was actually applied — a change in this count is.
    """
    if transformer is None:
        return 0
    n = 0
    for module in transformer.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or not scaling:
            continue
        if adapter_name is None or adapter_name in scaling:
            n += 1
    return n


def _safe_delete_adapter(pipe, adapter_name: str,
                         log_prefix: str = "[LoRA]") -> None:
    """Best-effort removal of a single (possibly empty) adapter registration so
    a failed fast-path load doesn't leave a phantom adapter behind before we
    retry via the robust path. Never raises."""
    for target in (pipe, getattr(pipe, "transformer", None)):
        if target is None:
            continue
        for meth in ("delete_adapters", "delete_adapter"):
            fn = getattr(target, meth, None)
            if callable(fn):
                try:
                    fn(adapter_name)
                    return
                except Exception:
                    pass


def _load_lora_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]",
                       weight: float = 1.0) -> None:
    """Load a standard LoRA adapter with multi-level fallback.

    Handles both ``lora_A``/``lora_B`` and ``lora_down``/``lora_up``
    naming conventions (the latter is normalised to the former).

    1. ``pipe.load_lora_weights(state_dict)`` — best integration with
       diffusers' adapter management.
    2. Direct PEFT ``inject_adapter_in_model`` on the transformer —
       still supports ``set_adapters()`` for weight control.
    3. Direct weight merge (B @ A * scale) — last resort, weight is
       baked in at load time.
    """
    # Normalise lora_down/lora_up → lora_A/lora_B
    state_dict = _rename_lora_down_up(state_dict)
    transformer = getattr(pipe, "transformer", None)
    # ── Try 1: Pipeline LoRA loading with normalised keys ────────────
    try:
        before = _live_lora_layer_count(transformer)
        pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
        if _live_lora_layer_count(transformer) > before:
            print(f"{log_prefix} LoRA loaded via pipeline with normalised keys")
            return
        # Silent no-op: diffusers matched 0 modules (only warned). Clean up the
        # phantom adapter and fall through to direct PEFT injection.
        print(f"{log_prefix} Pipeline load matched 0 modules; "
              "using direct PEFT injection instead")
        _safe_delete_adapter(pipe, adapter_name, log_prefix)
    except (ValueError, RuntimeError) as e:
        print(f"{log_prefix} Pipeline LoRA load with normalised keys "
              f"failed: {str(e)[:120]}")

    # ── Try 2: Direct PEFT injection on the transformer ──────────────
    try:
        _load_lora_adapter_peft(pipe, state_dict, adapter_name, log_prefix)
        return
    except (ValueError, RuntimeError) as e:
        print(f"{log_prefix} Direct PEFT LoRA injection failed: "
              f"{str(e)[:120]}")

    # ── Try 3: Direct weight merge (last resort) ─────────────────────
    print(f"{log_prefix} Falling back to direct LoRA weight merge...")
    _load_lora_adapter_direct(pipe, state_dict, adapter_name, log_prefix,
                              weight=weight)


def _load_lora_adapter_peft(pipe, state_dict: dict, adapter_name: str,
                            log_prefix: str = "[LoRA]") -> None:
    """Inject standard LoRA adapter directly via PEFT on the transformer."""
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    transformer = pipe.transformer

    # Normalise lora_down/lora_up → lora_A/lora_B before PEFT sees them
    state_dict = _rename_lora_down_up(state_dict)

    # Determine rank from first lora_A tensor
    r_val = None
    alpha_val = None
    for key, val in state_dict.items():
        if "lora_A" in key and val.ndim >= 2:
            r_val = val.shape[0]
            break
    if r_val is None:
        r_val = 64  # common default

    for key, val in state_dict.items():
        if key.endswith(".alpha") and val.numel() == 1:
            alpha_val = val.item()
            break
    if alpha_val is None:
        alpha_val = float(r_val)

    config = LoraConfig(
        r=r_val,
        lora_alpha=alpha_val,
        target_modules=["_dummy"],  # overridden by state_dict keys
    )

    inject_adapter_in_model(config, transformer,
                            adapter_name=adapter_name,
                            state_dict=state_dict)
    incompatible = set_peft_model_state_dict(transformer, state_dict,
                                             adapter_name=adapter_name)

    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    if incompatible:
        unexpected = getattr(incompatible, "unexpected_keys", [])
        non_alpha = [k for k in unexpected if not k.endswith(".alpha")]
        if non_alpha:
            print(f"{log_prefix} LoRA unexpected keys: {non_alpha[:5]}...")
        missing = getattr(incompatible, "missing_keys", [])
        if missing:
            print(f"{log_prefix} LoRA missing keys: {missing[:5]}...")

    print(f"{log_prefix} LoRA loaded via direct PEFT injection")


def _load_lora_adapter_direct(pipe, state_dict: dict, adapter_name: str,
                              log_prefix: str = "[LoRA]",
                              weight: float = 1.0) -> None:
    """Apply standard LoRA by directly merging B @ A * scale into weights.

    The user *weight* is baked in at merge time.  ``set_adapters()`` will
    not be able to change it afterwards (a limitation of direct merge).
    """
    transformer = pipe.transformer
    model_sd = dict(transformer.named_parameters())

    modules: dict[str, dict] = {}
    for k, v in state_dict.items():
        path = _adapter_module_path(k)
        param_name = k[len(path) + 1:]
        modules.setdefault(path, {})[param_name] = v

    applied = 0
    skipped = 0
    for mod_path, params in modules.items():
        lora_A = params.get("lora_A.weight") or params.get(
            "lora_A.default.weight")
        lora_B = params.get("lora_B.weight") or params.get(
            "lora_B.default.weight")
        if lora_A is None or lora_B is None:
            skipped += 1
            continue

        # Scaling: delta = B @ A * (alpha / r) * weight
        alpha = params.get("alpha")
        if alpha is not None:
            alpha_val = alpha.item()
            r_val = lora_A.shape[0]
            scale = (alpha_val / r_val) * weight if r_val > 0 else weight
        else:
            scale = weight

        delta = (lora_B.float() @ lora_A.float()) * scale

        target_key = mod_path + ".weight"
        if target_key not in model_sd:
            target_key = mod_path
        if target_key not in model_sd:
            skipped += 1
            continue

        param = model_sd[target_key]
        if delta.shape != param.shape:
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                print(f"{log_prefix} LoRA shape mismatch for {mod_path}: "
                      f"delta {delta.shape} vs param {param.shape}, skipping")
                skipped += 1
                continue

        # Backup for unloading
        backup_key = f"_lora_backup_{adapter_name}"
        if not hasattr(transformer, backup_key):
            setattr(transformer, backup_key, {})
        backup = getattr(transformer, backup_key)
        if target_key not in backup:
            backup[target_key] = param.data.clone()

        param.data.add_(delta.to(dtype=param.dtype, device=param.device))
        applied += 1

    # Register in peft_config for adapter discovery
    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    transformer.peft_config[adapter_name] = {
        "_type": "lora_direct",
        "_applied_modules": applied,
        "_weight": weight,
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoRA direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    if skipped > applied:
        sample_keys = list(state_dict.keys())[:5]
        sample_modules = sorted(model_sd.keys())[:5]
        print(f"{log_prefix} WARNING: Many modules skipped. "
              f"State-dict keys (sample): {sample_keys}")
        print(f"{log_prefix}   Model params (sample): {sample_modules}")


def _set_adapters_safe(pipe, adapter_name: str, weight: float,
                       log_prefix: str = "[LoRA]") -> None:
    """Call ``pipe.set_adapters()`` with graceful handling for direct-merge
    adapters that don't have PEFT tuner layers.

    For PEFT-injected adapters ``set_adapters()`` adjusts the scaling.
    For direct-merge adapters the weight is already baked into the model
    parameters at load time, so ``set_adapters()`` is a no-op and we just
    log a note.
    """
    # Check if this is a direct-merge adapter
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        peft_cfg = getattr(transformer, "peft_config", {})
        cfg = peft_cfg.get(adapter_name, {})
        if isinstance(cfg, dict) and cfg.get("_type", "").endswith("_direct"):
            if abs(weight - cfg.get("_weight", 1.0)) > 1e-6:
                print(f"{log_prefix} NOTE: '{adapter_name}' was loaded via "
                      f"direct weight merge (weight={cfg.get('_weight', 1.0)})."
                      f" Changing weight requires reloading the LoRA.")
            return

    try:
        pipe.set_adapters([adapter_name], adapter_weights=[weight])
    except Exception as e:
        print(f"{log_prefix} set_adapters note: {str(e)[:100]}  "
              f"(adapter may have been loaded via direct merge)")


# ═══════════════════════════════════════════════════════════════════════
#  Original-format Krea LoRA → diffusers Krea2Transformer2DModel converter
# ═══════════════════════════════════════════════════════════════════════
# Krea/CivitAI LoRAs are trained against the ORIGINAL (BFL-style) checkpoint
# module names and shipped in kohya (``lora_unet_...`` with underscores) or
# ComfyUI (``diffusion_model....`` with dots) layout. diffusers' Krea2 loader
# only understands its own renamed modules (``transformer_blocks.N.attn.to_q``,
# ``ff.gate``, ``img_in`` ...), so it silently matches 0 keys. This converter
# rewrites the keys to diffusers naming so ``pipe.load_lora_weights`` works.
#
# Krea2's loader hardcodes ``network_alphas=None`` (see Krea2LoraLoaderMixin),
# so per-layer alpha is folded into ``lora_B`` here: with the default PEFT
# scaling (alpha=r → 1.0) the reproduced delta equals ``(alpha/r) * B @ A``.

# Attention / MLP leaf names inside a block (original → diffusers).
_KREA_LEAF = {
    "attn_wq": "attn.to_q",
    "attn_wk": "attn.to_k",
    "attn_wv": "attn.to_v",
    "attn_wo": "attn.to_out.0",
    "attn_gate": "attn.to_gate",
    "mlp_gate": "ff.gate",
    "mlp_up": "ff.up",
    "mlp_down": "ff.down",
}

# Top-level (non-block) modules (original → diffusers).
_KREA_STANDALONE = {
    "first": "img_in",
    "last_linear": "final_layer.linear",
    "tmlp_0": "time_embed.linear_1",
    "tmlp_2": "time_embed.linear_2",
    "tproj_1": "time_mod_proj",
    "txtmlp_1": "txt_in.linear_1",
    "txtmlp_3": "txt_in.linear_2",
    "txtfusion_projector": "text_fusion.projector",
}


def _prod(shape) -> int:
    """Product of a shape tuple (number of elements)."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _map_krea_module(tok: str):
    """Map an underscore-joined ORIGINAL Krea module token to its diffusers
    module path, or return ``None`` if it isn't a recognised Krea module."""
    if tok in _KREA_STANDALONE:
        return _KREA_STANDALONE[tok]
    m = re.match(r"^blocks_(\d+)_(.+)$", tok)
    if m and m.group(2) in _KREA_LEAF:
        return f"transformer_blocks.{m.group(1)}.{_KREA_LEAF[m.group(2)]}"
    m = re.match(r"^txtfusion_(layerwise|refiner)_blocks_(\d+)_(.+)$", tok)
    if m and m.group(3) in _KREA_LEAF:
        return f"text_fusion.{m.group(1)}_blocks.{m.group(2)}.{_KREA_LEAF[m.group(3)]}"
    return None


def _krea_key_parts(key: str):
    """Split a LoRA key into (underscore-token module, kind) where kind is one
    of 'down' | 'up' | 'alpha', canonicalising kohya (``lora_unet_``) and Comfy
    (``diffusion_model.``) prefixes/separators. Returns (None, None) for keys
    that are already diffusers-format or unrecognised."""
    if key.endswith(".lora_down.weight"):
        base, kind = key[: -len(".lora_down.weight")], "down"
    elif key.endswith(".lora_up.weight"):
        base, kind = key[: -len(".lora_up.weight")], "up"
    elif key.endswith(".lora_A.weight"):
        base, kind = key[: -len(".lora_A.weight")], "down"
    elif key.endswith(".lora_B.weight"):
        base, kind = key[: -len(".lora_B.weight")], "up"
    elif key.endswith(".alpha"):
        base, kind = key[: -len(".alpha")], "alpha"
    elif key.endswith(".diff_b"):
        base, kind = key[: -len(".diff_b")], "diff_b"
    elif key.endswith(".diff"):
        base, kind = key[: -len(".diff")], "diff"
    else:
        return None, None

    if base.startswith("transformer."):
        return None, None            # already diffusers format
    if base.startswith("lora_unet_"):
        tok = base[len("lora_unet_"):]
    elif base.startswith("diffusion_model."):
        tok = base[len("diffusion_model."):].replace(".", "_")
    elif base.startswith("lora_transformer_"):
        tok = base[len("lora_transformer_"):]
    else:
        tok = base.replace(".", "_")
    return tok, kind


def _convert_krea_lora_to_diffusers(state_dict, model=None, log_prefix="[LoRA]"):
    """Detect an original-format Krea LoRA/patch and rewrite it to diffusers
    ``Krea2Transformer2DModel`` naming.

    Handles, in one pass, every lightweight format these checkpoints ship in:
      * low-rank LoRA (``lora_down``/``lora_up`` or ``lora_A``/``lora_B``,
        optional ``.alpha``) -> ``transformer.<module>.lora_A/B.weight`` with
        the alpha/rank scale folded into ``lora_B`` (Krea2's loader hardcodes
        ``network_alphas=None``);
      * full-weight / bias deltas (``.diff`` / ``.diff_b``, the ComfyUI patch
        format used by many slider / enhancer LoRAs) -> raw deltas keyed by
        diffusers module path for a direct (sticky) merge.

    Returns ``(converted_lora, direct_deltas, n_mapped, n_total)`` where
    ``converted_lora`` is a diffusers LoRA state-dict (or ``None``) and
    ``direct_deltas`` maps ``diffusers_module -> {'weight': t, 'bias': t}``
    (or ``None``). Both are ``None`` when the file isn't a Krea checkpoint."""
    # Group down/up/alpha/diff/diff_b by original module token.
    groups = {}
    for k, v in state_dict.items():
        tok, kind = _krea_key_parts(k)
        if tok is None:
            continue
        groups.setdefault(tok, {})[kind] = v

    # Only treat as a Krea LoRA if a meaningful share of tokens map.
    mappable = [t for t in groups if _map_krea_module(t) is not None]
    if not groups or len(mappable) < max(1, 0.5 * len(groups)):
        return None, None, 0, 0

    # Parameter shapes for validation (covers weight AND bias, any ndim).
    param_shapes = {}
    if model is not None:
        for name, p in model.named_parameters():
            param_shapes[name] = tuple(p.shape)

    converted = {}
    direct = {}          # diffusers_module -> {"weight": t, "bias": t}
    n_map = 0
    n_tot = len(groups)
    unmapped = []
    for tok, g in groups.items():
        diff_mod = _map_krea_module(tok)
        if diff_mod is None:
            unmapped.append(tok)
            continue
        mapped_this = False

        # (1) Low-rank LoRA ------------------------------------------------
        down, up = g.get("down"), g.get("up")
        if (down is not None and up is not None
                and down.ndim == 2 and up.ndim == 2):
            ok = True
            if param_shapes:
                shp = param_shapes.get(f"{diff_mod}.weight")
                if shp is None:
                    unmapped.append(f"{tok}->{diff_mod}(missing)")
                    ok = False
                elif len(shp) != 2 or down.shape[1] != shp[1] or up.shape[0] != shp[0]:
                    print(f"{log_prefix}   Krea map: shape mismatch {tok}->{diff_mod} "
                          f"(down {tuple(down.shape)}, up {tuple(up.shape)} vs "
                          f"weight {shp}); skipped")
                    ok = False
            if ok:
                rank = down.shape[0]
                alpha = g.get("alpha")
                scale = (float(alpha.item()) / rank) if (alpha is not None and rank > 0) else 1.0
                up_scaled = (up.float() * scale).to(up.dtype)
                base = f"transformer.{diff_mod}"
                converted[f"{base}.lora_A.weight"] = down
                converted[f"{base}.lora_B.weight"] = up_scaled
                mapped_this = True

        # (2) Full-weight / bias deltas (.diff / .diff_b) ------------------
        wdelta = g.get("diff")
        if wdelta is not None:
            shp = param_shapes.get(f"{diff_mod}.weight") if param_shapes else None
            if shp is None or tuple(wdelta.shape) == shp or wdelta.numel() == _prod(shp):
                direct.setdefault(diff_mod, {})["weight"] = wdelta
                mapped_this = True
            else:
                print(f"{log_prefix}   Krea map: diff shape mismatch {tok}->{diff_mod} "
                      f"({tuple(wdelta.shape)} vs {shp}); skipped")
        bdelta = g.get("diff_b")
        if bdelta is not None:
            shp = param_shapes.get(f"{diff_mod}.bias") if param_shapes else None
            if shp is None or tuple(bdelta.shape) == shp:
                direct.setdefault(diff_mod, {})["bias"] = bdelta
                mapped_this = True
            else:
                print(f"{log_prefix}   Krea map: diff_b shape mismatch {tok}->{diff_mod} "
                      f"({tuple(bdelta.shape)} vs {shp}); skipped")

        if mapped_this:
            n_map += 1

    if unmapped:
        print(f"{log_prefix}   Krea map: {len(unmapped)} unmapped module(s), e.g. "
              f"{sorted(unmapped)[:5]}")
    if n_map == 0:
        return None, None, 0, n_tot
    return (converted or None), (direct or None), n_map, n_tot


def _apply_krea_direct_deltas(pipe, deltas, adapter_name, log_prefix="[LoRA]",
                              weight: float = 1.0) -> int:
    """Add full-weight/bias deltas (``.diff`` / ``.diff_b``) straight into the
    transformer parameters (a sticky direct merge, since these are full-rank
    and can't be expressed as a PEFT adapter).

    Originals are backed up on the transformer as ``_lora_backup_<adapter>`` so
    ``unload_all_loras`` can restore them (no accumulation across runs), and a
    ``diff_direct`` marker is registered in ``peft_config`` so the stack knows
    the value is baked in and won't try to re-weight it per stage."""
    transformer = pipe.transformer
    params = dict(transformer.named_parameters())
    backup_key = f"_lora_backup_{adapter_name}"
    if not hasattr(transformer, backup_key):
        setattr(transformer, backup_key, {})
    backup = getattr(transformer, backup_key)

    applied = 0
    skipped = 0
    for mod_path, d in deltas.items():
        for suffix in ("weight", "bias"):
            delta = d.get(suffix)
            if delta is None:
                continue
            # PEFT wraps low-rank-targeted Linears, renaming X.weight ->
            # X.base_layer.weight. Try both so diffs land on the real tensor
            # even when another adapter already wrapped the same module.
            target_key = f"{mod_path}.{suffix}"
            param = params.get(target_key)
            if param is None:
                wrapped_key = f"{mod_path}.base_layer.{suffix}"
                param = params.get(wrapped_key)
                if param is not None:
                    target_key = wrapped_key
            if param is None:
                skipped += 1
                continue
            dv = delta.to(dtype=torch.float32)
            if tuple(dv.shape) != tuple(param.shape):
                try:
                    dv = dv.reshape(param.shape)
                except RuntimeError:
                    print(f"{log_prefix}   diff shape mismatch for {target_key}: "
                          f"{tuple(delta.shape)} vs {tuple(param.shape)}, skipping")
                    skipped += 1
                    continue
            if target_key not in backup:
                backup[target_key] = param.data.clone()
            param.data.add_((dv * float(weight)).to(dtype=param.dtype,
                                                     device=param.device))
            applied += 1

    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    # Don't clobber a real PEFT LoraConfig if this adapter also loaded low-rank.
    cfg_key = adapter_name if adapter_name not in transformer.peft_config \
        else f"{adapter_name}__diff"
    transformer.peft_config[cfg_key] = {
        "_type": "diff_direct",
        "_applied_modules": applied,
        "_weight": float(weight),
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} diff/diff_b direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    return applied


def load_lora_with_key_fix(pipe, lora_path: str, adapter_name: str,
                          log_prefix: str = "[LoRA]",
                          weight: float = 1.0) -> None:
    """Load a LoRA / LoKR / LoHa adapter with automatic format detection.

    1. **Fast path** — tries ``pipe.load_lora_weights()`` (handles well-
       formatted standard LoRA files).
    2. **Fallback** — loads the state dict manually, normalises keys
       (strips ``transformer.`` prefix), and detects the adapter type:

       * **Standard LoRA** (``lora_A`` / ``lora_B`` keys) — re-loads
         through the pipeline with cleaned keys.
       * **LoKR** (``lokr_w1`` / ``lokr_w2`` keys) — injects via PEFT's
         ``inject_adapter_in_model`` + ``set_peft_model_state_dict``.
       * **LoHa** (``hada_w1_a`` / ``hada_w2_a`` keys) — same approach
         as LoKR but with ``LoHaConfig``.
    """
    # ── Fast path: try standard loading ──────────────────────────────
    transformer = getattr(pipe, "transformer", None)
    try:
        before = _live_lora_layer_count(transformer)
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        if _live_lora_layer_count(transformer) > before:
            return   # genuinely applied to the transformer
        # diffusers matched 0 keys and only WARNED (no exception). Remove the
        # phantom adapter and fall through to manual load + format detection.
        print(f"{log_prefix} Fast-path load matched 0 transformer modules "
              "(prefix/key mismatch); retrying with key normalisation...")
        _safe_delete_adapter(pipe, adapter_name, log_prefix)
    except Exception as e:
        # ANY fast-path failure (ValueError/RuntimeError/OSError from a .bin,
        # a key/format mismatch, an unreadable file, etc.) routes to the manual
        # loader + Krea conversion below. If that also fails it raises its own
        # clear error, so nothing is masked.
        _safe_delete_adapter(pipe, adapter_name, log_prefix)
        print(f"{log_prefix} Standard load failed ({type(e).__name__}: "
              f"{str(e)[:120]}); attempting manual load + key conversion...")

    # ── Fallback: manual load + format detection ─────────────────────
    state_dict = _load_state_dict(lora_path)

    # Original-format Krea LoRA (kohya 'lora_unet_' / Comfy 'diffusion_model.')
    # → rewrite keys to diffusers naming, then load through the pipeline so
    # set_adapters()/unload continue to work for per-stage weights. Full-weight
    # (.diff/.diff_b) patches are merged directly (sticky) since they're
    # full-rank and can't be a PEFT adapter.
    converted, direct_deltas, n_map, n_tot = _convert_krea_lora_to_diffusers(
        state_dict, model=transformer, log_prefix=log_prefix)
    if converted or direct_deltas:
        kinds = []
        if converted:
            kinds.append(f"{len(converted) // 2} low-rank")
        if direct_deltas:
            kinds.append(f"{len(direct_deltas)} full-diff")
        print(f"{log_prefix} Krea original-format LoRA: remapped {n_map}/{n_tot} "
              f"module(s) [{', '.join(kinds)}] to diffusers naming")
        loaded_any = False
        if converted:
            before = _live_lora_layer_count(transformer)
            try:
                pipe.load_lora_weights(converted, adapter_name=adapter_name)
            except (ValueError, RuntimeError) as e:
                print(f"{log_prefix} converted Krea LoRA pipeline load failed: "
                      f"{str(e)[:160]}")
            else:
                if _live_lora_layer_count(transformer) > before:
                    loaded_any = True
                else:
                    print(f"{log_prefix} converted Krea LoRA matched 0 modules")
                    _safe_delete_adapter(pipe, adapter_name, log_prefix)
        if direct_deltas:
            if _apply_krea_direct_deltas(pipe, direct_deltas, adapter_name,
                                         log_prefix, weight) > 0:
                loaded_any = True
        # This IS a recognised Krea LoRA: return whether or not anything applied
        # (never fall through to the generic detector, which would misreport a
        # .diff-only file as 'Unrecognised adapter format' and fail the stack).
        if not loaded_any:
            print(f"{log_prefix} WARNING: recognised as a Krea LoRA but nothing "
                  f"was applied (0 low-rank matched, 0 diffs merged); skipping.")
        return

    state_dict = _normalize_keys(state_dict, model=transformer)

    adapter_type = _detect_adapter_type(state_dict)
    print(f"{log_prefix} Detected adapter format: {adapter_type}")

    if adapter_type == "lokr":
        _load_lokr_adapter(pipe, state_dict, adapter_name, log_prefix,
                           weight=weight)
    elif adapter_type == "loha":
        _load_loha_adapter(pipe, state_dict, adapter_name, log_prefix,
                           weight=weight)
    elif adapter_type == "lora":
        # Standard LoRA with key prefix issues — multi-level fallback
        _load_lora_adapter(pipe, state_dict, adapter_name, log_prefix,
                           weight=weight)
    else:
        raise ValueError(
            f"{log_prefix} Unrecognised adapter format.  First 5 keys: "
            f"{list(state_dict.keys())[:5]}"
        )


def _diagnose_lora_compatibility(lora_path: str, transformer) -> str:
    """Analyze LoRA compatibility against a transformer without applying it.

    Returns a multi-line diagnostic report string.
    """
    lines = []
    filename = os.path.basename(lora_path)
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"LoRA Diagnostic: {filename}")
    lines.append(sep)

    # ── Load raw state dict ───────────────────────────────────────────
    try:
        sd = _load_state_dict(lora_path)
    except Exception as exc:
        return f"{sep}\nLoRA Diagnostic: {filename}\n{sep}\nERROR: Could not load file — {exc}\n{sep}"

    total_keys = len(sd)
    lines.append(f"Total tensor keys: {total_keys}")

    # Text-encoder keys (we don't apply these to the transformer)
    te_keys = [k for k in sd if k.startswith(("text_encoder.", "text_encoder_2."))]
    if te_keys:
        lines.append(f"Text encoder keys (not applied): {len(te_keys)}")

    # ── Adapter type detection ────────────────────────────────────────
    adapter_type = _detect_adapter_type(sd)
    lines.append(f"Adapter type: {adapter_type.upper()}")

    # ── Rank / alpha (LoRA only) ──────────────────────────────────────
    if adapter_type == "lora":
        sd_r = _rename_lora_down_up(sd)
        ranks, alphas = [], []
        for k, v in sd_r.items():
            if "lora_A" in k and v.ndim >= 2:
                ranks.append(v.shape[0])
            if k.endswith(".alpha") and v.numel() == 1:
                alphas.append(round(v.item(), 3))
        if ranks:
            unique_ranks = sorted(set(ranks))
            lines.append(f"Rank(s): {unique_ranks}")
            avg_rank = sum(ranks) / len(ranks)
            if avg_rank > 128:
                lines.append(f"  NOTE: High rank ({avg_rank:.0f}) — may cause instability")
        if alphas:
            unique_alphas = sorted(set(alphas))
            lines.append(f"Alpha(s): {unique_alphas}")
            if ranks:
                ratio = (sum(alphas) / len(alphas)) / (sum(ranks) / len(ranks))
                lines.append(f"  alpha/rank ratio: {ratio:.3f}  (1.0 = neutral)")
        if not ranks:
            lines.append("  WARNING: No lora_A tensors found — file may be malformed")
    elif adapter_type == "lokr":
        lines.append("Format: LoKR (Kronecker decomposition)")
    elif adapter_type == "loha":
        lines.append("Format: LoHa (Hadamard product)")
        for k, v in sd.items():
            if ".hada_w1_b" in k and v.ndim >= 2:
                lines.append(f"  Rank (from w1_b): {v.shape[0]}")
                break
    else:
        lines.append("WARNING: Unrecognised format — corrupt file or unsupported training tool.")
        lines.append(f"  First 5 raw keys: {list(sd.keys())[:5]}")

    # ── Key-prefix detection (before normalisation) ───────────────────
    raw_paths = {_adapter_module_path(k) for k in sd
                 if not k.startswith(("text_encoder.", "text_encoder_2."))}
    found_prefixes = [p for p in
                      ("transformer.", "diffusion_model.", "model.diffusion_model.", "model.", "unet.")
                      if any(rp.startswith(p) for rp in raw_paths)]
    if found_prefixes:
        lines.append(f"Key prefix(es) auto-stripped: {found_prefixes}")

    # ── Normalise keys and match against model ────────────────────────
    sd_norm = _normalize_keys(sd, model=transformer)
    sd_module_paths = {_adapter_module_path(k) for k in sd_norm}
    model_module_names = {name for name, _ in transformer.named_modules() if name}

    matched = sd_module_paths & model_module_names
    unmatched = sd_module_paths - model_module_names
    n_total = len(sd_module_paths)
    n_matched = len(matched)
    match_pct = n_matched / n_total * 100 if n_total > 0 else 0.0

    lines.append("")
    lines.append(f"Adapter modules: {n_total}")
    lines.append(f"Matched modules:  {n_matched} ({match_pct:.0f}%)")

    # ── Verdict ───────────────────────────────────────────────────────
    lines.append("")
    if match_pct >= 75:
        lines.append("VERDICT: COMPATIBLE")
        lines.append("Most adapter modules match the transformer architecture.")
        if n_total > n_matched:
            lines.append(f"  {n_total - n_matched} unmatched module(s) will be skipped.")
    elif match_pct >= 30:
        lines.append("VERDICT: PARTIAL MATCH  \u26a0")
        lines.append(f"Only {match_pct:.0f}% of modules matched.  May produce partial effects or artifacts.")
        lines.append("If output looks like noise, this LoRA targets a different architecture variant.")
        if unmatched:
            lines.append(f"  Unmatched sample: {sorted(unmatched)[:5]}")
    else:
        lines.append("VERDICT: INCOMPATIBLE  \u2717")
        lines.append(f"Only {match_pct:.0f}% of modules matched — this LoRA was almost certainly")
        lines.append("trained for a DIFFERENT architecture (e.g. SDXL, SD1.5, Flux).")
        lines.append("Applying it will likely produce corrupted output (pure noise / artifacts).")
        lines.append(f"  Sample LoRA modules:  {sorted(sd_module_paths)[:5]}")
        lines.append(f"  Sample model modules: {sorted(model_module_names)[:5]}")

    lines.append(sep)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
#  Krea2 stack orchestration (Eric_Krea2 additions, not from the Qwen node)
# ─────────────────────────────────────────────────────────────────────────────
# Declarative-stack / ephemeral model used by the Krea2 Apply-LoRA nodes + the
# Multi-Stage Ultra node:
#   * Apply nodes only DECLARE {lora, per-stage weights} into pipeline["lora_stack"].
#   * The generate node calls realize_lora_stack() once (unload-all -> load the
#     declared stack fresh), set_stack_stage_weights() before each stage, and
#     unload_all_loras() in a finally (ephemeral -> cleared after every run, even
#     on cancel/error). Net: adapters exist only during the generation call, so
#     there are no stale or orphaned adapters across runs/workflows.

def unload_all_loras(pipe, log_prefix="[EricKrea2-LoRA]"):
    """Remove all LoRA effects from the pipeline/transformer.

    Two mechanisms are cleared:
      * PEFT adapters (low-rank LoRAs) via ``unload_lora_weights``;
      * sticky direct merges (``.diff``/``.diff_b`` full-weight patches and the
        LoKR/LoHa direct fallbacks) by restoring the ``_lora_backup_<adapter>``
        snapshots taken at load time — this prevents deltas from accumulating
        across runs (each generate() realizes the stack fresh)."""
    transformer = getattr(pipe, "transformer", None)

    # 1) Restore sticky direct-merge backups (baked into base weights).
    if transformer is not None:
        backup_attrs = [a for a in list(vars(transformer).keys())
                        if a.startswith("_lora_backup_")]
        restored = 0
        if backup_attrs:
            params = dict(transformer.named_parameters())
            for attr in backup_attrs:
                backup = getattr(transformer, attr, None) or {}
                for tkey, orig in backup.items():
                    p = params.get(tkey)
                    if p is not None:
                        p.data.copy_(orig.to(dtype=p.dtype, device=p.device))
                        restored += 1
                try:
                    delattr(transformer, attr)
                except Exception:
                    pass
        # Drop direct-merge markers so peft_config reflects reality.
        peft_cfg = getattr(transformer, "peft_config", None)
        if isinstance(peft_cfg, dict):
            for name in [n for n, c in list(peft_cfg.items())
                         if isinstance(c, dict)
                         and str(c.get("_type", "")).endswith("_direct")]:
                peft_cfg.pop(name, None)
        if restored:
            print(f"{log_prefix} restored {restored} direct-merged param(s) "
                  "to base weights")

    # 2) Remove PEFT adapters.
    try:
        pipe.unload_lora_weights()
        print(f"{log_prefix} unloaded all LoRA adapters")
    except Exception as e:
        print(f"{log_prefix} unload note: {e}")


def _registered_adapters(pipe):
    names = set()
    try:
        for comp_adapters in pipe.get_list_adapters().values():
            names.update(comp_adapters)
    except Exception:
        pass
    return names


def verify_lora_active(pipe, log_prefix="[EricKrea2-LoRA]") -> dict:
    """Inspect the LIVE transformer and report whether PEFT LoRA layers are
    actually present and active. This is the definitive 'is it doing anything?'
    check: it counts injected tuner layers, the adapters currently switched on,
    and their effective scaling. Returns a summary dict and prints a one-line
    verdict.

    A healthy result looks like:
        active layers=336, adapters=['my_lora'], scales={'my_lora': 1.0}
    'layers=0' means NOTHING is applied to the transformer (wiring problem,
    incompatible LoRA, or a direct-merge fallback that PEFT could not inject).
    """
    transformer = getattr(pipe, "transformer", None)
    summary = {
        "lora_layers": 0,        # number of PEFT tuner layers holding LoRA weights
        "active_adapters": [],   # adapters currently switched ON
        "scales": {},            # adapter_name -> effective scaling on a sample layer
        "direct_merge": [],      # adapters applied via the sticky direct-merge fallback
    }
    if transformer is None:
        print(f"{log_prefix} verify: no transformer on pipe.")
        return summary

    active = set()
    layer_count = 0
    for module in transformer.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or not scaling:
            continue
        # This is a PEFT BaseTunerLayer holding one or more LoRA adapters.
        layer_count += 1
        act = getattr(module, "active_adapters", None)
        if callable(act):
            try:
                act = act()
            except Exception:
                act = None
        if isinstance(act, (list, set, tuple)):
            active.update(act)
        # Capture a representative scale per adapter (first layer we see it on).
        for name, val in scaling.items():
            if name not in summary["scales"]:
                try:
                    summary["scales"][name] = float(val)
                except (TypeError, ValueError):
                    summary["scales"][name] = val

    # Direct-merge adapters live only in peft_config with a _direct marker.
    peft_cfg = getattr(transformer, "peft_config", {}) or {}
    for name, cfg in peft_cfg.items():
        if isinstance(cfg, dict) and str(cfg.get("_type", "")).endswith("_direct"):
            summary["direct_merge"].append(name)

    summary["lora_layers"] = layer_count
    summary["active_adapters"] = sorted(active)

    if layer_count == 0 and not summary["direct_merge"]:
        print(f"{log_prefix} verify: NO LoRA layers on the transformer -> the LoRA is "
              "NOT affecting generation. Check that the Apply-LoRA output pipeline is "
              "wired into the generate node, and run the Diagnose LoRA node.")
    else:
        scale_txt = ", ".join(f"{n}={v}" for n, v in summary["scales"].items()) or "n/a"
        print(f"{log_prefix} verify: {layer_count} PEFT LoRA layer(s) live, "
              f"active={summary['active_adapters'] or 'none-switched-on'}, scales=[{scale_txt}]"
              + (f", direct-merge={summary['direct_merge']}" if summary["direct_merge"] else ""))
    return summary


def realize_lora_stack(pipe, stack, log_prefix="[EricKrea2-LoRA]"):
    """Unload everything, then load exactly the declared stack fresh. Returns the
    list of PEFT-registered adapter names (controllable via set_adapters, removable
    via unload). Entries that fell back to a direct weight-merge are NOT returned
    (sticky: weight baked at load, survive unload)."""
    unload_all_loras(pipe, log_prefix)   # clean slate -> kills stale adapters by construction
    if not stack:
        return []
    registered = []
    for entry in stack:
        path = entry["path"]
        adapter_name = entry["adapter_name"]
        # load at the S1 weight so a sticky direct-merge bakes a sensible value
        init_w = float(entry.get("weight_s1", entry.get("strength", 1.0)))
        try:
            load_lora_with_key_fix(pipe, path, adapter_name, log_prefix=log_prefix, weight=init_w)
        except Exception as e:
            print(f"{log_prefix} FAILED to load {entry.get('filename', adapter_name)}: {e}")
            continue
        # Classify how the LoRA actually attached, using the LIVE transformer
        # (not just registration, which diffusers sets even on a 0-key no-op).
        transformer = getattr(pipe, "transformer", None)
        peft_cfg = getattr(transformer, "peft_config", {}) or {}
        cfg = peft_cfg.get(adapter_name, {})
        is_direct = isinstance(cfg, dict) and str(cfg.get("_type", "")).endswith("_direct")
        live = _live_lora_layer_count(transformer, adapter_name)
        if is_direct:
            print(f"{log_prefix} NOTE: '{entry.get('filename', adapter_name)}' applied via direct "
                  "weight-merge (full-rank .diff/.diff_b patch or PEFT fallback; baked at the S1 "
                  "weight so it won't respond to per-stage weight changes, but it IS cleared on "
                  "unload at the end of the run).")
        elif live > 0:
            registered.append(adapter_name)
        else:
            print(f"{log_prefix} WARNING: '{entry.get('filename', adapter_name)}' attached to 0 "
                  "transformer modules — it is NOT affecting generation. Run the Diagnose LoRA node; "
                  "it likely targets a different architecture or uses unrecognised key names.")
    # Definitive runtime check: are LoRA layers actually live on the transformer?
    verify_lora_active(pipe, log_prefix)
    return registered


def set_stack_stage_weights(pipe, stack, registered, stage_idx, log_prefix="[EricKrea2-LoRA]"):
    """Set adapter scales for stage_idx (1/2/3) from each entry's per-stage weight."""
    if not registered:
        return
    key = f"weight_s{stage_idx}"
    names, weights = [], []
    for entry in stack:
        if entry["adapter_name"] not in registered:
            continue
        names.append(entry["adapter_name"])
        weights.append(float(entry.get(key, entry.get("strength", 1.0))))
    if not names:
        return
    try:
        pipe.set_adapters(names, weights)
        print(f"{log_prefix} stage {stage_idx} weights: " +
              ", ".join(f"{n}={w:.2f}" for n, w in zip(names, weights)))
    except Exception as e:
        print(f"{log_prefix} set_adapters (stage {stage_idx}) note: {e}")
