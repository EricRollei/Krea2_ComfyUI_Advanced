# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Component Loader
===========================
Assemble a Krea2Pipeline from a base diffusers folder (configs, scheduler,
tokenizer, and any component you don't override) plus optional per-component
overrides for the transformer, VAE, and text encoder. Each override auto-detects
its format:

  transformer_path : diffusers folder | bare .safetensors/.bin/.pt | .gguf
  text_encoder_path: HF/transformers folder | bare .safetensors | .gguf
  vae_path         : diffusers folder | bare .safetensors

GGUF here is Path A (dequantize-on-load): weights are dequantized to the compute
dtype and loaded full-precision into the diffusers module (no VRAM savings, but
loads any GGUF-only finetune). See _gguf_utils for the transformer remap.

Requires a diffusers build with Krea2Pipeline. Text-encoder class is
transformers.Qwen3VLModel (per the base model_index.json).

Author: Eric Hiss (GitHub: EricRollei)
"""

import json
import os
import time
import torch
import gc
import builtins
import weakref

from .._compat import apply_attention_backend
from .. import _settings

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class _Krea2PipelineHandle(dict):
    """Plain-dict wrapper for the loader's output that ALSO supports weakref (a bare
    dict does not). This lets the loader keep only a *weak* reference to the exact
    object ComfyUI caches as this node's output, so with keep_in_vram=False we can
    later null its 'pipeline' key and let VRAM be reclaimed - without our module
    holding a strong reference that would silently pin the weights resident."""


def _raise_if_interrupted():
    """Cooperative cancel checkpoint. ComfyUI's Stop only sets a flag that is checked
    at explicit points; the loader's phases are long BLOCKING native calls with none,
    so we poll the flag at phase boundaries to allow bailing out BEFORE the next
    multi-GB step (e.g. the CPU->VRAM transfer). Can't interrupt a single transfer
    mid-flight, but lets 'Stop' actually take effect between phases."""
    try:
        import comfy.model_management as _mm
        _mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _norm_path(p):
    """Canonicalize a path for cache-key comparison so cosmetic differences don't
    force a spurious reload. Two ComfyUI tabs (or a preset vs. a hand-typed path)
    can point at the SAME model with strings that differ only by trailing slash,
    `/` vs `\\`, or drive-letter casing (`A:\\` vs `a:\\`); without this each variant
    hashes to a different cache_key -> cache miss -> a second copy stacked in VRAM
    (ComfyUI still pins the first via its own node-output cache). normcase+normpath
    collapses all of those to one key. Empty/blank stays "" (an unset override)."""
    p = (p or "").strip()
    if not p:
        return ""
    try:
        return os.path.normcase(os.path.normpath(p))
    except Exception:
        return p

# Separate cache from the plain loader (keyed by every component path + opts).
# Anchored on `builtins` so it SURVIVES module hot-reloads / re-imports. ComfyUI
# dev tooling (and LG_HotReload) re-executes custom-node modules on some queues,
# which would otherwise reset this dict to {pipeline: None} and force a full,
# memory-churning reassembly every run - the cause of both "reloads every run
# despite keep_in_vram" and the fp8-dequant `Fatal Python error: Aborted` after
# a few of those rebuilds. Recovering the same dict across reloads makes
# keep_in_vram actually persistent.
_COMP_CACHE = getattr(builtins, "_ERIC_KREA2_COMP_CACHE", None)
if not isinstance(_COMP_CACHE, dict):
    _COMP_CACHE = {"pipeline": None, "cache_key": None, "is_distilled": False}
    builtins._ERIC_KREA2_COMP_CACHE = _COMP_CACHE


def _free_current(cache, log=None):
    """Drop the cached pipeline reference and empty the CUDA cache. We deliberately do
    NOT move it to CPU: that balloons system RAM and, under memory pressure, was crashing
    the fp8 dequant of the next model (Windows access violation). VRAM is reclaimed by the
    allocator once the last reference is gone."""
    # Release a prior keep_in_vram=False output that is still pinned only by
    # ComfyUI's node-output cache (we kept just a weakref to it), so a rebuild
    # doesn't briefly hold two model copies in VRAM.
    _h = getattr(builtins, "_ERIC_KREA2_LAST_HANDLE", None)
    if _h is not None:
        try:
            _d = _h()
            if isinstance(_d, dict) and _d.get("pipeline") is not None:
                _d["pipeline"] = None
                if log:
                    log("released previous (uncached) pipeline held by ComfyUI's node cache")
        except Exception:
            pass
        builtins._ERIC_KREA2_LAST_HANDLE = None
    p = cache.get("pipeline")
    cache["pipeline"] = None
    cache["cache_key"] = None
    if p is not None:
        del p
        if log:
            log("released previous pipeline reference")
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _read_is_distilled(base_path):
    try:
        with open(os.path.join(base_path, "model_index.json"), "r", encoding="utf-8") as f:
            return bool(json.load(f).get("is_distilled", False))
    except Exception:
        return False


def _load_single_file_sd(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


# ── ComfyUI native quant support (fp8-scaled / INT8 / INT8-ConvRot) ───────────

_HADAMARD_CACHE = {}


def _regular_hadamard(size, device):
    """Normalized REGULAR Hadamard matrix as used by ConvRot (mirrors
    comfy_kitchen.tensor.int8_utils._build_hadamard): Kronecker powers of the 4x4
    regular Hadamard base, scaled by 1/sqrt(size). This matrix is symmetric AND
    orthogonal (an involution: H @ H = I), which is why comfy-kitchen uses the
    SAME rotation op for both quantize and dequantize."""
    key = (size, str(device))
    h = _HADAMARD_CACHE.get(key)
    if h is not None:
        return h
    n = size
    while n >= 4 and n % 4 == 0:
        n //= 4
    if n != 1 or size < 4:
        raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=torch.float32, device=device)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h
    return h


def _convrot_unrotate(w, group_size):
    """Undo the ConvRot group-wise rotation on a dequantized fp32 2D weight.
    Storage-time rotation is W_rot = W @ blockdiag(H)^T per 256-wide input group
    (comfy_kitchen int8_utils._rotate_weight); H is its own inverse (symmetric
    orthogonal) so applying the identical op once more restores W."""
    out_f, in_f = w.shape
    n_groups = in_f // group_size
    h = _regular_hadamard(group_size, w.device)
    w = torch.matmul(w.reshape(out_f, n_groups, group_size), h.T.to(w.dtype))
    return w.reshape(out_f, in_f)


def _strip_quant_prefix(name):
    for pre in ("model.diffusion_model.", "diffusion_model.", "model."):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def _read_safetensors_quant_meta(path):
    """Per-layer quant config from the safetensors __metadata__
    ('_quantization_metadata' -> {"layers": {module: conf}}), which the native
    ComfyUI INT8/ConvRot converters write. safetensors.load_file drops
    __metadata__, so parse the 8-byte-length + JSON header directly (no tensor
    data is read). Returns {} when absent."""
    if not (path or "").endswith(".safetensors"):
        return {}
    try:
        import struct
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        qm = (hdr.get("__metadata__") or {}).get("_quantization_metadata")
        if not qm:
            return {}
        layers = json.loads(qm).get("layers", {})
        return layers if isinstance(layers, dict) else {}
    except Exception:
        return {}


def _decode_comfy_quant_marker(t):
    """Decode a '<module>.comfy_quant' marker tensor (JSON bytes; canonically
    uint8, but some converters save the byte values as bf16 - 0..255 are exactly
    representable there) -> conf dict, or None."""
    try:
        v = t.flatten()
        if v.dtype != torch.uint8:
            v = v.to(torch.float32).round().clamp(0, 255).to(torch.uint8)
        return json.loads(bytes(v.tolist()).decode("utf-8"))
    except Exception:
        return None


def _layer_quant_confs(sd, path=None, log=None):
    """{stripped module name: quant conf} merged from per-layer .comfy_quant
    markers and the file-level _quantization_metadata (which wins on conflict).
    Names are prefix-stripped so lookups from tensor keys line up regardless of
    'model.diffusion_model.' style prefixes."""
    confs = {}
    for k, v in sd.items():
        if k.endswith(".comfy_quant"):
            c = _decode_comfy_quant_marker(v)
            if isinstance(c, dict):
                confs[_strip_quant_prefix(k[:-len(".comfy_quant")])] = c
    if path:
        for name, c in _read_safetensors_quant_meta(path).items():
            if isinstance(c, dict):
                confs[_strip_quant_prefix(name)] = c
    if log and confs:
        n_rot = sum(1 for c in confs.values() if _conf_convrot(c)[0])
        log(f"quant metadata: {len(confs)} quantized layer(s), {n_rot} ConvRot")
    return confs


def _conf_convrot(conf):
    """(convrot: bool, groupsize: int) from a layer conf, honouring the nested
    'params' fallback the native ComfyUI loader also accepts (comfy/ops.py)."""
    if not isinstance(conf, dict):
        return False, 256
    params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
    cr = bool(conf.get("convrot", params.get("convrot", False)))
    gs = int(conf.get("convrot_groupsize", params.get("convrot_groupsize", 256)))
    return cr, gs


def _dequant_comfy_quant(sd, dtype, log=None, device=None, quant_confs=None):
    """Dequantize ComfyUI-quantized tensors to `dtype`:

      * scaled fp8 / INT8:  X.weight [F8_E4M3 | I8] * X.weight_scale
        ([] tensorwise scalar or [out, 1] row-wise) -> dtype
      * INT8 ConvRot:       as above, then the group-wise regular-Hadamard
        un-rotation (_convrot_unrotate). Which layers are rotated comes from
        `quant_confs` (see _layer_quant_confs) - without the un-rotation a
        ConvRot checkpoint decodes to pure noise.
      * everything else:    cast to dtype (bare fp8 checkpoints etc.)

    weight_scale + comfy_quant marker tensors are dropped. Returns
    (new_sd, n_dequantized). Prefix-agnostic.

    The heavy math runs on `device` (GPU) when one is given and CUDA is
    available: CPU has NO vectorized float8 conversion kernel, so host-side
    upcasts are the dominant load-time cost for fp8 checkpoints (minutes, with
    no VRAM spike - the exact symptom). On GPU it's near-instant, and the
    ConvRot un-rotation (an [out, groups, g] @ [g, g] matmul per layer) is
    likewise trivial there. Each result is copied back to the host so the
    assembled state dict stays CPU-side for load_state_dict into the (still
    CPU/meta) model. On any GPU error we fall back to CPU math."""
    quant_confs = quant_confs or {}
    use_gpu = False
    if device is not None and str(device) != "cpu":
        try:
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
    dev = device if use_gpu else "cpu"
    out = {}
    n = 0
    n_rot = 0
    n_cast = 0
    for k, v in sd.items():
        if k.endswith((".weight_scale", ".comfy_quant")):
            continue
        skey = k + "_scale"
        if k.endswith(".weight") and skey in sd:
            convrot, gs = _conf_convrot(quant_confs.get(_strip_quant_prefix(k[:-len(".weight")])))

            def _deq(target_dev):
                w = v.to(target_dev, torch.float32) * sd[skey].to(target_dev, torch.float32)
                if convrot:
                    if w.ndim == 2 and w.shape[1] % gs == 0:
                        return _convrot_unrotate(w, gs), True
                    if log:
                        log(f"WARNING: ConvRot layer {k} has shape {tuple(w.shape)} "
                            f"incompatible with group size {gs}; loaded UN-rotated "
                            f"(expect artifacts)")
                return w, False

            if use_gpu:
                try:
                    w, rotated = _deq(dev)
                    out[k] = w.to(dtype).cpu()
                    del w
                except Exception as e:
                    if log:
                        log(f"GPU dequant failed ({e}); falling back to CPU for this weight")
                    w, rotated = _deq("cpu")
                    out[k] = w.to(dtype)
            else:
                w, rotated = _deq("cpu")
                out[k] = w.to(dtype)
            n += 1
            if rotated:
                n_rot += 1
        else:
            if v.dtype == torch.int8 and k.endswith(".weight") and log:
                log(f"WARNING: INT8 tensor {k} has no companion weight_scale - casting "
                    f"raw int8 values (checkpoint is probably broken/unsupported)")
            # Plain (unscaled) tensor: cast to the target dtype. When the source is a
            # low-precision type the CPU has no vectorized conversion for (a "bare fp8"
            # checkpoint with no companion _scale tensors), route the cast through the GPU
            # too - that CPU upcast is otherwise the hidden multi-minute cost. Tensors
            # already in the target dtype are a no-op and stay on the host.
            if use_gpu and v.dtype != dtype:
                try:
                    out[k] = v.to(dev, dtype).cpu()
                    n_cast += 1
                except Exception:
                    out[k] = v.to(dtype)
            else:
                out[k] = v.to(dtype)
    if use_gpu:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if log and (n or n_cast):
        log(f"comfy quant: dequantized {n} scaled ({n_rot} ConvRot-unrotated) + "
            f"cast {n_cast} plain weights on {'GPU' if use_gpu else 'CPU'}")
    return out, n


# Back-compat alias for the pre-INT8 name.
_dequant_comfy_fp8 = _dequant_comfy_quant


# ── fast (meta-device) model instantiation ───────────────────────────────

def _instantiate_empty(build_fn, log=None, label="model"):
    """Instantiate a model skeleton with all *parameters* on the meta device:
    zero RAM allocation and - crucially - zero random-weight initialization.
    That init pass is otherwise the dominant load cost: from_config /
    Qwen3VLModel(cfg) run full kaiming / trunc-normal initializers on CPU in
    fp32 for weights load_state_dict immediately overwrites (~50s for the 12.8B
    transformer, ~215s for the 4B Qwen3-VL). include_buffers=False keeps
    buffers real so computed values (rotary inv_freq etc., usually absent from
    checkpoints) stay correct. Returns (module, on_meta)."""
    try:
        from accelerate import init_empty_weights
    except Exception as e:
        if log:
            log(f"accelerate unavailable ({e}); {label} falling back to slow full init")
        return build_fn(), False
    with init_empty_weights(include_buffers=False):
        return build_fn(), True


def _load_into(module, sd, dtype, on_meta, label="model"):
    """load_state_dict with assign=True when params are on meta (you cannot
    copy_ into meta tensors; assign adopts the checkpoint tensors directly,
    which also skips a full copy). With meta init a missing key is NOT
    'randomly initialized' - it stays meta and crashes (or silently corrupts)
    at inference - so verify none were left behind and fail loudly. Finally
    cast the module to `dtype`: a no-op for params the dequant step already
    cast, converts real buffers exactly like the old pre-load .to(dtype) did,
    and fixes up the fp32 params of the non-meta fallback path.
    Returns (module, missing, unexpected)."""
    missing, unexpected = module.load_state_dict(sd, strict=False, assign=on_meta)
    if on_meta:
        metas = [pn for pn, p in module.named_parameters() if p.is_meta]
        if metas:
            raise RuntimeError(
                f"{label}: {len(metas)} parameter(s) missing from the checkpoint were "
                f"left uninitialized (meta device); cannot proceed. Sample: "
                + ", ".join(metas[:8]))
    module = module.to(dtype)
    return module, missing, unexpected


def _load_qwen3vl_single_file(path, dtype, log, device=None):
    """Load a single-file Qwen3-VL checkpoint into a Qwen3VLModel-compatible state dict:
    dequantize ComfyUI-quantized weights (fp8-scaled / INT8 / INT8-ConvRot, on `device`
    when CUDA) and strip the CausalLM 'model.' prefix."""
    raw_sd = _load_single_file_sd(path)
    qconfs = _layer_quant_confs(raw_sd, path=path, log=log)
    deq, n_deq = _dequant_comfy_quant(raw_sd, dtype, log=log, device=device,
                                      quant_confs=qconfs)
    del raw_sd
    out = {(k[6:] if k.startswith("model.") else k): v for k, v in deq.items()}
    log(f"text encoder single-file: dequantized {n_deq} quantized weights, "
        f"stripped 'model.' prefix -> {len(out)} tensors")
    return out


def _detect_format(path):
    """none | gguf | single_file | subfolder(<name>/) | direct(folder)."""
    if not path or not path.strip():
        return "none"
    path = path.strip()
    if os.path.isfile(path):
        if path.endswith(".gguf"):
            return "gguf"
        if path.endswith((".safetensors", ".bin", ".pt", ".pth")):
            return "single_file"
        return "unknown"
    if os.path.isdir(path):
        return "dir"
    return "unknown"


class EricKrea2ComponentLoader:
    CATEGORY = "Eric/Krea2"
    FUNCTION = "load"
    RETURN_TYPES = ("KREA2_PIPELINE", "STRING")
    RETURN_NAMES = ("krea2_pipeline", "settings")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_pipeline_path": ("STRING", {
                    "default": r"H:\Training\Krea-2-Raw",
                    "tooltip": "Full Krea 2 diffusers folder. Supplies model_index.json, scheduler, "
                               "tokenizer, and any component you don't override below."}),
            },
            "optional": {
                "loader_preset": (_settings.list_preset_names("loader"), {"default": "custom",
                    "tooltip": "Load a saved loader recipe (paths / precision / attention / device) "
                               "into the panel for this run. 'custom' = use the panel as-is. Use the "
                               "★ Save Preset button to add one; the list refreshes on graph reload."}),
                "transformer_path": ("STRING", {"default": "",
                    "tooltip": "Override transformer: diffusers folder, bare .safetensors/.bin/.pt, "
                               "or .gguf (dequantized on load). Empty = use base."}),
                "text_encoder_path": ("STRING", {"default": "",
                    "tooltip": "Override Qwen3-VL text encoder: HF/transformers folder, bare "
                               ".safetensors, or .gguf. Empty = use base."}),
                "vae_path": ("STRING", {"default": "",
                    "tooltip": "Override VAE (AutoencoderKLQwenImage): diffusers folder or bare "
                               ".safetensors. Empty = use base."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16",
                    "tooltip": "Compute dtype (bf16 recommended for Blackwell). GGUF is dequantized "
                               "to this dtype."}),
                "attention_backend": (["auto", "flash", "sage", "sdpa"], {"default": "auto",
                    "tooltip": "Transformer attention kernel. auto = flash if available else SDPA."}),
                "device": (["cuda", "cuda:0", "cuda:1", "cpu"], {"default": "cuda"}),
                "keep_in_vram": ("BOOLEAN", {"default": True,
                    "tooltip": "Cache the assembled pipeline between runs."}),
                "offload_vae": ("BOOLEAN", {"default": False,
                    "tooltip": "Keep the VAE on CPU during transformer inference (saves ~1 GB)."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, base_pipeline_path, transformer_path="", text_encoder_path="",
                   vae_path="", precision="bf16", attention_backend="auto",
                   device="cuda", keep_in_vram=True, offload_vae=False, loader_preset="custom"):
        # Cheap, GPU-free check (no dequant/segfault risk - see §8 of the session
        # handoff for why an earlier unconditional IS_CHANGED=nan was removed).
        # If our cache is empty (fresh start, or EricKrea2UnloadModels just ran),
        # force exactly one rebuild by returning a value that always differs from
        # last time. Otherwise return the stable cache_key so unchanged widget
        # values keep reusing the cached pipeline (no rebuild storms).
        if _COMP_CACHE.get("pipeline") is None:
            import time
            return time.time()
        return _COMP_CACHE.get("cache_key")

    # ── component override builders ──────────────────────────────────────

    def _override_transformer(self, base, path, dtype, log, device="cpu"):
        from diffusers import Krea2Transformer2DModel
        fmt = _detect_format(path)
        log(f"transformer override ({fmt}): {path}")
        if fmt == "gguf":
            from .._gguf_utils import load_krea2_transformer_from_gguf
            return load_krea2_transformer_from_gguf(base, path, dtype, log=log)
        if fmt == "single_file":
            from .._gguf_utils import remap_krea2_state_dict
            _t = time.perf_counter()
            cfg = Krea2Transformer2DModel.load_config(base, subfolder="transformer")
            m, on_meta = _instantiate_empty(
                lambda: Krea2Transformer2DModel.from_config(cfg), log=log, label="transformer")
            log(f"  [timing] model init ({'meta' if on_meta else 'full'}): "
                f"{time.perf_counter() - _t:.1f}s")
            n_layers = getattr(m.config, "num_layers", None) or sum(
                1 for n, _ in m.named_modules()
                if n.startswith("transformer_blocks.") and n.count(".") == 1)
            _t = time.perf_counter()
            raw_sd = _load_single_file_sd(path)
            log(f"  [timing] disk load: {time.perf_counter() - _t:.1f}s")
            _t = time.perf_counter()
            qconfs = _layer_quant_confs(raw_sd, path=path, log=log)
            raw, n_deq = _dequant_comfy_quant(raw_sd, dtype, log=log, device=device,
                                              quant_confs=qconfs)
            del raw_sd
            log(f"  [timing] dequant/cast: {time.perf_counter() - _t:.1f}s")
            _t = time.perf_counter()
            sd = remap_krea2_state_dict(raw, n_layers, log=log)
            # dequant already produced `dtype` tensors; avoid a redundant full-dict copy.
            sd = {k: (v if v.dtype == dtype else v.to(dtype)) for k, v in sd.items()}
            log(f"  [timing] remap: {time.perf_counter() - _t:.1f}s")
            _t = time.perf_counter()
            m, missing, unexpected = _load_into(m, sd, dtype, on_meta, label="transformer")
            log(f"  [timing] load_state_dict: {time.perf_counter() - _t:.1f}s")
            log(f"transformer single-file: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                log("  MISSING sample: " + ", ".join(list(missing)[:10]))
            if unexpected:
                log("  UNEXPECTED sample: " + ", ".join(list(unexpected)[:10]))
            return m
        if fmt == "dir":
            if os.path.isdir(os.path.join(path, "transformer")):
                return Krea2Transformer2DModel.from_pretrained(
                    path, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
            return Krea2Transformer2DModel.from_pretrained(
                path, torch_dtype=dtype, local_files_only=True)
        raise ValueError(f"Cannot determine transformer format: {path}")

    def _override_vae(self, base, path, dtype, log):
        from diffusers import AutoencoderKLQwenImage
        fmt = _detect_format(path)
        log(f"VAE override ({fmt}): {path}")
        if fmt == "single_file":
            cfg = AutoencoderKLQwenImage.load_config(base, subfolder="vae")
            m, on_meta = _instantiate_empty(
                lambda: AutoencoderKLQwenImage.from_config(cfg), log=log, label="vae")
            m, _missing, _unexpected = _load_into(
                m, _load_single_file_sd(path), dtype, on_meta, label="vae")
            return m
        if fmt == "dir":
            if os.path.isdir(os.path.join(path, "vae")):
                return AutoencoderKLQwenImage.from_pretrained(
                    path, subfolder="vae", torch_dtype=dtype, local_files_only=True)
            return AutoencoderKLQwenImage.from_pretrained(
                path, torch_dtype=dtype, local_files_only=True)
        raise ValueError(f"Unsupported VAE format ({fmt}): {path}")

    def _override_text_encoder(self, base, path, dtype, log, device="cpu"):
        try:
            from transformers import Qwen3VLModel
        except Exception as e:
            raise RuntimeError(
                "transformers.Qwen3VLModel is unavailable in this build; cannot override the "
                f"Krea2 text encoder. Update transformers. ({e})")
        fmt = _detect_format(path)
        log(f"text encoder override ({fmt}): {path}")
        te_base = os.path.join(base, "text_encoder")
        if fmt == "gguf":
            try:
                te = Qwen3VLModel.from_pretrained(
                    te_base, gguf_file=path, torch_dtype=dtype, local_files_only=True)
            except Exception as e:
                raise RuntimeError(
                    "GGUF text-encoder load failed - transformers may not yet support the qwen3vl "
                    f"GGUF arch in this build. Use a .safetensors or HF folder for now. ({e})")
        elif fmt == "single_file":
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(te_base, local_files_only=True)
            _t = time.perf_counter()

            def _build_te():
                try:
                    return Qwen3VLModel._from_config(cfg)
                except Exception:
                    return Qwen3VLModel(cfg)

            te, on_meta = _instantiate_empty(_build_te, log=log, label="text encoder")
            log(f"  [timing] TE init ({'meta' if on_meta else 'full'}): "
                f"{time.perf_counter() - _t:.1f}s")
            _t = time.perf_counter()
            sd = _load_qwen3vl_single_file(path, dtype, log, device=device)
            log(f"  [timing] TE disk+dequant: {time.perf_counter() - _t:.1f}s")
            _t = time.perf_counter()
            te, missing, unexpected = _load_into(te, sd, dtype, on_meta, label="text encoder")
            log(f"  [timing] TE load_state_dict: {time.perf_counter() - _t:.1f}s")
            log(f"  TE load: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                log("  TE MISSING sample: " + ", ".join(list(missing)[:8]))
            if unexpected:
                log("  TE UNEXPECTED sample: " + ", ".join(list(unexpected)[:8]))
        elif fmt == "dir":
            sub = "text_encoder" if os.path.isdir(os.path.join(path, "text_encoder")) else ""
            # A CausalLM folder (Qwen3VLForConditionalGeneration) loads into Qwen3VLModel fine;
            # transformers strips the LM head and warns - that's expected.
            te = Qwen3VLModel.from_pretrained(
                path, subfolder=sub, torch_dtype=dtype, local_files_only=True)
        else:
            raise ValueError(f"Unsupported text encoder format ({fmt}): {path}")
        self._warn_te_dim(base, te, log)
        return te

    def _warn_te_dim(self, base, te, log):
        def _hs(cfg):
            return (getattr(cfg, "hidden_size", None)
                    or getattr(getattr(cfg, "text_config", None), "hidden_size", None))
        try:
            from transformers import AutoConfig
            base_cfg = AutoConfig.from_pretrained(
                os.path.join(base, "text_encoder"), local_files_only=True)
            bh, th = _hs(base_cfg), _hs(te.config)
            if bh and th and bh != th:
                log(f"WARNING: text encoder hidden_size {th} != base {bh}. Krea2's text-fusion "
                    "projector was trained for the base encoder's dim, so a different-size encoder "
                    "(e.g. Qwen3-VL-8B vs the 4B base) will mismatch and fail downstream. "
                    "Use a 4B Qwen3-VL variant.")
        except Exception:
            pass

    # ── main ─────────────────────────────────────────────────────────────

    def load(self, base_pipeline_path, transformer_path="", text_encoder_path="",
             vae_path="", precision="bf16", attention_backend="auto",
             device="cuda", keep_in_vram=True, offload_vae=False, loader_preset="custom"):
        # Headless / API-run fallback: apply a named preset to the params (the JS
        # normally writes these into the visible panel at edit time).
        if loader_preset and loader_preset != "custom":
            _vals = {
                "base_pipeline_path": base_pipeline_path, "transformer_path": transformer_path,
                "text_encoder_path": text_encoder_path, "vae_path": vae_path,
                "precision": precision, "attention_backend": attention_backend,
                "device": device, "keep_in_vram": keep_in_vram, "offload_vae": offload_vae,
            }
            n = _settings.apply_named_preset("loader", loader_preset, _vals, set(_vals.keys()))
            base_pipeline_path = _vals["base_pipeline_path"]
            transformer_path = _vals["transformer_path"]
            text_encoder_path = _vals["text_encoder_path"]
            vae_path = _vals["vae_path"]
            precision = _vals["precision"]
            attention_backend = _vals["attention_backend"]
            device = _vals["device"]
            keep_in_vram = _vals["keep_in_vram"]
            offload_vae = _vals["offload_vae"]
            print(f"[EricKrea2-Comp] loader_preset '{loader_preset}': applied {n} field(s).")

        try:
            from diffusers import Krea2Pipeline
        except Exception:
            raise RuntimeError(
                "Krea2Pipeline is not available in this diffusers build. Install diffusers with "
                "Krea 2 support (from source).")

        def log(m):
            print("[EricKrea2-Comp] " + m)

        transformer_path = (transformer_path or "").strip()
        text_encoder_path = (text_encoder_path or "").strip()
        vae_path = (vae_path or "").strip()

        loader_settings = _settings.wrap("loader", {
            "base_pipeline_path": base_pipeline_path,
            "transformer_path": transformer_path,
            "text_encoder_path": text_encoder_path,
            "vae_path": vae_path,
            "precision": precision,
            "attention_backend": attention_backend,
            "device": device,
            "keep_in_vram": keep_in_vram,
            "offload_vae": offload_vae,
        })

        cache_key = (f"{_norm_path(base_pipeline_path)}|{_norm_path(transformer_path)}|"
                     f"{_norm_path(text_encoder_path)}|{_norm_path(vae_path)}|"
                     f"{precision}|{device}|{offload_vae}")
        cache = _COMP_CACHE
        if keep_in_vram and cache["pipeline"] is not None and cache["cache_key"] == cache_key:
            log("using cached component pipeline")
            apply_attention_backend(cache["pipeline"].transformer, attention_backend, log=log)
            return ({"pipeline": cache["pipeline"], "is_distilled": cache["is_distilled"],
                     "model_path": base_pipeline_path}, loader_settings)

        # Building a new pipeline: free the previous one FIRST so we never hold two sets of
        # weights in VRAM at once (the transformer-swap OOM), and so keep_in_vram=False releases
        # the old model. Only reached on a cache miss or when keep_in_vram is off.
        _free_current(cache, log)
        # If Stop was pressed, bail out HERE - the old model is already freed, so the
        # user is left with clear VRAM (exactly the "remove weights to switch workflows"
        # case) instead of a half-built reload they can't cancel.
        _raise_if_interrupted()

        dtype = _DTYPES.get(precision, torch.bfloat16)
        is_distilled = _read_is_distilled(base_pipeline_path)
        kwargs = {}
        if transformer_path:
            _t = time.perf_counter()
            kwargs["transformer"] = self._override_transformer(base_pipeline_path, transformer_path, dtype, log, device=device)
            log(f"  [timing] transformer override: {time.perf_counter() - _t:.1f}s")
        if vae_path:
            _t = time.perf_counter()
            kwargs["vae"] = self._override_vae(base_pipeline_path, vae_path, dtype, log)
            log(f"  [timing] vae override: {time.perf_counter() - _t:.1f}s")
        if text_encoder_path:
            _t = time.perf_counter()
            kwargs["text_encoder"] = self._override_text_encoder(base_pipeline_path, text_encoder_path, dtype, log, device=device)
            log(f"  [timing] text_encoder override: {time.perf_counter() - _t:.1f}s")

        log(f"assembling Krea2Pipeline from {base_pipeline_path} "
            f"({'Turbo' if is_distilled else 'Raw'}); overrides={list(kwargs.keys()) or ['none']}")
        _t = time.perf_counter()
        pipe = Krea2Pipeline.from_pretrained(
            base_pipeline_path, torch_dtype=dtype, local_files_only=True, **kwargs)
        log(f"  [timing] Krea2Pipeline.from_pretrained: {time.perf_counter() - _t:.1f}s")
        _raise_if_interrupted()  # last chance to abort before the CPU->VRAM transfer

        if device != "cpu":
            _t = time.perf_counter()
            pipe.to(device)
            log(f"  [timing] pipe.to({device}): {time.perf_counter() - _t:.1f}s")
            if offload_vae:
                log("moving VAE to CPU")
                pipe.vae = pipe.vae.to("cpu")

        apply_attention_backend(pipe.transformer, attention_backend, log=log)

        try:
            params_b = sum(p.numel() for p in pipe.transformer.parameters()) / 1e9
            log(f"assembled - transformer {params_b:.2f}B params")
        except Exception:
            pass

        result = _Krea2PipelineHandle(pipeline=pipe, is_distilled=is_distilled,
                                      model_path=base_pipeline_path)

        if keep_in_vram:
            # Persist for reuse across runs (see the cache-hit check at the top).
            cache["pipeline"] = pipe
            cache["cache_key"] = cache_key
            cache["is_distilled"] = is_distilled
        else:
            # Honor keep_in_vram=False literally: keep NO strong reference of our own.
            # After this run ComfyUI's node-output cache is the SOLE owner, so the
            # dashboard 'Free model and node cache' button (or the Krea2 Unload node)
            # can actually reclaim the VRAM - a module-cached strong ref would silently
            # prevent that (the exact "keep_in_vram off but weights stayed resident"
            # trap). Keep only a weakref so the next build can null ComfyUI's cached
            # copy first and avoid a transient double-load.
            cache["pipeline"] = None
            cache["cache_key"] = None
            cache["is_distilled"] = is_distilled
            builtins._ERIC_KREA2_LAST_HANDLE = weakref.ref(result)
            log("keep_in_vram=False: not caching the pipeline. After this run, free VRAM with "
                "ComfyUI's 'Free model and node cache' button or the Krea2 Unload node.")

        return (result, loader_settings)


NODE_CLASS_MAPPINGS = {"EricKrea2ComponentLoader": EricKrea2ComponentLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2ComponentLoader": "Eric Krea2 Component Loader"}
