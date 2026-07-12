# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Decode nodes
======================
Consume a KREA2_LATENT (packed latents + pixel dims) produced by the Generate
or Multi-Stage node and turn it into an image.

  EricKrea2Decode          -> 2x upscale decode via the spacepxl Wan upscale VAE
                              (Krea 2 shares the Qwen-Image latent space, so this
                              is a free 2x super-resolution at decode time).
  EricKrea2VAEDecode       -> plain 1x decode. Optionally accepts an alternate
                              decode_vae (e.g. base Wan 2.1) to change the
                              reconstruction grain/texture.
  EricKrea2DecodeVAELoader -> load an alternate 1x decode VAE (base Wan 2.1
                              family) for grain/texture A/B vs the Qwen VAE.
"""

from __future__ import annotations

from .._upscale_vae import decode_latents_with_upscale_vae
from .._latent_utils import standard_decode


class EricKrea2DecodeVAELoader:
    """Load an alternate 1x decode VAE (base Wan 2.1 family) for grain/texture A/B.

    Krea 2's Qwen-Image VAE and the Wan 2.1 VAE share an *identical* latent space
    (same latents_mean/std), so a base Wan VAE can decode Krea2 latents directly.
    The community uses this to trade the Qwen "plastic skin" look for the more
    natural grain of the Wan decoder - no upscale, just a different reconstruction.

    Accepts any of:
      - a HF repo id (default: the base Wan 2.1 diffusers VAE),
      - a local diffusers folder (containing config.json), or
      - a local single-file .safetensors (native ComfyUI/Wan format), converted
        on load via diffusers' from_single_file + convert_wan_vae_to_diffusers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_source": ("STRING", {
                    "default": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    "tooltip": "HF repo id (uses its 'vae' subfolder), a local diffusers "
                               "folder, or a local .safetensors file (e.g. a downloaded "
                               "wan_2.1_vae.safetensors)."}),
            },
            "optional": {
                "dtype": (["float32", "bfloat16", "float16"], {
                    "default": "float32",
                    "tooltip": "float32 is safest for a clean decode; bf16/fp16 are lighter."}),
            },
        }

    RETURN_TYPES = ("KREA2_DECODE_VAE",)
    RETURN_NAMES = ("decode_vae",)
    FUNCTION = "load"
    CATEGORY = "Eric/Krea2"

    def load(self, vae_source, dtype="float32"):
        import os
        import torch
        from diffusers import AutoencoderKLWan
        import comfy.model_management as mm

        td = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}.get(dtype, torch.float32)
        src = (vae_source or "").strip()
        device = mm.get_torch_device()

        # Wan 2.1 VAE config bundled with this package, so single-file loads never
        # need to reach HuggingFace for a config.json (the diffusers default points
        # at a repo whose config lives in a subfolder, which fails offline/in ComfyUI).
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wan_cfg_dir = os.path.join(pkg_root, "configs", "wan21_vae")
        have_bundled = os.path.isfile(os.path.join(wan_cfg_dir, "config.json"))

        def _finish(vae, how):
            vae = vae.to(device=device, dtype=td).eval()
            zc = getattr(vae.config, "z_dim", "?")
            print(f"[EricKrea2] Decode VAE loaded via {how}: {type(vae).__name__} "
                  f"(z_dim={zc}) -> {device}/{dtype}")
            return (vae,)

        if not src:
            raise ValueError("[EricKrea2] vae_source is empty.")

        # 0) stale/moved local path -> re-resolve by filename through ComfyUI's
        #    registered vae folders (survives model-library reorganizations).
        looks_like_path = ("\\" in src or "/" in src
                           or src.lower().endswith(".safetensors"))
        if looks_like_path and not os.path.exists(src):
            try:
                import folder_paths
                base = os.path.basename(src.replace("/", "\\").rstrip("\\"))
                cand = None
                for rel in folder_paths.get_filename_list("vae"):
                    if os.path.basename(rel) == base:
                        cand = folder_paths.get_full_path("vae", rel)
                        break
                if cand and os.path.isfile(cand):
                    print(f"[EricKrea2] vae_source '{src}' not found; resolved "
                          f"'{base}' via ComfyUI model paths -> {cand}")
                    src = cand
                else:
                    raise FileNotFoundError(
                        f"[EricKrea2] vae_source '{src}' does not exist and no file "
                        f"named '{base}' was found in any registered vae folder. "
                        f"Update the node's vae_source (models now live in A:\\Models).")
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"[EricKrea2] path re-resolution failed: {e}")

        # 1) local single-file safetensors -> from_single_file (handles native Wan keys).
        #    Use the bundled local config first (offline, deterministic).
        if src.lower().endswith(".safetensors") and os.path.isfile(src):
            last_err = None
            if have_bundled:
                try:
                    return _finish(
                        AutoencoderKLWan.from_single_file(src, config=wan_cfg_dir, torch_dtype=td),
                        "from_single_file(bundled config)")
                except Exception as e:
                    last_err = e
                    print(f"[EricKrea2] single_file with bundled config failed: {e}; "
                          "trying diffusers default ...")
            try:
                return _finish(AutoencoderKLWan.from_single_file(src, torch_dtype=td),
                               "from_single_file(default)")
            except Exception as e:
                raise RuntimeError(
                    f"[EricKrea2] Could not load single-file VAE '{src}'. "
                    f"Last error: {e}. Workaround: set vae_source to a diffusers VAE "
                    f"*folder* that contains config.json (e.g. a model's '...\\vae' folder)."
                ) from (last_err or e)

        # 2) local diffusers folder (has config.json)
        if os.path.isdir(src) and os.path.isfile(os.path.join(src, "config.json")):
            return _finish(AutoencoderKLWan.from_pretrained(src, torch_dtype=td),
                           "from_pretrained(folder)")

        # 3) HF repo id -> try the bundled config locally if it's the known Wan repo,
        #    else the 'vae' subfolder, then the repo root.
        if looks_like_path:
            raise FileNotFoundError(
                f"[EricKrea2] vae_source '{src}' looks like a local path but does not "
                f"exist; refusing to treat it as a HuggingFace repo id.")
        try:
            return _finish(
                AutoencoderKLWan.from_pretrained(src, subfolder="vae", torch_dtype=td),
                "from_pretrained(repo:vae)")
        except Exception:
            return _finish(AutoencoderKLWan.from_pretrained(src, torch_dtype=td),
                           "from_pretrained(repo)")


class EricKrea2Decode:
    """2x upscale-decode of Krea2 latents (no re-denoise; ideal for Turbo)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipeline": ("KREA2_PIPELINE",),
                "latent": ("KREA2_LATENT",),
                "upscale_vae": ("UPSCALE_VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "Eric/Krea2"

    def decode(self, krea2_pipeline, latent, upscale_vae):
        pipe = krea2_pipeline["pipeline"]
        vsf = latent.get("vae_scale_factor", 8)
        image = decode_latents_with_upscale_vae(
            latent["packed"], upscale_vae, pipe.vae,
            latent["height"], latent["width"], vae_scale_factor=vsf,
        )
        print(f"[EricKrea2] Upscale-decoded to {image.shape[1]}x{image.shape[2]} px")
        return (image,)


class EricKrea2VAEDecode:
    """Plain 1x decode of Krea2 latents. Optionally use an alternate decode VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipeline": ("KREA2_PIPELINE",),
                "latent": ("KREA2_LATENT",),
            },
            "optional": {
                "decode_vae": ("KREA2_DECODE_VAE", {
                    "tooltip": "Optional base Wan 2.1 (or other Wan-family) VAE to decode "
                               "with instead of the pipeline's Qwen VAE - changes grain/"
                               "texture without an upscale."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "Eric/Krea2"

    def decode(self, krea2_pipeline, latent, decode_vae=None):
        pipe = krea2_pipeline["pipeline"]
        image = standard_decode(pipe, latent["packed"], latent["height"], latent["width"],
                                decode_vae=decode_vae)
        tag = "alt VAE" if decode_vae is not None else "pipeline VAE"
        print(f"[EricKrea2] 1x decode ({tag}) -> {image.shape[2]}x{image.shape[1]} px")
        return (image,)


class EricKrea2LatentToComfy:
    """Adapter: KREA2_LATENT (packed) -> ComfyUI LATENT (unpacked), so a Generate /
    Multi-Stage latent can be wired into ComfyUI's *native* VAEDecode or any other
    LATENT consumer (our KREA2_LATENT is a custom packed type that won't wire).

    The packed flow latent is unpacked to a spatial [B, 16, 1, H_lat, W_lat] tensor
    and returned as {"samples": ...}. This is the NORMALISED z-latent, exactly what
    ComfyUI keeps in LATENT['samples']; ComfyUI's VAE applies its own process_out
    de-normalisation at decode. Krea2 / Qwen-Image and Wan 2.1 share identical
    latents_mean/std, so pair this with a ComfyUI-loaded **Wan 2.1** (or Qwen-Image)
    VAE + the stock VAEDecode node.

    Primary purpose: A/B our diffusers decode path (diffusers tiling + cosine blend,
    only tiles >128px) against ComfyUI's native tiled decode of the *same* latent,
    to isolate decode-side artifacts (bottom band / grid) from generation-side ones.
    If the band/grid vanishes through native VAEDecode, it's our decode path; if it
    survives, it's baked into the latent (generation).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("KREA2_LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "convert"
    CATEGORY = "Eric/Krea2"

    def convert(self, latent):
        from .._latent_utils import _unpack_latents
        vsf = latent.get("vae_scale_factor", 8)
        spatial = _unpack_latents(latent["packed"], latent["height"], latent["width"], vsf)
        spatial = spatial.contiguous().float().cpu()   # [B, 16, 1, H_lat, W_lat], normalised
        print(f"[EricKrea2] KREA2_LATENT -> ComfyUI LATENT: samples {tuple(spatial.shape)} "
              f"(normalised; pair with a ComfyUI Wan2.1/Qwen VAE + VAEDecode)")
        return ({"samples": spatial},)


NODE_CLASS_MAPPINGS = {
    "EricKrea2DecodeVAELoader": EricKrea2DecodeVAELoader,
    "EricKrea2Decode": EricKrea2Decode,
    "EricKrea2VAEDecode": EricKrea2VAEDecode,
    "EricKrea2LatentToComfy": EricKrea2LatentToComfy,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EricKrea2DecodeVAELoader": "Eric Krea2 Decode VAE Loader",
    "EricKrea2Decode": "Eric Krea2 Upscale Decode (2x)",
    "EricKrea2VAEDecode": "Eric Krea2 VAE Decode",
    "EricKrea2LatentToComfy": "Eric Krea2 Latent -> ComfyUI LATENT",
}
