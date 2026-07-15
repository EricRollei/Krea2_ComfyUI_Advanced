# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Eric_Krea2 - ComfyUI custom nodes for the Krea 2 (Raw/Turbo) image model.

"""
Eric_Krea2
==========
Self-contained ComfyUI nodes for Krea 2 (single-stream MMDiT, flow matching,
Qwen3-VL conditioning, AutoencoderKLQwenImage VAE). No dependency on any other
node pack.

Requires a diffusers build that includes Krea2Pipeline:
    python_embeded\\python.exe -m pip install --upgrade --force-reinstall --no-deps git+https://github.com/huggingface/diffusers.git
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MODULES = []
try:
    from .nodes import (krea2_loader, krea2_generate, krea2_decode, krea2_encode,
                        krea2_multistage_ultra, krea2_multistage_ultra_v2, krea2_lora,
                        krea2_lora_stack, krea2_component_loader, krea2_settings,
                        krea2_settings_from_image, krea2_sigmas, krea2_resolution, krea2_latent_resize,
                        krea2_prompt, krea2_vision_prompt, krea2_ref_latents, krea2_debug,
                        krea2_unload)
    from . import _upscale_vae
    _MODULES += [krea2_loader, krea2_generate, krea2_decode, krea2_encode,
                 krea2_multistage_ultra, krea2_multistage_ultra_v2, krea2_lora,
                 krea2_lora_stack, krea2_component_loader, krea2_settings,
                 krea2_settings_from_image, krea2_sigmas, krea2_resolution, krea2_latent_resize,
                 krea2_prompt, krea2_vision_prompt, krea2_ref_latents, krea2_debug,
                 krea2_unload, _upscale_vae]
except Exception as e:  # pragma: no cover - surfaced in ComfyUI console
    import traceback
    print(f"[Eric_Krea2] Failed to import nodes: {e}")
    traceback.print_exc()

for _m in _MODULES:
    NODE_CLASS_MAPPINGS.update(getattr(_m, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_m, "NODE_DISPLAY_NAME_MAPPINGS", {}))

# Save-preset backend endpoint for the ★ Save Preset button (no-op if the
# ComfyUI server isn't importable, e.g. standalone import).
try:
    from . import server as _krea2_server
    _krea2_server.register_routes()
except Exception as e:  # pragma: no cover
    print(f"[Eric_Krea2] save-preset endpoint not registered: {e}")

# Serve web/ (JS extensions) to the ComfyUI frontend.
WEB_DIRECTORY = "./web"

print(f"[Eric_Krea2] Loaded {len(NODE_CLASS_MAPPINGS)} node(s): "
      f"{', '.join(NODE_CLASS_MAPPINGS.keys())}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
