# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Unload Models
=========================
Frees VRAM held by a Krea2Pipeline (component-loader or plain-loader output)
without restarting ComfyUI.

WHY this needs to exist (see Session Handoff Sec8/Sec9 #1): the Krea2 pipeline is
a raw HF diffusers object living in module-level caches
(krea2_component_loader._COMP_CACHE / krea2_loader._PIPELINE_CACHE), not a
comfy ModelPatcher registered with comfy.model_management. Stock "unload all
models" nodes only walk comfy's own loaded_models() list, so they see nothing
to free here - hence they no-op on Krea2.

Two references keep the pipeline alive even after our own cache is cleared:
  1. our module-level cache dict (_COMP_CACHE / _PIPELINE_CACHE)
  2. ComfyUI's own node-output cache, which holds the exact dict object the
     loader node returned last run (reused across queues whenever the loader's
     own widget inputs haven't changed - this is *why* repeat queues don't
     reload the model).

This node kills both:
  - mutates the incoming krea2_pipeline dict IN PLACE (sets "pipeline": None)
    so ComfyUI's cached copy - the SAME object - loses its reference too
    (the same trick the community UnloadModelNode uses on comfy MODEL dicts).
  - clears our own module-level cache dict.
  - gc.collect() + torch.cuda.empty_cache() (+ ipc_collect) to actually
    reclaim the VRAM once refcount hits zero.

Pairs with EricKrea2ComponentLoader.IS_CHANGED / EricKrea2Loader.IS_CHANGED
(added alongside this node): those return a fresh value whenever their own
cache is empty (post-unload, or first run), forcing exactly one rebuild on
the next queue - so "unload -> change nothing -> queue again" correctly
reloads instead of handing a stale pipeline=None dict downstream.

Wire it as a passthrough at the end of your graph: `value` = whatever you
want to keep flowing (e.g. the final IMAGE, or just the krea2_pipeline
itself), `krea2_pipeline` = the pipeline to free. Safe to run even if nothing
is loaded (no-op + prints "nothing to free").
"""

import gc

import torch


class _AnyType(str):
    """Matches any ComfyUI type for the passthrough slot (mirrors the common '*' pattern)."""
    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class EricKrea2UnloadModels:
    CATEGORY = "Eric/Krea2"
    FUNCTION = "unload"
    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("value",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (_ANY, {
                    "tooltip": "Passthrough - wire anything here (e.g. the final IMAGE) so this "
                               "node sits downstream in the graph and runs as part of the queue."}),
            },
            "optional": {
                "krea2_pipeline": ("KREA2_PIPELINE", {
                    "tooltip": "The pipeline to free. Connect the Component Loader or plain "
                               "Loader's krea2_pipeline output here."}),
                "also_clear_module_caches": ("BOOLEAN", {"default": True,
                    "tooltip": "Also clear the loader nodes' module-level caches "
                               "(_COMP_CACHE / _PIPELINE_CACHE), not just this input. Leave ON "
                               "unless you specifically want to free only the connected pipeline "
                               "and keep another cached one alive."}),
                "aggressive_teardown": ("BOOLEAN", {"default": True,
                    "tooltip": "Explicitly break the pipe's internal references to its submodules "
                               "(transformer/vae/text_encoder) before dropping the pipe itself, "
                               "instead of relying on the top-level pipe object's refcount hitting "
                               "zero on its own. Use this when 'keep in vram'=True on the loader and "
                               "VRAM isn't dropping after unload - something (accelerate hooks, "
                               "cached attention workspace, an offload dispatch table) may be holding "
                               "a submodule reference independent of the pipe wrapper."}),
                "also_unload_comfy_models": ("BOOLEAN", {"default": True,
                    "tooltip": "Also unload models ComfyUI's own model_management is tracking "
                               "(SAM / GroundingDINO / DepthPro and anything else loaded through "
                               "regular comfy nodes), then soft-empty comfy's cache. NOTE: nodes "
                               "that cache in their own globals (e.g. SegmentAnything Ultra with "
                               "cache_model=True) are invisible to this - turn their own cache "
                               "toggle off as well."}),
                "verbose": ("BOOLEAN", {"default": True,
                    "tooltip": "Print refcount + CUDA memory before/after, to diagnose cases where "
                               "VRAM still doesn't drop (tells us whether something else is holding "
                               "a reference, vs. it being driver/kernel-level memory that only frees "
                               "on process exit)."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @staticmethod
    def _teardown_pipe(pipe, log):
        """Explicitly drop references to the pipe's heavy submodules before dropping
        the pipe itself. Covers the case where something OTHER than the top-level
        pipe object (an accelerate hook, an offload dispatch table, a cached
        attention-backend workspace tied to the module instance) is keeping a
        transformer/vae/text_encoder alive independent of the pipe's own refcount."""
        for attr in ("transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer"):
            try:
                comp = getattr(pipe, attr, None)
                if comp is not None:
                    try:
                        comp.to("meta")  # drop parameter storage without needing a real device
                    except Exception:
                        pass
                    setattr(pipe, attr, None)
                    log(f"  torn down pipe.{attr}")
            except Exception as e:
                log(f"  could not tear down pipe.{attr}: {e}")
        # Any hook registry accelerate may have attached (dispatch_model / cpu offload).
        for attr in ("_all_hooks", "hf_device_map"):
            if hasattr(pipe, attr):
                try:
                    setattr(pipe, attr, None)
                except Exception:
                    pass

    def unload(self, value=None, krea2_pipeline=None, also_clear_module_caches=True,
               aggressive_teardown=True, also_unload_comfy_models=True, verbose=True):
        import sys
        freed_something = False

        def log(m):
            if verbose:
                print(f"[EricKrea2-Unload] {m}")

        if verbose and torch.cuda.is_available():
            a0 = torch.cuda.memory_allocated() / 1e9
            r0 = torch.cuda.memory_reserved() / 1e9
            log(f"before: allocated={a0:.2f} GB, reserved={r0:.2f} GB")

        # 1) Mutate the incoming dict in place - this is the SAME object ComfyUI's
        #    node-output cache is holding, so clearing the key here removes that
        #    reference too, not just our local kwarg copy.
        if isinstance(krea2_pipeline, dict) and krea2_pipeline.get("pipeline") is not None:
            pipe = krea2_pipeline["pipeline"]
            if verbose:
                # 3 baseline refs are expected here: this local var, the frame's fast-locals
                # copy getrefcount takes as its own arg, and the dict entry we're about to
                # clear. Anything meaningfully higher means something ELSE also holds it.
                log(f"connected pipeline refcount (incl. this check + the dict entry): "
                    f"{sys.getrefcount(pipe)}")
            log(f"releasing connected pipeline ({krea2_pipeline.get('model_path', '?')})")
            if aggressive_teardown:
                self._teardown_pipe(pipe, log)
            krea2_pipeline["pipeline"] = None
            del pipe
            freed_something = True

        # 2) Clear our own module-level caches so the loader's cache-hit check
        #    (`cache["pipeline"] is not None`) fails next run and it rebuilds.
        if also_clear_module_caches:
            try:
                from .krea2_component_loader import _COMP_CACHE
                if _COMP_CACHE.get("pipeline") is not None:
                    if aggressive_teardown and _COMP_CACHE["pipeline"] is not krea2_pipeline:
                        self._teardown_pipe(_COMP_CACHE["pipeline"], log)
                    _COMP_CACHE["pipeline"] = None
                    _COMP_CACHE["cache_key"] = None
                    freed_something = True
            except Exception as e:
                log(f"could not reach component-loader cache: {e}")
            try:
                from .krea2_loader import _PIPELINE_CACHE
                if _PIPELINE_CACHE.get("pipeline") is not None:
                    if aggressive_teardown:
                        self._teardown_pipe(_PIPELINE_CACHE["pipeline"], log)
                    _PIPELINE_CACHE["pipeline"] = None
                    _PIPELINE_CACHE["cache_key"] = None
                    freed_something = True
            except Exception as e:
                log(f"could not reach plain-loader cache: {e}")

        # 3) Sweep ComfyUI-managed models too (the Krea2 pipe is invisible to comfy's
        #    model_management, but the reverse is also true: SAM/DINO/DepthPro etc.
        #    loaded by regular comfy nodes are invisible to steps 1-2 - the exact
        #    residual VRAM a sharpening/segmentation subgraph leaves behind).
        if also_unload_comfy_models:
            try:
                import comfy.model_management as mm
                n_before = len(getattr(mm, "current_loaded_models", []) or [])
                mm.unload_all_models()
                try:
                    mm.soft_empty_cache()
                except TypeError:
                    mm.soft_empty_cache(True)
                if n_before:
                    log(f"unloaded {n_before} ComfyUI-managed model(s) "
                        f"(comfy.model_management)")
                    freed_something = True
                else:
                    log("no ComfyUI-managed models were loaded")
            except Exception as e:
                log(f"comfy model sweep failed: {e}")

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as e:
            log(f"cache empty failed: {e}")

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            msg = f"{'freed pipeline reference(s)' if freed_something else 'nothing to free'}; VRAM free: {free_b/1e9:.2f} / {total_b/1e9:.2f} GB"
            if verbose:
                a1 = torch.cuda.memory_allocated() / 1e9
                r1 = torch.cuda.memory_reserved() / 1e9
                msg += f" | torch allocated={a1:.2f} GB, reserved={r1:.2f} GB"
                if freed_something and r1 > 1.0:
                    msg += ("\n[EricKrea2-Unload]   NOTE: reserved memory is still high after a real "
                            "free. If this doesn't drop further, it's likely driver/kernel-level "
                            "memory (e.g. FlashAttention workspace buffers) outside PyTorch's caching "
                            "allocator, which torch.cuda.empty_cache() can't touch - that only "
                            "releases on process exit, same as the restart workaround.")
            print(f"[EricKrea2-Unload] {msg}")
        else:
            log(f"{'freed pipeline reference(s)' if freed_something else 'nothing to free'}")

        return (value,)


NODE_CLASS_MAPPINGS = {"EricKrea2UnloadModels": EricKrea2UnloadModels}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2UnloadModels": "Eric Krea2 Unload Models"}
