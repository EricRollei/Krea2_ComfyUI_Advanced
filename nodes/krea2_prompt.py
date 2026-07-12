# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Prompt nodes
======================
Krea 2 rewards dense, natural-language prompts (not JSON/tags like Ideogram).

  EricKrea2Prompt       -> assemble structured fields into a dense, comma-joined
                           natural-language prompt; quotes text-to-render.
  EricKrea2MagicPrompt  -> expand a short prompt via a local LLM (LM Studio /
                           Ollama / any OpenAI-compatible endpoint), the
                           natural-language analogue of Ideogram Magic Prompt.

For best fidelity to Krea's tuned expander, download Krea's official
``expansion.txt`` from https://github.com/krea-ai/krea-2 and point
``system_prompt_path`` at it. A faithful built-in default (based on Krea's public
prompting guidance) is used otherwise.
"""

from __future__ import annotations

# Default expander system prompt - authored from Krea's public prompting guidance
# (natural language, dense visual/material/camera detail, preserve intent, quote
# text). Not a copy of Krea's expansion.txt.
_DEFAULT_EXPANDER_SYSTEM = (
    "You are a prompt expander for the Krea 2 image model. Given a short user "
    "prompt, rewrite it into a single long, dense, natural-language image "
    "description. Preserve the user's subject and intent exactly and never "
    "introduce unrelated subjects. Add concrete visual detail: medium and style, "
    "materials and textures, lighting, color palette, camera and lens, angle and "
    "distance, composition, and mood. Prefer specific, art-directed or "
    "photographic language. If the user specifies words that must appear as text "
    "in the image, keep them inside double quotes. Output only the expanded "
    "prompt as a single paragraph, with no preamble, labels, or surrounding quotes."
)


def _clean_llm_output(text: str) -> str:
    """Strip common LLM wrappers (think blocks, preamble lines, surrounding quotes)."""
    import re
    t = (text or "").strip()
    # reasoning models: drop <think>...</think> / <thinking>... blocks entirely,
    # including an unclosed leading block (some models stream the tag unbalanced)
    t = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", t).strip()
    t = re.sub(r"(?is)^<think(?:ing)?>.*", "", t).strip() or t
    # drop a leading "Here is ...:" style line
    if "\n" in t:
        first, rest = t.split("\n", 1)
        low = first.lower()
        if low.startswith(("here is", "here's", "expanded prompt", "sure,", "certainly")):
            t = rest.strip()
    if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
        t = t[1:-1].strip()
    return t


def _probe_openai(endpoint, timeout=2.5):
    """True if an OpenAI-compatible server answers at endpoint/models."""
    import requests
    try:
        return requests.get(endpoint.rstrip("/") + "/models", timeout=timeout).ok
    except Exception:
        return False


def _probe_ollama(base, timeout=2.5):
    import requests
    try:
        return requests.get(base.rstrip("/") + "/api/tags", timeout=timeout).ok
    except Exception:
        return False


def _find_local_qwen3vl(model_field=""):
    """Locate a local Qwen3-VL folder. Returns (dir_or_None, importable_bool).

    Uses video_prompter's qwen3_vl backend if that package is installed
    (soft dependency - public Eric_Krea2 users without it fall through to
    the endpoint tiers)."""
    try:
        from video_prompter.qwen3_vl_backend import generate_text_with_qwen3_vl  # noqa: F401
    except Exception:
        return None, False
    import os
    if model_field and os.path.isdir(model_field.strip()):
        return model_field.strip(), True
    try:
        import folder_paths
        from pathlib import Path
        base = Path(folder_paths.models_dir) / "VLM"
        preferred = base / "Qwen3-VL-4B-Instruct"
        if preferred.is_dir():
            return str(preferred), True
        hits = sorted(base.glob("Qwen*-VL-*")) if base.exists() else []
        for h in hits:
            if h.is_dir():
                return str(h), True
    except Exception:
        pass
    return None, True


def _expand_qwen3vl(prompt, system_prompt, model_dir, temperature, max_tokens, device="auto"):
    from video_prompter.qwen3_vl_backend import generate_text_with_qwen3_vl
    hint = None if device in ("", "auto") else f"device={device}"
    result = generate_text_with_qwen3_vl(
        prompt=f"{system_prompt}\n\n{prompt}",
        model_spec=f"local:{model_dir}",
        backend_hint=hint,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
    )
    if not result.get("success"):
        raise RuntimeError(result.get("error", "Qwen3-VL generation failed"))
    return _clean_llm_output(result.get("response", ""))


def _expand_ollama(prompt, system_prompt, base, model, temperature, max_tokens):
    import requests
    base = base.rstrip("/").removesuffix("/v1")
    if not (model or "").strip():
        try:
            tags = requests.get(f"{base}/api/tags", timeout=5).json().get("models", [])
            model = tags[0]["name"] if tags else "default"
            print(f"[EricKrea2] Auto-detected Ollama model: {model}")
        except Exception:
            model = "default"
    r = requests.post(f"{base}/api/generate", json={
        "model": model,
        "prompt": f"{system_prompt}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
    }, timeout=120)
    r.raise_for_status()
    return _clean_llm_output(r.json().get("response", ""))


def _llm_expand(prompt, system_prompt, endpoint, model, temperature, max_tokens):
    """OpenAI-compatible chat completion (LM Studio or any /v1 endpoint)."""
    import requests
    endpoint = endpoint.rstrip("/")
    if not (model or "").strip():
        try:
            r = requests.get(f"{endpoint}/models", timeout=5)
            data = r.json().get("data", [])
            model = (data[0].get("id") or data[0].get("model")) if data else "default"
            print(f"[EricKrea2] Auto-detected LLM model: {model}")
        except Exception:
            model = "default"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    r = requests.post(f"{endpoint}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _clean_llm_output(content)


class EricKrea2Prompt:
    """Assemble structured fields into a dense natural-language Krea 2 prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        S = lambda d="": ("STRING", {"multiline": True, "default": d})
        return {
            "required": {"subject": ("STRING", {"multiline": True, "default": ""})},
            "optional": {
                "scene": S(), "style_medium": S(), "lighting": S(),
                "color": S(), "camera": S(), "composition": S(),
                "mood": S(), "extra_details": S(),
                "text_to_render": ("STRING", {"default": "",
                    "tooltip": "Words to render as text in the image (auto-wrapped in quotes)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"

    def build(self, subject, scene="", style_medium="", lighting="", color="",
              camera="", composition="", mood="", extra_details="", text_to_render=""):
        parts = [subject, style_medium, scene, composition, camera, lighting,
                 color, mood, extra_details]
        prompt = ", ".join(p.strip() for p in parts if p and p.strip())
        if text_to_render and text_to_render.strip():
            prompt += f', with the text "{text_to_render.strip()}"'
        return (prompt,)


class EricKrea2MagicPrompt:
    """Expand a short prompt into a dense Krea 2 prompt via a local LLM.

    Backend tiers (same architecture as video_prompter):
      qwen3_vl  -> local Qwen3-VL from the models folder (via video_prompter's
                   backend when installed; no server needed)
      lm_studio -> OpenAI-compatible server (default :1234/v1)
      ollama    -> Ollama server (:11434)
      endpoint  -> any OpenAI-compatible /v1 URL as given
      auto      -> first available of: local qwen3_vl -> lm_studio -> ollama.
                   A missing local model is skipped, never auto-downloaded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "backend": (["auto", "qwen3_vl", "lm_studio", "ollama", "endpoint"], {
                    "default": "auto",
                    "tooltip": "auto: local Qwen3-VL -> LM Studio -> Ollama, first available."}),
                "endpoint": ("STRING", {"default": "http://localhost:1234/v1",
                    "tooltip": "lm_studio/endpoint: OpenAI-compatible base URL. "
                               "ollama: base URL (:11434, /v1 stripped automatically)."}),
                "model": ("STRING", {"default": "",
                    "tooltip": "Server model id (blank = auto-detect), or for qwen3_vl a "
                               "local model folder path (blank = auto-detect in models/VLM)."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "system_prompt": ("STRING", {"multiline": True, "default": _DEFAULT_EXPANDER_SYSTEM}),
                "system_prompt_path": ("STRING", {"default": "",
                    "tooltip": "Optional path to Krea's expansion.txt; overrides system_prompt if set."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "Off = pass the prompt through unchanged."}),
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {"default": "auto",
                    "tooltip": "qwen3_vl local tier only: device for the weights. "
                               "auto = accelerate device_map (usually cuda:0). "
                               "cuda:1 keeps the 6000 free for the Krea2 pipeline."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "status")
    FUNCTION = "expand"
    CATEGORY = "Eric/Krea2"

    def _pick_auto(self, endpoint, model):
        """Return (tier_name, detail) for the first available backend."""
        local_dir, importable = _find_local_qwen3vl(model)
        if importable and local_dir:
            return "qwen3_vl", local_dir
        lm = endpoint if "1234" in endpoint else "http://localhost:1234/v1"
        if _probe_openai(lm):
            return "lm_studio", lm
        if _probe_ollama("http://localhost:11434"):
            return "ollama", "http://localhost:11434"
        return None, ("no local Qwen3-VL model" if importable else
                      "video_prompter backend not installed") + \
            ", LM Studio (:1234) and Ollama (:11434) unreachable"

    def expand(self, prompt, backend="auto", endpoint="http://localhost:1234/v1",
               model="", temperature=0.7, max_tokens=512,
               system_prompt=_DEFAULT_EXPANDER_SYSTEM, system_prompt_path="",
               enabled=True, device="auto"):
        if not enabled or not prompt.strip():
            return (prompt, "passthrough (disabled or empty prompt)")

        sys_prompt = system_prompt
        if system_prompt_path and system_prompt_path.strip():
            try:
                with open(system_prompt_path.strip(), "r", encoding="utf-8") as f:
                    sys_prompt = f.read().strip()
                print(f"[EricKrea2] Loaded expander system prompt from {system_prompt_path}")
            except Exception as e:
                print(f"[EricKrea2] Could not read system_prompt_path ({e}); using default.")

        tier, detail = backend, ""
        if backend == "auto":
            tier, detail = self._pick_auto(endpoint, model)
            if tier is None:
                msg = f"\u274c no backend available ({detail}) - prompt passed through UNEXPANDED"
                print(f"[EricKrea2] Magic Prompt: {msg}")
                return (prompt, msg)

        try:
            if tier == "qwen3_vl":
                if (model or "").strip().lower().endswith(".gguf"):
                    msg = ("\u274c GGUF files can't load in the local transformers tier; "
                           "LM Studio serves them - set backend=lm_studio and put the model "
                           "id (e.g. 'huihui-qwen3-vl-8b-instruct-abliterated', 'joycaption@f16') "
                           "in the model field")
                    print(f"[EricKrea2] Magic Prompt: {msg}")
                    return (prompt, msg)
                local_dir = detail or _find_local_qwen3vl(model)[0]
                if not local_dir:
                    raise FileNotFoundError(
                        "no local Qwen3-VL model folder found (models/VLM) and none given in 'model'")
                expanded = _expand_qwen3vl(prompt, sys_prompt, local_dir, temperature,
                                           max_tokens, device)
                via = f"qwen3_vl local ({local_dir}, device={device})"
            elif tier == "ollama":
                base = detail or endpoint
                expanded = _expand_ollama(prompt, sys_prompt, base, model, temperature, max_tokens)
                via = "ollama"
            else:  # lm_studio or generic endpoint
                ep = detail or endpoint
                expanded = _llm_expand(prompt, sys_prompt, ep, model, temperature, max_tokens)
                via = f"{tier} ({ep})"

            if not expanded:
                msg = f"\u274c empty expansion from {via} - prompt passed through UNEXPANDED"
                print(f"[EricKrea2] Magic Prompt: {msg}")
                return (prompt, msg)
            msg = f"\u2705 expanded {len(prompt)} -> {len(expanded)} chars via {via}"
            print(f"[EricKrea2] Magic Prompt: {msg}")
            return (expanded, msg)
        except Exception as e:
            msg = f"\u274c {tier} failed: {e} - prompt passed through UNEXPANDED"
            print(f"[EricKrea2] Magic Prompt: {msg}")
            return (prompt, msg)


NODE_CLASS_MAPPINGS = {
    "EricKrea2Prompt": EricKrea2Prompt,
    "EricKrea2MagicPrompt": EricKrea2MagicPrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EricKrea2Prompt": "Eric Krea2 Prompt",
    "EricKrea2MagicPrompt": "Eric Krea2 Magic Prompt",
}
