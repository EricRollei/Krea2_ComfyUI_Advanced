# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Vision Prompt
========================
Push 1-3 reference images through Krea2's own text encoder (Qwen3-VL) vision
path, so the resulting conditioning is visually grounded by those images - a
"prompt from a picture" effect - using Krea2's ACTUAL trained descriptor
template rather than the generic Qwen-Image-Edit template.

Background (see chat history for the full trace): the community pattern of
wiring the core `TextEncodeQwenImageEditPlus` node into Krea2 has two problems
- Krea2's DiT has no reference-latent pathway (the VAE-encoded reference is
silently discarded), and the generic node uses the wrong prompt template with
images attached. This node fixes both: no VAE step at all (nothing for Krea2 to
discard), and Krea2's own `prompt_template_encode_prefix/suffix` (read straight
off the loaded pipeline, so it can't drift out of sync).

Mechanics, assembled from two verified real sources (not guessed):
  1. Image placeholder expansion - the "how many <|image_pad|> tokens does this
     picture need" logic - is `transformers.Qwen3VLProcessor`'s own algorithm
     (`image_grid_thw[i].prod() // merge_size**2` repeats), reused verbatim.
  2. Krea2's fixed-length tokenization - pads in the MIDDLE of the template
     ([prefix | prompt | PAD] then the assistant suffix appended AFTER the
     padding) - is `Krea2Pipeline.get_text_hidden_states`'s own scheme, reused
     verbatim so a vision prompt tokenizes exactly like a text-only one apart
     from the extra image tokens.
  3. Position IDs for the mixed text+image sequence use
     `Qwen3VLModel.get_rope_index()` - the model's own correct multimodal RoPE,
     NOT Krea2Pipeline's simplified text-only scheme (which is provably just
     the text-only special case of get_rope_index, so this is a strict
     superset, not a competing approach).

Output KREA2_CONDITIONING is a plain dict {"embeds", "mask"} - the same shape
`encode_prompt` returns internally. Wire it into the Multi-Stage Ultra node's
optional `prompt_conditioning` input; Ultra passes it straight to
`pipe(prompt_embeds=..., prompt_embeds_mask=...)` at every stage instead of
re-deriving from a prompt string, and cond_rebalance applies to it exactly as
it would to a text-only encoding (same 4D (B, seq, 12, dim) tensor shape).

Positive-conditioning only by design (see chat history): steering toward a
reference makes sense; steering away from one via a reference image is a much
stranger ask, and Turbo (the checkpoint this is aimed at) mostly runs cfg=0
anyway, so there is no negative pass to feed.

UNTESTED ON A LIVE MODEL as of writing - built and cross-checked against the
installed diffusers/transformers source (verified: template text, tokenization
scheme, image_grid_thw mechanics, get_rope_index reduction to the text-only
case) and compiles clean, but I have no GPU access to run it. Use print_prompt
+ debug_shapes on the first test - if something's off, that output is the
fastest way to see where.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class EricKrea2VisionPrompt:
    CATEGORY = "Eric/Krea2"
    FUNCTION = "encode"
    RETURN_TYPES = ("KREA2_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipeline": ("KREA2_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Your instruction/description. Combined with the reference image(s) inside "
                               "Krea2's own trained descriptor template - not a generic image-caption template."}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Reference image 1. Fed through the vision path only - no VAE, "
                                                "nothing for Krea2's DiT to silently discard."}),
                "image2": ("IMAGE", {"tooltip": "Reference image 2 (optional)."}),
                "image3": ("IMAGE", {"tooltip": "Reference image 3 (optional)."}),
                "vision_position": (["before prompt", "after prompt"], {"default": "before prompt",
                    "tooltip": "Where the 'Picture N: <image>' placeholders sit relative to your prompt text "
                               "inside the user turn. Matches the community node's convention."}),
                "vision_megapixels": ("FLOAT", {"default": 0.40, "min": 0.05, "max": 4.0, "step": 0.05,
                    "tooltip": "PER-IMAGE ceiling in megapixels, before auto-sharing (see below). Krea2's text "
                               "budget is fixed at 512 tokens total (prefix + all images + your prompt) - it "
                               "doesn't grow with more images, so this node automatically divides ~0.40 MP of "
                               "shared budget across however many images are connected (1 image ~0.40 MP, 2 "
                               "images ~0.20 MP each, 3 images ~0.13 MP each) and takes the smaller of that "
                               "share and this ceiling. Lower this to reserve more room for a long prompt; it "
                               "never raises the auto-share, only caps it further."}),
                "vision_processor_source": ("STRING", {
                    "default": r"H:\Testing\Qwen3-VL-4B-Instruct-heretic-7refusal",
                    "tooltip": "Folder with a preprocessor_config.json for the Qwen3-VL image processor. Krea2's "
                               "own diffusers folder ships text-only (no image config), so this points at any "
                               "full Qwen3-VL-4B checkpoint folder that has one - the vision tower is identical, "
                               "only the image-preprocessing CONFIG (resize/normalize rules) is read from here."}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 1024,
                    "tooltip": "Must match the value your generation node uses (Krea2 default 512)."}),
                "print_prompt": ("BOOLEAN", {"default": True,
                    "tooltip": "Print the assembled prompt text + token/image-grid shapes to the console. "
                               "Turn this on for your first test of this node."}),
            },
        }

    # ── image prep ───────────────────────────────────────────────────────
    def _prep_pil(self, image_tensor, cap_mp: float):
        """ComfyUI IMAGE (B,H,W,C float 0-1) -> single PIL image, downscaled to
        fit cap_mp megapixels (never upscaled). Takes the first frame of the batch."""
        arr = (image_tensor[0].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB" if arr.shape[-1] == 3 else "RGBA")
        w, h = img.size
        cur_mp = (w * h) / 1_000_000.0
        if cur_mp > cap_mp:
            scale = (cap_mp / cur_mp) ** 0.5
            new_w = max(16, int(round(w * scale)))
            new_h = max(16, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img.convert("RGB")

    def encode(self, krea2_pipeline, prompt, image1=None, image2=None, image3=None,
               vision_position="before prompt", vision_megapixels=1.0,
               vision_processor_source=r"H:\Testing\Qwen3-VL-4B-Instruct-heretic-7refusal",
               max_sequence_length=512, print_prompt=True):
        pipe = krea2_pipeline["pipeline"]
        if pipe is None:
            from .krea2_multistage_ultra import EricKrea2MultistageUltra
            pipe = EricKrea2MultistageUltra._recover_unloaded_pipeline(krea2_pipeline)
        device = getattr(pipe, "_execution_device", None) or "cuda"

        images = [im for im in (image1, image2, image3) if im is not None]

        # ── text-only fallback: no images connected, behave like plain encode_prompt ──
        if not images:
            embeds, mask = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                              max_sequence_length=int(max_sequence_length))
            if print_prompt:
                print(f"[EricKrea2-VisionPrompt] no images connected; text-only encode. "
                      f"embeds {tuple(embeds.shape)}, mask {tuple(mask.shape)}")
            return ({"embeds": embeds, "mask": mask},)

        from transformers import AutoProcessor

        # ── 1) image prep (PIL, downscaled-only, never upscaled) ──
        # Krea2's text budget (512 tokens by default) is FIXED regardless of how many
        # images are connected, so a flat per-image cap that's safe for 1 image overflows
        # at 2-3 (confirmed empirically: 1 image fits fine at 0.40 MP, but 2 images at the
        # same 0.40 MP each overflows the whole budget before the prompt is even added).
        # Auto-share ~0.40 MP of budget across however many images are connected, and take
        # the smaller of that share and the user's ceiling - so plugging in a 2nd or 3rd
        # reference image doesn't require manually retuning vision_megapixels each time.
        _shared_budget_mp = 0.40
        _auto_share_mp = _shared_budget_mp / max(1, len(images))
        _effective_mp = min(float(vision_megapixels), _auto_share_mp)
        if print_prompt and len(images) > 1:
            print(f"[EricKrea2-VisionPrompt] {len(images)} images connected; auto-sharing budget -> "
                  f"{_effective_mp:.3f} MP/image (ceiling was {float(vision_megapixels):.3f})")
        pil_images = [self._prep_pil(im, _effective_mp) for im in images]

        # ── 2) run the HF image processor to get pixel_values + image_grid_thw ──
        # AutoProcessor resolves to Qwen3VLProcessor wrapping a Qwen2VLImageProcessor
        # (Qwen3-VL reuses Qwen2-VL's image processor - there is no Qwen3-specific
        # class in this transformers version; verified against the installed build
        # rather than assumed). Using AutoProcessor avoids hardcoding either name.
        proc = AutoProcessor.from_pretrained(vision_processor_source)
        image_processor = proc.image_processor
        img_inputs = image_processor(images=pil_images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device=device, dtype=pipe.text_encoder.dtype)
        image_grid_thw = img_inputs["image_grid_thw"].to(device)

        # ── 3) build the "Picture N: <image>" placeholder text (ethanfel's convention) ──
        image_token = "<|image_pad|>"
        vision_start, vision_end = "<|vision_start|>", "<|vision_end|>"
        image_prompt = ""
        for slot in range(len(pil_images)):
            image_prompt += f"Picture {slot + 1}: {vision_start}{image_token}{vision_end}"
        user_text = (prompt + image_prompt) if vision_position == "after prompt" else (image_prompt + prompt)

        # ── 4) expand each <|image_pad|> into the right count of repeats -
        #        Qwen3VLProcessor's own algorithm, reused verbatim (not re-derived). ──
        merge_length = image_processor.merge_size ** 2
        expanded = user_text
        idx = 0
        while image_token in expanded:
            num_tokens = int(image_grid_thw[idx].prod().item()) // merge_length
            expanded = expanded.replace(image_token, "<|placeholder|>" * num_tokens, 1)
            idx += 1
        expanded = expanded.replace("<|placeholder|>", image_token)

        # ── 5) tokenize with Krea2's OWN fixed-length, pad-in-the-middle scheme
        #        (verbatim structure from Krea2Pipeline.get_text_hidden_states) ──
        prefix_idx = pipe.prompt_template_encode_start_idx
        n_suffix = pipe.prompt_template_encode_num_suffix_tokens
        full_text = pipe.prompt_template_encode_prefix + expanded
        budget = int(max_sequence_length) + prefix_idx - n_suffix

        # Validate BEFORE tokenizing with truncation=True. Silent truncation here would
        # cut into the middle of the image-placeholder run, corrupting the token/position
        # alignment - that's exactly what produced a cryptic 'shape mismatch' three layers
        # deep inside get_rope_index instead of a clear message at the actual mistake.
        true_len = len(pipe.tokenizer(full_text, truncation=False).input_ids)
        n_image_tokens = expanded.count(image_token)
        if true_len > budget:
            raise ValueError(
                f"[EricKrea2-VisionPrompt] Prompt + image(s) is {true_len} tokens, but Krea2's fixed text "
                f"budget at max_sequence_length={int(max_sequence_length)} is only {budget} tokens "
                f"(of which {n_image_tokens} are image placeholders alone, from {len(pil_images)} image(s) "
                f"at vision_megapixels={float(vision_megapixels)}). Truncating would silently cut into the "
                f"image token block and corrupt the position ids (this is the 'shape mismatch...get_rope_index' "
                f"crash). Fix by: lowering vision_megapixels (roughly {n_image_tokens} tokens now; try "
                f"{max(0.1, float(vision_megapixels) * budget / true_len * 0.85):.2f}), using fewer reference "
                f"images, shortening the prompt, or raising max_sequence_length - though note Krea2 was trained "
                f"with a fixed 512-token budget, so going well above that is untested territory for output "
                f"quality, not just a number to turn up.")

        text_tokens = pipe.tokenizer(
            [full_text], truncation=True, padding="max_length",
            max_length=budget,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = pipe.tokenizer([pipe.prompt_template_encode_suffix], return_tensors="pt").to(device)
        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

        # ── 6) mm_token_type_ids (text=0 / image=1), needed by get_rope_index ──
        # Build a Qwen3VLProcessor paired with KREA2's OWN tokenizer (not the one
        # AutoProcessor loaded from vision_processor_source) so the image_token_id
        # it looks up matches the vocab/IDs actually used in step 5's tokenization.
        # In practice these are almost certainly identical vocabs (same Qwen3-VL-4B
        # family), but pairing it with pipe.tokenizer explicitly removes that as an
        # assumption rather than relying on it silently matching.
        # NOTE: needs video_processor too (ProcessorMixin rejects None for it), and
        # create_mm_token_type_ids returns a plain nested list, not a tensor - both
        # confirmed by an actual dry run against real Krea2/Qwen3-VL files, not assumed.
        from transformers import Qwen3VLProcessor
        mm_proc = Qwen3VLProcessor(image_processor=image_processor, tokenizer=pipe.tokenizer,
                                   video_processor=proc.video_processor)
        mm_token_type_ids = torch.tensor(
            mm_proc.create_mm_token_type_ids(input_ids.cpu()), dtype=torch.long, device=device)

        # ── 7) correct multimodal position ids for THIS mixed text+image sequence -
        #        Qwen3VLModel's own get_rope_index, not Krea2's text-only shortcut. ──
        position_ids, _ = pipe.text_encoder.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw, attention_mask=attention_mask,
        )

        if print_prompt:
            print(f"[EricKrea2-VisionPrompt] assembled prompt (prefix + user turn):\n"
                  f"{pipe.prompt_template_encode_prefix}{expanded}")
            print(f"[EricKrea2-VisionPrompt] {len(pil_images)} image(s), sizes={[im.size for im in pil_images]}, "
                  f"image_grid_thw={image_grid_thw.tolist()}, input_ids {tuple(input_ids.shape)}, "
                  f"position_ids {tuple(position_ids.shape)}")

        # ── 8) run the text encoder with real vision inputs and tap the same layers ──
        outputs = pipe.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = torch.stack([outputs.hidden_states[i] for i in pipe.text_encoder_select_layers], dim=2)
        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]

        if print_prompt:
            print(f"[EricKrea2-VisionPrompt] final conditioning: embeds {tuple(hidden_states.shape)}, "
                  f"mask {tuple(attention_mask.shape)}")

        return ({"embeds": hidden_states, "mask": attention_mask},)


NODE_CLASS_MAPPINGS = {"EricKrea2VisionPrompt": EricKrea2VisionPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2VisionPrompt": "Eric Krea2 Vision Prompt"}
