# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 edit / negative-guidance text encoding
=================================================
* grounded_encode  - the identity-edit LoRA's semantic path: Qwen3-VL sees the source
  image(s) AND the instruction in one user turn
      system(Krea2 default) | <vision>xN | instruction | suffix
  at NATURAL length (no 512 padding - the trainer encodes unpadded, and two 768 px
  references alone exceed 512 tokens). Grounding image: longest side <= grounding_px,
  LANCZOS (trainer: jitter 384..768, node default 768).
* NegPiP prompt syntax  (phrase:-1.0)  - negative weight = the phrase's attention VALUE
  rows are multiplied by the weight inside the joint blocks (negative = flipped: the
  concept is subtracted instead of added). Two placements:
    lifted   - the phrase is removed from the prompt, encoded on its own and its rows are
               appended to the conditioning (Qwen never reads the word in context);
    in_place - the phrase stays in the prompt; its tokens are flagged where they stand.
  Positive weights are not supported by Krea2 conditioning and are stripped (warned).
  Clean reimplementation of the NegPiP idea (hako-mikan/sd-webui-negpip); no code from
  the AGPL Krea2 port was used.
* plain_encode - pipe.encode_prompt (NAG negative, refine-stage prompts).

Every encoder returns a dict {"embeds", "mask", "vmul"}: vmul is None or a float tensor
[L] (1.0 = untouched, w < 0 = NegPiP row), aligned with embeds' token axis.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import re

import torch

_LOG = "[EricKrea2-EditText]"
_GROUP_RE = re.compile(r"(?<!\\)\(([^()]*?):\s*(-?\d+(?:\.\d+)?)\s*\)")
_VIS = "<|vision_start|><|image_pad|><|vision_end|>"


# -- NegPiP parsing ----------------------------------------------------------------------

def parse_weighted(text):
    """-> (clean_text, negatives [(phrase, w, (start, end) span in clean_text)], n_positive).
    Every (phrase:w) group is replaced by the bare phrase; spans index into clean_text."""
    out, negs, npos, last = [], [], 0, 0
    pos = 0
    for m in _GROUP_RE.finditer(text):
        out.append(text[last:m.start()])
        pos += len(text[last:m.start()])
        phrase, w = m.group(1).strip(), float(m.group(2))
        if w < 0 and phrase:
            negs.append((phrase, w, (pos, pos + len(phrase))))
        elif w != 1.0:
            npos += 1
        out.append(phrase)
        pos += len(phrase)
        last = m.end()
    out.append(text[last:])
    return "".join(out), negs, npos


def _tidy(text):
    for _ in range(3):
        prev = text
        text = re.sub(r"(?:,[ \t]*){2,}", ", ", text)
        text = re.sub(r"([.!?;:])[ \t]*,", r"\1", text)
        text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        if text == prev:
            break
    return text.strip(" \t,")


def lift_negatives(text):
    """-> (prompt without negative groups, [(phrase, w)], n_positive)."""
    negs, npos = [], 0

    def rep(m):
        nonlocal npos
        phrase, w = m.group(1).strip(), float(m.group(2))
        if w < 0 and phrase:
            negs.append((phrase, w))
            return ""
        if w != 1.0:
            npos += 1
        return phrase
    return _tidy(_GROUP_RE.sub(rep, text)), negs, npos


def _token_indices_for_spans(tokenizer, full_text, spans, char_offset):
    """Token indices (into tokenizer(full_text)) overlapping any char span (span coords
    are relative to the substring starting at char_offset)."""
    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    idx = []
    for ti, (a, b) in enumerate(enc["offset_mapping"]):
        for s, e, w in spans:
            if b > char_offset + s and a < char_offset + e:
                idx.append((ti, w))
                break
    return idx


# -- encoders ----------------------------------------------------------------------------

def plain_encode(pipe, prompt, device):
    e, m = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)
    return {"embeds": e, "mask": m, "vmul": None}


def _phrase_rows(pipe, phrase, device):
    """Hidden-state rows of a phrase encoded on its own (template prefix dropped, suffix
    excluded) -> [1, n, 12, D]."""
    e, m = pipe.get_text_hidden_states(phrase, 512, device)
    n_valid = int(m[0].sum().item())
    n_sfx = int(pipe.prompt_template_encode_num_suffix_tokens)
    n = max(0, n_valid - n_sfx)
    return e[:, :n]


def negpip_encode(pipe, prompt, device, mode="lifted", strength=1.0, log=print):
    """Plain prompt with NegPiP groups -> {"embeds", "mask", "vmul"}."""
    if mode == "lifted":
        clean, negs, npos = lift_negatives(prompt)
        if npos:
            log(f"{_LOG} {npos} positive (phrase:w) weight(s) ignored - Krea2 has no prompt weighting")
        out = plain_encode(pipe, clean, device)
        if not negs:
            return out
        rows, vm = [], []
        for phrase, w in negs:
            r = _phrase_rows(pipe, phrase, device)
            if r.shape[1] == 0:
                log(f"{_LOG} NegPiP phrase '{phrase}' encoded to nothing - skipped")
                continue
            rows.append(r)
            vm += [float(w) * float(strength)] * int(r.shape[1])
        if not rows:
            return out
        e = torch.cat([out["embeds"]] + [r.to(out["embeds"].dtype) for r in rows], dim=1)
        m = torch.cat([out["mask"], out["mask"].new_ones((1, len(vm)))], dim=1)
        vmul = torch.ones(e.shape[1], dtype=torch.float32)
        vmul[-len(vm):] = torch.tensor(vm)
        log(f"{_LOG} NegPiP (lifted): {len(rows)} phrase(s) -> {len(vm)} appended rows "
            f"[{', '.join(f'{p}:{w:g}' for p, w in negs)}]; prompt now: {clean[:160]!r}")
        return {"embeds": e, "mask": m, "vmul": vmul}
    # in_place
    clean, negs, npos = parse_weighted(prompt)
    if npos:
        log(f"{_LOG} {npos} positive (phrase:w) weight(s) ignored - Krea2 has no prompt weighting")
    out = plain_encode(pipe, clean, device)
    if not negs:
        return out
    prefix = pipe.prompt_template_encode_prefix
    pidx = int(pipe.prompt_template_encode_start_idx)
    hits = _token_indices_for_spans(pipe.tokenizer, prefix + clean,
                                    [(s, e_, w) for _, w, (s, e_) in negs], len(prefix))
    vmul = torch.ones(out["embeds"].shape[1], dtype=torch.float32)
    n = 0
    for ti, w in hits:
        j = ti - pidx
        if 0 <= j < vmul.numel() and bool(out["mask"][0, j]):
            vmul[j] = float(w) * float(strength)
            n += 1
    log(f"{_LOG} NegPiP (in_place): {len(negs)} phrase(s) -> {n} flagged tokens "
        f"[{', '.join(f'{p}:{w:g}' for p, w, _ in negs)}]")
    out["vmul"] = vmul if n else None
    return out


def grounding_pils(images, grounding_px):
    """ComfyUI IMAGEs -> PIL list, longest side capped at grounding_px (LANCZOS, trainer)."""
    import numpy as np
    from PIL import Image
    pils = []
    for im in images:
        arr = (im[0, :, :, :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr)
        cap = int(grounding_px or 0)
        if cap > 0 and max(pil.size) > cap:
            s = cap / max(pil.size)
            pil = pil.resize((max(1, round(pil.size[0] * s)), max(1, round(pil.size[1] * s))),
                             Image.LANCZOS)
        pils.append(pil.convert("RGB"))
    return pils


@torch.no_grad()
def grounded_encode(pipe, instruction, images, device, grounding_px=768, system_prompt="",
                    processor_source=None, negpip_mode=None, negpip_strength=1.0, log=print):
    """Image-grounded instruction encode (identity-edit training recipe), natural length.
    images: list of ComfyUI IMAGE (scene first, subject second). Returns the dict format."""
    from transformers import AutoProcessor, Qwen3VLProcessor
    instr = instruction or ""
    lifted, vm_rows = [], []
    if negpip_mode == "lifted":
        instr, negs, npos = lift_negatives(instr)
        lifted = negs
    elif negpip_mode == "in_place":
        instr, negs_in, npos = parse_weighted(instr)
    else:
        npos = 0
    if npos:
        log(f"{_LOG} {npos} positive (phrase:w) weight(s) ignored - Krea2 has no prompt weighting")
    pils = grounding_pils(images, grounding_px)
    src = processor_source or getattr(pipe, "_eric_vl_processor_source", None) \
        or r"H:\Testing\Qwen3-VL-4B-Instruct-heretic-7refusal"
    proc = AutoProcessor.from_pretrained(src)
    ip = proc.image_processor
    img_in = ip(images=pils, return_tensors="pt")
    pixel_values = img_in["pixel_values"].to(device=device, dtype=pipe.text_encoder.dtype)
    grid_thw = img_in["image_grid_thw"].to(device)

    prefix = pipe.prompt_template_encode_prefix
    if system_prompt and system_prompt.strip():
        prefix = ("<|im_start|>system\n" + system_prompt.strip()
                  + "<|im_end|>\n<|im_start|>user\n")
    pidx = len(pipe.tokenizer(prefix, add_special_tokens=False).input_ids)
    if not (system_prompt and system_prompt.strip()) and pidx != int(pipe.prompt_template_encode_start_idx):
        log(f"{_LOG} WARNING: prefix tokenizes to {pidx}, pipeline says "
            f"{pipe.prompt_template_encode_start_idx}")
    user = _VIS * len(pils) + instr
    merge = ip.merge_size ** 2
    expanded, i = user, 0
    while "<|image_pad|>" in expanded:
        n = int(grid_thw[i].prod().item()) // merge
        expanded = expanded.replace("<|image_pad|>", "<|placeholder|>" * n, 1)
        i += 1
    expanded = expanded.replace("<|placeholder|>", "<|image_pad|>")
    full = prefix + expanded + pipe.prompt_template_encode_suffix
    tok = pipe.tokenizer([full], return_tensors="pt", add_special_tokens=False).to(device)
    input_ids, attn = tok.input_ids, tok.attention_mask.bool()
    mm = Qwen3VLProcessor(image_processor=ip, tokenizer=pipe.tokenizer,
                          video_processor=proc.video_processor)
    mm_ids = torch.tensor(mm.create_mm_token_type_ids(input_ids.cpu()), dtype=torch.long,
                          device=device)
    pos_ids, _ = pipe.text_encoder.get_rope_index(input_ids=input_ids, mm_token_type_ids=mm_ids,
                                                  image_grid_thw=grid_thw, attention_mask=attn)
    out = pipe.text_encoder(input_ids=input_ids, attention_mask=attn, position_ids=pos_ids,
                            pixel_values=pixel_values, image_grid_thw=grid_thw,
                            output_hidden_states=True)
    hs = torch.stack([out.hidden_states[k] for k in pipe.text_encoder_select_layers], dim=2)
    embeds = hs[:, pidx:]
    mask = attn[:, pidx:]
    vmul = None
    if negpip_mode == "in_place" and negs_in:
        hits = _token_indices_for_spans(pipe.tokenizer, prefix + expanded,
                                        [(s, e_, w) for _, w, (s, e_) in negs_in],
                                        len(prefix) + len(expanded) - len(instr))
        vmul = torch.ones(embeds.shape[1], dtype=torch.float32)
        for ti, w in hits:
            j = ti - pidx
            if 0 <= j < vmul.numel():
                vmul[j] = float(w) * float(negpip_strength)
        log(f"{_LOG} NegPiP (in_place, grounded): {len(hits)} flagged tokens")
    if lifted:
        rows = []
        for phrase, w in lifted:
            r = _phrase_rows(pipe, phrase, device)
            if r.shape[1]:
                rows.append(r)
                vm_rows += [float(w) * float(negpip_strength)] * int(r.shape[1])
        if rows:
            embeds = torch.cat([embeds] + [r.to(embeds.dtype) for r in rows], dim=1)
            mask = torch.cat([mask, mask.new_ones((1, len(vm_rows)))], dim=1)
            vmul = torch.ones(embeds.shape[1], dtype=torch.float32)
            vmul[-len(vm_rows):] = torch.tensor(vm_rows)
            log(f"{_LOG} NegPiP (lifted, grounded): {len(rows)} phrase(s), {len(vm_rows)} rows")
    log(f"{_LOG} grounded encode: {len(pils)} image(s) {[p.size for p in pils]} "
        f"(grounding_px {grounding_px}), {embeds.shape[1]} conditioning tokens (natural length)")
    return {"embeds": embeds, "mask": mask, "vmul": vmul}
