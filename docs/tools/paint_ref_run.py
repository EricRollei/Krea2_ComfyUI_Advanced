"""Reference run of the AUTHOR's AnyPaint pipeline (Krea2OstrisEditPipeline + known restore,
Apache-2.0) on showcase cases, bf16, 8 steps, guidance 0. Saves each result plus the exact
prompt embeddings (VLM saw the condition) so our runtime can be compared on equal conditioning.

Usage: python paint_ref_run.py CASE_ID [CASE_ID ...]
"""

import importlib.util
import json
import os
import sys
import time

import torch
from PIL import Image

SRC = r"A:\ai-tools\scratch\paint_adapters_src"
TEST = r"A:\ai-tools\scratch\paint_test"
TURBO = r"H:\Training\Krea-2-Turbo"
LORA = r"L:\Models\loras\Krea2\paint\krea2_anypaint_rank32.safetensors"
OUT = os.path.join(TEST, "out")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, SRC)
    pl = load_mod("anypaint_pipeline", os.path.join(SRC, "krea2-anypaint__pipeline.py"))
    ap = load_mod("anypaint_helpers", os.path.join(SRC, "krea2-anypaint__anypaint.py"))
    man = {c["id"]: c for c in json.load(open(os.path.join(TEST, "manifest.json")))}
    t0 = time.perf_counter()
    pipe = pl.Krea2OstrisEditPipeline.from_pretrained(
        TURBO, torch_dtype=torch.bfloat16
    ).to("cuda:0")
    pipe.load_lora_weights(LORA, adapter_name="anypaint")
    pipe.set_adapters(["anypaint"], weights=[1.0])
    print(
        f"author pipeline loaded {time.perf_counter() - t0:.0f}s on {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for cid in sys.argv[1:]:
        c = man[cid]
        w, h = c["canvas_size"]
        prep = ap.prepare_anypaint(
            Image.open(os.path.join(TEST, c["source"])),
            Image.open(os.path.join(TEST, c["mask"])),
            (w, h),
            tuple(c["bbox"]),
        )
        cond_t = pipe._to_chw_tensor(prep.condition)
        vl = pipe._prep_vl_images([cond_t.to("cuda:0")], 384 * 384)
        emb, emb_mask = pipe.encode_prompt(
            c["prompt"], vl, 1, 512, torch.device("cuda:0")
        )
        torch.save(
            {
                "embeds": emb.cpu(),
                "mask": emb_mask.cpu(),
                "condition_size": prep.condition.size,
            },
            os.path.join(OUT, f"{cid}_embeds.pt"),
        )
        prep.condition.save(os.path.join(OUT, f"{cid}_condition.png"))
        g = torch.Generator(device="cpu").manual_seed(int(c["edit_seed"]))
        t = time.perf_counter()
        img = pipe(
            prompt=c["prompt"],
            image=prep.condition,
            width=w,
            height=h,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=g,
            reference_max_pixels=384 * 384,
            reference_placements=[prep.reference_placement],
            encode_reference_in_prompt=True,
            kv_cache=True,
            known_image=prep.known_image,
            known_mask=prep.keep_mask,
        ).images[0]
        img.save(os.path.join(OUT, f"{cid}_author.png"))
        print(
            f"{cid}: author pipeline {time.perf_counter() - t:.1f}s, cond {prep.condition.size}, "
            f"embeds {tuple(emb.shape)}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
