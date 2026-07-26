# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Sweep Plan / Sweep -> Preset
=======================================
Purpose-built parameter sweeping for the Ultra V2 node (spec:
docs/spec_sweep_system_2026-07-24_v2_final.md).

`EricKrea2SweepPlan` emits a KREA2_SWEEP plan (grid or JSON-lines combos) that
Ultra V2 executes inside ONE queue item when wired into its `sweep` input -
model + LoRA state stay resident, every existing pipeline feature runs
untouched per cell. The engine itself lives in `_sweep_core.py` and is
package-agnostic (the porting template for other node sets).

`EricKrea2SweepToPreset` promotes a swept cell from a sweep_manifest.json into
`ultra_presets.json` (same shape the ★ Save Preset endpoint writes), auto-tagged
with the checkpoint name so presets stay per-fine-tune.
"""

from __future__ import annotations

import json
import os

from .krea2_multistage_ultra import _SAMPLERS, _SCHEDULES, _NOISE_TYPES

_SHIFT_MODES = ["fixed", "resolution", "manual"]

_STAGES = {"s1": ("s1",), "s2": ("s2",), "s3": ("s3",), "all_stages": ("s1", "s2", "s3")}

_SORT_CHOICES = ["plan_order", "sharpness_laplacian", "sharpness_tenengrad",
                 "noise_sigma", "flat_chroma_var", "clip_fraction", "time_s"]


def _split(s: str) -> list:
    return [t.strip() for t in str(s or "").replace("\n", ",").split(",") if t.strip()]


def _floats(s: str, name: str) -> list:
    out = []
    for t in _split(s):
        try:
            out.append(float(t))
        except ValueError:
            raise ValueError(f"SweepPlan: '{name}' entry '{t}' is not a number")
    return out


def _ints(s: str, name: str) -> list:
    return [int(round(v)) for v in _floats(s, name)]


def _validate(vals: list, allowed: list, name: str):
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(f"SweepPlan: unknown {name} {bad}; allowed: {allowed}")


class EricKrea2SweepPlan:
    """Build a KREA2_SWEEP plan. Grid axes left blank are not swept (the Ultra
    panel's live value is used). Hard-refuses over-cap grids with an axis-size
    breakdown instead of silently truncating."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["grid", "combos"], {"default": "grid"}),
                "stage_scope": (["s1", "s2", "s3", "all_stages"], {"default": "s1",
                    "tooltip": "Which stage(s) the swept keys apply to. all_stages sets the same "
                               "value on s1+s2+s3 together (NOT a per-stage cross product). "
                               "Workflow intent: perfect S1 first (quick_screen), then sweep "
                               "s2/s3 refinement on the short list."}),
            },
            "optional": {
                "samplers": ("STRING", {"multiline": True, "default": "",
                    "tooltip": f"Comma/newline-separated. Allowed: {', '.join(_SAMPLERS)}. Blank = not swept."}),
                "schedulers": ("STRING", {"multiline": True, "default": "",
                    "tooltip": f"Allowed: {', '.join(_SCHEDULES)}. Blank = not swept."}),
                "pair_lock": ("BOOLEAN", {"default": False,
                    "tooltip": "ON: zip samplers+schedulers line-by-line into fixed pairs (lists must "
                               "be equal length) - for testing known recipes. OFF: full cross product."}),
                "steps": ("STRING", {"default": "",
                    "tooltip": "e.g. '8, 10, 12'. Maps to <stage>_steps."}),
                "windows": ("STRING", {"default": "",
                    "tooltip": "Refine windows 'start-end', e.g. '5-12, 6-12, 7-12'. Maps to "
                               "<stage>_start_step/<stage>_end_step. Pair with matching steps or "
                               "leave steps blank and window within the panel's step count."}),
                "etas": ("STRING", {"default": "", "tooltip": "e.g. '0, 0.3, 1.0' -> <stage>_eta."}),
                "noise_types": ("STRING", {"default": "",
                    "tooltip": f"Allowed: {', '.join(_NOISE_TYPES)}. lcm-family samplers force white "
                               "(per the 2026-07-23 rainbow fix); those cells are auto-coerced and "
                               "labeled, never silently divergent."}),
                "cfgs": ("STRING", {"default": "",
                    "tooltip": "e.g. '1.0, 1.5, 2.0' -> <stage>_cfg. Diagnostic for RAW-drifted fine "
                               "tunes: tolerance of cfg>1 tracks how far training eroded the turbo "
                               "distillation."}),
                "shift_modes": ("STRING", {"default": "",
                    "tooltip": "distilled_shift values to sweep: fixed / resolution / manual."}),
                "mus": ("STRING", {"default": "",
                    "tooltip": "shift mu values for the scoped stage(s), e.g. '1.15, 2.1, 3.0'. "
                               "Forces distilled_shift=manual on those cells (unless shift_modes is "
                               "also swept, which then wins the cross)."}),
                "combos_file": ("STRING", {"default": "",
                    "tooltip": "combos mode: path to a JSON file with the runs - either a JSON "
                               "array of override objects, {\"combos\": [...]}, or JSON-lines. "
                               "Takes precedence over combos_json below. Keeps sweep campaigns "
                               "as versionable files."}),
                "combos_json": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "combos mode: one JSON object per line using Ultra V2 widget keys "
                               "(same keys as presets / PNG recipes), e.g.\n"
                               '{"s2_sampler": "lcm_hybrid", "s2_steps": 12, "s2_start_step": 6}\n'
                               "Copy-paste directly from a preset or a stamped PNG."}),
                "seed_policy": (["fixed", "seeds_axis"], {"default": "fixed",
                    "tooltip": "fixed: every cell uses the Ultra panel seed (required for fair "
                               "comparison). seeds_axis: 'seeds' below becomes one more grid axis."}),
                "seeds": ("STRING", {"default": "", "tooltip": "seeds_axis values, e.g. '1, 2, 3'."}),
                "quick_screen": ("BOOLEAN", {"default": False,
                    "tooltip": "Force S1-only at screen_megapixels for every cell (upscale_to_stage2/3 "
                               "= 0). Cheap wide screening; re-run the short list full-pipeline after."}),
                "screen_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 8.0, "step": 0.25}),
                "score_metrics": ("BOOLEAN", {"default": True,
                    "tooltip": "Compute technical metrics per cell (CPU, ms): Laplacian + Tenengrad "
                               "sharpness, wavelet noise sigma, flat-region chroma variance "
                               "(splotch detector), clip fraction. Scores annotate + can sort the "
                               "sheet; they never hide a cell."}),
                "sort_by": (_SORT_CHOICES, {"default": "plan_order",
                    "tooltip": "Contact-sheet order. Metrics sort best-first; time_s fastest-first. "
                               "Manifest keeps plan order + this ordering separately."}),
                "tile_px": ("INT", {"default": 448, "min": 128, "max": 1024, "step": 32}),
                "max_combos": ("INT", {"default": 64, "min": 1, "max": 4096,
                    "tooltip": "Hard cap. Over-cap plans REFUSE with an axis-size breakdown "
                               "(no silent truncation)."}),
                "output_folder": ("STRING", {"default": "",
                    "tooltip": "Where sweep folders are created. Blank = ComfyUI output/sweeps. "
                               "An absolute path is used as-is; a relative path is created "
                               "under the ComfyUI output directory."}),
                "hybrid_splits": ("STRING", {"default": "",
                    "tooltip": "Sweep hybrid_split, e.g. '0.3, 0.5, 0.7' (0.05-0.95). Handoff "
                               "fraction for lcm_hybrid / lcm_hybrid2 only - pair with a hybrid "
                               "in the samplers axis (or on the panel); it does nothing on other "
                               "samplers, and such cells are deduped as inert."}),
                "megapixels": ("STRING", {"default": "",
                    "tooltip": "Sweep s1_megapixels, e.g. '0.5, 1.0, 2.0, 4.5'. Always targets "
                               "Stage 1 regardless of stage_scope (stage 2/3 sizes chain from it "
                               "via the upscale factors). Mixed cell sizes are fine: metrics and "
                               "the contact sheet use each cell's native output; the IMAGE batch "
                               "output center-pads smaller cells to the largest. quick_screen "
                               "does NOT override a swept megapixels value. Note: resolution "
                               "changes the noise grid, so composition shifts with size even at "
                               "a fixed seed - that is part of what you are measuring."}),
                "lora_mode": (["none", "bakeoff", "strength_ladder"], {"default": "none",
                    "tooltip": "bakeoff: each LoRA in lora_files at each lora_strengths value - "
                               "by default the swept LoRA runs ALONE (panel stack replaced) so "
                               "every style is measured against the bare checkpoint; "
                               "keep_panel_stack layers it on top instead. strength_ladder: keep "
                               "the panel-declared stack, step ONLY the lora_target entry "
                               "through lora_strengths. Sweep-injected entries are ephemeral "
                               "(fused per cell, cleaned per cell)."}),
                "lora_files": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "bakeoff: one LoRA per line - a name from the loras folder "
                               "(same resolution as the LoRA Stack node) or a full path."}),
                "lora_strengths": ("STRING", {"default": "",
                    "tooltip": "e.g. '0.6, 0.8, 1.0'. One value drives all three stage weights "
                               "(weight_s1=s2=s3). bakeoff: every file at every value; "
                               "ladder: the target entry's values."}),
                "lora_target": ("STRING", {"default": "1",
                    "tooltip": "strength_ladder: which panel-stack entry to vary - 1-based "
                               "index, or a name substring (matches lora_name/filename)."}),
                "include_baseline": ("BOOLEAN", {"default": True,
                    "tooltip": "Add one reference cell: bakeoff = no LoRA at all (or the panel "
                               "stack alone when keep_panel_stack is ON); ladder = the panel "
                               "stack exactly as declared."}),
                "keep_panel_stack": ("BOOLEAN", {"default": False,
                    "tooltip": "bakeoff only: append the swept LoRA ON TOP of the panel stack "
                               "instead of replacing it."}),
                "prompts": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "One prompt per line = a prompt axis (slowest axis, so the sheet "
                               "and CSV read in prompt-major blocks). Each prompt cell re-encodes "
                               "(prompt_conditioning is cleared; ~seconds/cell). Metric "
                               "comparisons are only meaningful WITHIN a prompt block."}),
                "sigma_profiles": ("STRING", {"default": "",
                    "tooltip": "Comma/newline list of saved sigma-shape preset NAMES (the ★ "
                               "presets of the Eric Krea2 Sigmas node). Each cell rebuilds the "
                               "full bundle from that preset and it OVERRIDES any wired sigmas "
                               "node for that cell. Blank = not swept."}),
            },
        }

    RETURN_TYPES = ("KREA2_SWEEP",)
    RETURN_NAMES = ("sweep",)
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"

    def build(self, mode="grid", stage_scope="s1", samplers="", schedulers="", pair_lock=False,
              steps="", windows="", etas="", noise_types="", cfgs="", shift_modes="", mus="",
              combos_json="", combos_file="", seed_policy="fixed", seeds="", quick_screen=False,
              screen_megapixels=1.0, score_metrics=True, sort_by="plan_order",
              tile_px=448, max_combos=64, output_folder="", megapixels="", hybrid_splits="",
              lora_mode="none", lora_files="", lora_strengths="", lora_target="1",
              include_baseline=True, keep_panel_stack=False, prompts="", sigma_profiles=""):
        from .._sweep_core import expand_grid, diff_label
        stages = _STAGES[stage_scope]

        # v2 axes shared by both modes. Order matters: expand_grid makes the
        # FIRST axis the slowest, so prompts -> profile -> lora lead, giving
        # prompt-major (then profile-major) blocks on the sheet and CSV.
        lead_axes = {}
        pl = [ln.strip() for ln in str(prompts or "").splitlines() if ln.strip()]
        if pl:
            lead_axes["prompt"] = [{"prompt": s} for s in pl]
        sp = _split(sigma_profiles)
        if sp:
            try:
                from .. import _settings as _st
                known = set(_st.list_preset_names("sigmas"))
                known.discard("custom")
                bad = [s for s in sp if s not in known]
                if bad:
                    raise ValueError(f"SweepPlan: unknown sigma profile(s) {bad}; "
                                     f"saved: {sorted(known) or '(none)'}")
            except ImportError:
                pass
            lead_axes["sigma_profile"] = [{"sigma_profile": s} for s in sp]
        if lora_mode != "none":
            sv = _floats(lora_strengths, "lora_strengths")
            vals = []
            if include_baseline:
                vals.append({"lora_baseline": True})
            if lora_mode == "bakeoff":
                files = [ln.strip() for ln in str(lora_files or "").splitlines() if ln.strip()]
                if not files:
                    raise ValueError("SweepPlan: lora_mode=bakeoff but lora_files is empty")
                if not sv:
                    raise ValueError("SweepPlan: lora_mode=bakeoff but lora_strengths is empty")
                vals += [{"lora_file": f, "lora_strength": s} for f in files for s in sv]
            else:  # strength_ladder
                if not sv:
                    raise ValueError("SweepPlan: strength_ladder needs lora_strengths")
                tgt = str(lora_target or "1").strip()
                vals += [{"lora_strength": s, "lora_target": tgt} for s in sv]
            lead_axes["lora"] = vals

        def per_stage(key_suffix, value):
            return {f"{st}_{key_suffix}": value for st in stages}

        if mode == "combos":
            combos = []
            src_text, src_name = str(combos_json or ""), "combos_json"
            if str(combos_file or "").strip():
                path = str(combos_file).strip().strip('"')
                with open(path, "r", encoding="utf-8") as f:
                    src_text = f.read()
                src_name = os.path.basename(path)
                try:  # whole-file JSON first: [ {...}, ... ] or {"combos": [...]}
                    doc = json.loads(src_text)
                    if isinstance(doc, dict):
                        doc = doc.get("combos", [])
                    if isinstance(doc, list):
                        for i, d in enumerate(doc):
                            if not isinstance(d, dict) or not d:
                                raise ValueError(f"SweepPlan: {src_name} entry {i} is not a "
                                                 f"non-empty JSON object")
                        combos = list(doc)
                        src_text = ""  # consumed
                except ValueError:
                    raise
                except Exception:
                    pass  # fall through to JSON-lines parsing of the file text
            for ln, line in enumerate(src_text.splitlines(), 1):
                line = line.strip().rstrip(",")
                if not line or line.startswith("#"):
                    continue
                try:
                    d = json.loads(line)
                except Exception as e:
                    raise ValueError(f"SweepPlan {src_name} line {ln}: not valid JSON ({e}): {line[:120]}")
                if not isinstance(d, dict) or not d:
                    raise ValueError(f"SweepPlan combos line {ln}: expected a non-empty JSON object")
                combos.append(d)
            if not combos:
                raise ValueError(f"SweepPlan: combos mode but {src_name} has no entries")
            if lead_axes:
                cross = dict(lead_axes)
                cross["explicit"] = combos
                combos = expand_grid(cross)
        else:
            smp = _split(samplers)
            sch = _split(schedulers)
            _validate(smp, _SAMPLERS, "sampler(s)")
            _validate(sch, _SCHEDULES, "scheduler(s)")
            nz = _split(noise_types)
            _validate(nz, _NOISE_TYPES, "noise_type(s)")
            shm = _split(shift_modes)
            _validate(shm, _SHIFT_MODES, "shift_mode(s)")
            axes = {}
            if pair_lock and smp and sch:
                if len(smp) != len(sch):
                    raise ValueError(f"SweepPlan: pair_lock needs equal-length lists "
                                     f"(samplers={len(smp)}, schedulers={len(sch)})")
                axes["pair"] = [dict(per_stage("sampler", a), **per_stage("schedule", b))
                                for a, b in zip(smp, sch)]
            else:
                if smp:
                    axes["sampler"] = [per_stage("sampler", v) for v in smp]
                if sch:
                    axes["scheduler"] = [per_stage("schedule", v) for v in sch]
            if steps.strip():
                axes["steps"] = [per_stage("steps", v) for v in _ints(steps, "steps")]
            if windows.strip():
                wvals = []
                for t in _split(windows):
                    try:
                        a, b = t.split("-")
                        wvals.append((int(a), int(b)))
                    except Exception:
                        raise ValueError(f"SweepPlan: window '{t}' must be 'start-end', e.g. '6-12'")
                axes["window"] = [dict(per_stage("start_step", a), **per_stage("end_step", b))
                                  for a, b in wvals]
            if etas.strip():
                axes["eta"] = [per_stage("eta", v) for v in _floats(etas, "etas")]
            if nz:
                axes["noise"] = [per_stage("noise", v) for v in nz]
            if cfgs.strip():
                axes["cfg"] = [per_stage("cfg", v) for v in _floats(cfgs, "cfgs")]
            if hybrid_splits.strip():
                hs = _floats(hybrid_splits, "hybrid_splits")
                bad = [v for v in hs if not (0.05 <= v <= 0.95)]
                if bad:
                    raise ValueError(f"SweepPlan: hybrid_splits out of range (0.05-0.95): {bad}")
                axes["hybrid_split"] = [{"hybrid_split": v} for v in hs]
            if megapixels.strip():
                mps = _floats(megapixels, "megapixels")
                bad = [v for v in mps if not (0.1 <= v <= 16)]
                if bad:
                    raise ValueError(f"SweepPlan: megapixels values out of range (0.1-16): {bad}")
                axes["megapixels"] = [{"s1_megapixels": v} for v in mps]
            if shm:
                axes["shift_mode"] = [{"distilled_shift": v} for v in shm]
            if mus.strip():
                mu_axis = []
                for v in _floats(mus, "mus"):
                    d = {f"shift_mu_{st}": v for st in stages}
                    if not shm:  # mu only acts under manual; force it unless modes are swept too
                        d["distilled_shift"] = "manual"
                    mu_axis.append(d)
                axes["mu"] = mu_axis
            if seed_policy == "seeds_axis":
                sv = _ints(seeds, "seeds")
                if not sv:
                    raise ValueError("SweepPlan: seed_policy=seeds_axis but 'seeds' is empty")
                axes["seed"] = [{"_seed": v} for v in sv]
            if lead_axes:
                merged = dict(lead_axes)
                merged.update(axes)
                axes = merged
            if not axes:
                raise ValueError("SweepPlan: grid mode but every axis is blank - nothing to sweep")
            n = 1
            for v in axes.values():
                n *= len(v)
            if n > max_combos:
                sizes = ", ".join(f"{k}={len(v)}" for k, v in axes.items())
                raise ValueError(f"SweepPlan: {n} combos exceeds max_combos={max_combos} "
                                 f"(axes: {sizes}). Trim an axis or raise the cap deliberately.")
            combos = expand_grid(axes)

        # lcm-family x colored-noise -> coerce to white and mark (2026-07-23 fix)
        for c in combos:
            for st in ("s1", "s2", "s3"):
                sm, nk = c.get(f"{st}_sampler", ""), f"{st}_noise"
                if str(sm).startswith("lcm") and c.get(nk) not in (None, "white"):
                    c[nk] = "white"
                    c["_coerced_noise"] = True
        # Drop cells that differ only by inert knobs (2026-07-25): euler clamps
        # eta to 0 by contract (and noise shaping only acts at eta>0), and lcm
        # ignores both (always full white re-noise). Identical effective
        # recipes would burn GPU minutes producing byte-identical results, so
        # duplicates are removed with a console note. lcm_hybrid/2 keep eta
        # (their tail sampler uses it).
        def _effective_key(c):
            d = {k: v for k, v in c.items() if not str(k).startswith("_")}
            for st in ("s1", "s2", "s3"):
                if str(d.get(f"{st}_sampler", "")) in ("euler", "lcm"):
                    d.pop(f"{st}_eta", None)
                    d.pop(f"{st}_noise", None)
            # hybrid_split is global but only the hybrids read it. If the cell
            # explicitly sets sampler(s) and none is a hybrid, the split is
            # inert for this cell. (No explicit sampler -> panel decides ->
            # keep it, we can't know here.)
            samplers = [str(d[k]) for k in ("s1_sampler", "s2_sampler", "s3_sampler") if k in d]
            if samplers and not any(sm.startswith("lcm_hybrid") for sm in samplers):
                d.pop("hybrid_split", None)
            return json.dumps(d, sort_keys=True)
        _seen, _unique = set(), []
        for c in combos:
            k = _effective_key(c)
            if k in _seen:
                continue
            _seen.add(k)
            _unique.append(c)
        if len(_unique) < len(combos):
            print(f"[EricKrea2-Sweep] removed {len(combos) - len(_unique)} inert duplicate "
                  f"cell(s) (eta/noise have no effect for euler and lcm)")
        combos = _unique
        if len(combos) > max_combos:
            raise ValueError(f"SweepPlan: {len(combos)} combos exceeds max_combos={max_combos}")
        plan = {"version": 2, "mode": mode, "stage_scope": stage_scope,
                "lora": ({"mode": lora_mode, "keep_panel_stack": bool(keep_panel_stack)}
                         if lora_mode != "none" else None),
                "combos": combos, "labels": [diff_label(c) for c in combos],
                "seed_policy": seed_policy, "quick_screen": bool(quick_screen),
                "screen_megapixels": float(screen_megapixels),
                "score_metrics": bool(score_metrics), "sort_by": sort_by,
                "tile_px": int(tile_px), "output_folder": str(output_folder or "").strip()}
        print(f"[EricKrea2-Sweep] plan: {len(combos)} combo(s), scope={stage_scope}, "
              f"mode={mode}" + (", QUICK-SCREEN" if quick_screen else ""))
        return (plan,)


class EricKrea2SweepToPreset:
    """Promote a sweep cell into ultra_presets.json (the ★ preset library).
    Reads the cell's recipe from sweep_manifest.json; 'swept_only' saves a
    partial preset of just the swept keys (composable over any panel), while
    'full_recipe' snapshots the whole resolved ultra section."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": ("STRING", {"default": "", "tooltip":
                    "Wire from Ultra V2's sweep_manifest output, or paste the path."}),
                "cell_index": ("INT", {"default": 0, "min": 0, "max": 9999,
                    "tooltip": "The #index shown on the contact-sheet caption (plan order)."}),
                "preset_name": ("STRING", {"default": ""}),
            },
            "optional": {
                "save_scope": (["swept_only", "full_recipe"], {"default": "swept_only"}),
                "objective": ("STRING", {"default": "", "tooltip":
                    "Free-text tag stored with the preset, e.g. 'fast', 'quality'."}),
                "append_checkpoint": ("BOOLEAN", {"default": True, "tooltip":
                    "Append ' [checkpoint]' to the preset name - keeps presets per-fine-tune."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save"
    CATEGORY = "Eric/Krea2"
    OUTPUT_NODE = True

    def save(self, manifest_path, cell_index, preset_name, save_scope="swept_only",
             objective="", append_checkpoint=True):
        if not str(manifest_path).strip():
            raise ValueError("SweepToPreset: manifest_path is empty")
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        cell = next((c for c in man.get("cells", []) if c.get("index") == int(cell_index)), None)
        if cell is None:
            raise ValueError(f"SweepToPreset: no cell #{cell_index} in {manifest_path}")
        if "error" in cell:
            raise ValueError(f"SweepToPreset: cell #{cell_index} failed ({cell['error']}) - "
                             f"nothing to promote")
        fields = dict(cell["overrides"] if save_scope == "swept_only"
                      else cell.get("resolved", {}))
        fields.pop("seed", None)
        fields.pop("_seed", None)
        if not fields:
            raise ValueError("SweepToPreset: cell has no serializable fields")
        fields["_sweep_meta"] = {  # ignored on apply (key filter), kept for provenance
            "checkpoint": man.get("checkpoint", ""), "objective": objective,
            "sweep_id": man.get("sweep_id", ""), "cell": int(cell_index),
            "metrics": cell.get("metrics", {}), "time_s": cell.get("time_s")}
        name = str(preset_name).strip() or f"sweep_{man.get('sweep_id', 'x')}_c{cell_index}"
        if append_checkpoint and man.get("checkpoint"):
            tag = f" [{man['checkpoint']}]"
            if not name.endswith(tag):
                name += tag
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ultra_presets.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lib = json.load(f)
            if not isinstance(lib, dict):
                lib = {}
        except Exception:
            lib = {}
        lib[name] = {"ultra": fields}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(lib, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        n_real = len([k for k in fields if not k.startswith("_")])
        status = (f"saved preset '{name}' ({save_scope}, {n_real} field(s)) "
                  f"-> ultra_presets.json; dropdown refreshes after graph reload")
        print(f"[EricKrea2-Sweep] {status}")
        return (status,)


NODE_CLASS_MAPPINGS = {
    "EricKrea2SweepPlan": EricKrea2SweepPlan,
    "EricKrea2SweepToPreset": EricKrea2SweepToPreset,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EricKrea2SweepPlan": "Eric Krea2 Sweep Plan",
    "EricKrea2SweepToPreset": "Eric Krea2 Sweep → Preset",
}
