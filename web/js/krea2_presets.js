// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - generic preset UI (dropdown + ★ Save Preset) for any node.
// One helper drives every preset-capable node via the PRESET_NODES table below:
// the node exposes a `<section>_preset` dropdown widget, and this extension wires
// up load (write the chosen preset's values into the node's own widgets so the
// panel reflects what will run) + save (POST the current widget values to
// /eric_krea2/save_preset under the node's section). Selecting a preset keeps the
// dropdown showing that name; editing any recipe widget flips it back to "custom"
// so the shown name never disagrees with the panel.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// nodeClass -> { section, presetWidget, skip:[...] }. `skip` widgets are never part
// of a reusable recipe (per-run values + the preset control itself). Any node
// added here gets the full preset UI with zero extra code.
const PRESET_NODES = {
    EricKrea2MultistageUltraV2: {
        section: "ultra", presetWidget: "ultra_preset",
        skip: ["prompt", "negative_prompt", "seed", "control_after_generate", "ultra_preset"],
    },
    EricKrea2ComponentLoader: {
        section: "loader", presetWidget: "loader_preset",
        skip: ["control_after_generate", "loader_preset"],
    },
    EricKrea2MultiLoRA: {
        section: "lora", presetWidget: "lora_preset",
        skip: ["control_after_generate", "lora_preset", "paste_lora"],
    },
    EricKrea2Sigmas: {
        section: "sigmas", presetWidget: "sigmas_preset",
        skip: ["control_after_generate", "sigmas_preset"],
    },
    EricKrea2SweepPlan: {
        section: "sweep", presetWidget: "sweep_preset",
        // run_notes (2026-10-15): per run, never part of a preset, and editing it keeps the preset name
        skip: ["control_after_generate", "sweep_preset", "enabled", "run_notes", "sweep_cell_count", "clear_sweep", "show_plan",
               "share_sheets", "share_background"],
    },
    EricKrea2ApplyLoRA: {
        section: "apply_lora", presetWidget: "apply_lora_preset",
        skip: ["control_after_generate", "apply_lora_preset"],
    },
    EricKrea2DecodeVAELoader: {
        section: "decode_vae", presetWidget: "decode_vae_preset",
        skip: ["control_after_generate", "decode_vae_preset"],
    },
};

function findWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

// 2026-10-23: the frontend also fires widget callbacks WITHOUT a real edit (queueing,
// focus/blur, node selection...). The dropdown only flips to "custom" when a widget's
// value actually differs from what the preset (or the loaded workflow) set.
function sameVal(a, b) {
    if (typeof a === "number" && typeof b === "number") {
        return Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b));
    }
    return String(a) === String(b);
}

function snapshot(node, cfg) {
    const skip = new Set(cfg.skip);
    const base = {};
    for (const w of node.widgets || []) {
        if (!w || !w.name || w.type === "button") continue;
        if (w.name === cfg.presetWidget || skip.has(w.name)) continue;
        base[w.name] = w.value;
    }
    node.__krea2_baseline = base;
}

// Selecting a preset writes its values into this node's widgets (so the panel
// reflects exactly what will run), and the dropdown keeps showing the name.
function wireLoad(node, cfg) {
    const pw = findWidget(node, cfg.presetWidget);
    if (!pw) return;
    const skip = new Set(cfg.skip);
    let cache = null; // { presetName: { <section>: {...} } }, fetched once per node

    // Flip the dropdown to "custom" when the USER edits a recipe widget (but not
    // while we are applying a preset ourselves).
    for (const w of node.widgets || []) {
        if (!w || !w.name || w.type === "button") continue;
        if (w.name === cfg.presetWidget || skip.has(w.name)) continue;
        if (w.__krea2_dirtyHooked) continue;
        w.__krea2_dirtyHooked = true;
        const origW = w.callback;
        w.callback = function (v, ...a) {
            const r = origW ? origW.call(this, v, ...a) : undefined;
            if (!node.__krea2_applying && pw.value !== "custom") {
                const base = node.__krea2_baseline;
                if (base && (w.name in base) && sameVal(w.value, base[w.name])) return r;  // no real edit
                console.log(`[Eric_Krea2] '${w.name}' changed (${base ? base[w.name] : "?"} -> ${w.value}): ` +
                            `preset '${pw.value}' -> custom`);
                pw.value = "custom";
                node.__krea2_baseline = null;
                node.setDirtyCanvas(true, true);
            }
            return r;
        };
    }

    // A workflow loaded with a preset name showing: its saved values are the baseline.
    const origConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
        try { if (pw.value && pw.value !== "custom") snapshot(node, cfg); } catch (_e) { /* ignore */ }
        return r;
    };

    const origCallback = pw.callback;
    pw.callback = async function (value, ...rest) {
        if (origCallback) origCallback.call(this, value, ...rest);
        if (!value || value === "custom") return;
        try {
            if (!cache) {
                const resp = await api.fetchApi(
                    `/eric_krea2/get_presets?section=${encodeURIComponent(cfg.section)}`);
                const j = await resp.json();
                if (!resp.ok || !j.ok) throw new Error(j.error || `HTTP ${resp.status}`);
                cache = j.presets || {};
            }
            const entry = cache[value];
            const fields = entry && (entry[cfg.section] || entry);
            if (!fields) {
                console.warn(`[Eric_Krea2] preset '${value}' not found in ${cfg.section}_presets.json`);
                return;
            }
            node.__krea2_applying = true;
            let applied = 0;
            try {
                for (const w of node.widgets || []) {
                    if (!w || !w.name || skip.has(w.name)) continue;
                    if (!(w.name in fields)) continue;
                    w.value = fields[w.name];
                    if (typeof w.callback === "function") {
                        try { w.callback(w.value); } catch (_e) { /* ignore */ }
                    }
                    applied++;
                }
            } finally {
                node.__krea2_applying = false;
            }
            pw.value = value; // keep the dropdown showing the loaded name
            snapshot(node, cfg);
            node.setDirtyCanvas(true, true);
            console.log(`[Eric_Krea2] preset '${value}': applied ${applied} field(s) to the panel.`);
        } catch (e) {
            console.error(`[Eric_Krea2] failed to apply preset '${value}':`, e);
            alert(`Could not load preset "${value}": ${e && e.message ? e.message : e}`);
        }
    };
}

// ★ Save Preset: collect current widget values (minus skip) and POST them.
function wireSave(node, cfg) {
    const skip = new Set(cfg.skip);
    const sb = node.addWidget("button", "★ Save Preset", null, async () => {
        const data = {};
        for (const w of node.widgets || []) {
            if (!w || w.type === "button") continue;
            if (!w.name || skip.has(w.name)) continue;
            if (w.value === undefined) continue;
            data[w.name] = w.value;
        }

        let name = window.prompt(`Save ${cfg.section} preset as:`, "");
        if (name === null) return;           // cancelled
        name = name.trim();
        if (!name) { alert("Preset name cannot be empty."); return; }

        try {
            const resp = await api.fetchApi("/eric_krea2/save_preset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, section: cfg.section, data }),
            });
            const j = await resp.json().catch(() => ({}));
            if (!resp.ok || !j.ok) throw new Error(j.error || `HTTP ${resp.status}`);

            const pw = findWidget(node, cfg.presetWidget);
            if (pw && pw.options && Array.isArray(pw.options.values)) {
                if (!pw.options.values.includes(j.name)) pw.options.values.push(j.name);
                pw.value = j.name;
            }
            alert(`Saved ${cfg.section} preset "${j.name}" (${j.count} field${j.count === 1 ? "" : "s"}).`);
        } catch (e) {
            alert(`Save failed: ${e && e.message ? e.message : e}`);
        }
    });
    // 2026-10-23: a button has no value to save - without this it took a slot in the
    // workflow's positional widget values and blanked the next widget appended in Python.
    if (sb) sb.serialize = false;
}

app.registerExtension({
    name: "Eric.Krea2.PresetUI",
    async nodeCreated(node) {
        const cfg = node && PRESET_NODES[node.comfyClass];
        if (!cfg) return;
        if (!node.__krea2_load_added) {
            node.__krea2_load_added = true;
            wireLoad(node, cfg);
        }
        if (!node.__krea2_save_added) {
            node.__krea2_save_added = true;
            wireSave(node, cfg);
        }
    },
});
