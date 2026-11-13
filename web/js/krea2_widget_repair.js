// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - repair blank widget values on workflow load (2026-10-23).
//
// ComfyUI restores widget values by POSITION. Workflows saved before 2026-10-23
// carry trailing entries for JS-added widgets (the Save Preset button, the
// resolution read-out). When a Python widget is appended at the end of a node,
// it lands on one of those trailing slots and loads a blank ("" / null) - e.g.
// s1_sde_hi / s1_sde_lo on Ultra V2 ("could not convert string to float: ''").
//
// After a node is configured from a saved workflow, any NUMBER, BOOLEAN or COMBO
// widget whose value is blank (""/null/undefined) or a non-numeric value in a
// number field is reset to the value the widget had when the node was created
// (= its Python default), with a console note. Valid values are never touched;
// a combo holding a real-but-missing name (a deleted LoRA, preset...) is left
// alone so ComfyUI's validation still reports it. STRING widgets are skipped
// (blank is a legitimate value there).

import { app } from "../../../scripts/app.js";

function isBlank(v) {
    return v === "" || v === null || v === undefined;
}

function snapshotDefaults(node) {
    const d = {};
    for (const w of node.widgets || []) {
        if (!w || !w.name || w.type === "button") continue;
        d[w.name] = w.value;
    }
    return d;
}

function repair(node) {
    const defs = node.__krea2_widget_defaults;
    if (!defs) return;
    const fixed = [];
    for (const w of node.widgets || []) {
        if (!w || !w.name || !(w.name in defs)) continue;
        const dv = defs[w.name];
        let bad = false;
        if (typeof dv === "number") {
            bad = isBlank(w.value) || typeof w.value !== "number" || !Number.isFinite(w.value);
            if (bad && typeof w.value === "string" && w.value.trim() !== "" && Number.isFinite(Number(w.value))) {
                w.value = Number(w.value);          // numeric string: keep the number
                fixed.push(`${w.name}=${w.value} (from text)`);
                continue;
            }
        } else if (typeof dv === "boolean") {
            bad = typeof w.value !== "boolean" && isBlank(w.value);
        } else if (w.type === "combo") {
            bad = isBlank(w.value);
        }
        if (bad) {
            w.value = dv;
            fixed.push(`${w.name}=${dv}`);
        }
    }
    if (fixed.length) {
        console.warn(`[Eric_Krea2] ${node.comfyClass} #${node.id}: restored blank widget value(s) ` +
                     `to defaults: ${fixed.join(", ")} - re-save the workflow to make it permanent.`);
        node.setDirtyCanvas?.(true, true);
    }
}

app.registerExtension({
    name: "Eric.Krea2.WidgetRepair",
    async nodeCreated(node) {
        if (!node || !String(node.comfyClass || "").startsWith("EricKrea2")) return;
        if (node.__krea2_repair_hooked) return;
        node.__krea2_repair_hooked = true;
        // nodeCreated runs before a saved workflow configures the node, so these are the defaults.
        node.__krea2_widget_defaults = snapshotDefaults(node);
        const orig = node.onConfigure;
        node.onConfigure = function () {
            const r = orig ? orig.apply(this, arguments) : undefined;
            try { repair(this); } catch (e) { console.error("[Eric_Krea2] widget repair failed:", e); }
            return r;
        };
    },
});
