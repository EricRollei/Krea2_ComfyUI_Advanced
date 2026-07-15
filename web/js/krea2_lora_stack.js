// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - growable Multi-LoRA Stack node.
// The Python node declares MAX_SLOTS rows (lora_i / strength_i) so serialization
// and the native LoRA dropdowns are 100% standard. This extension just HIDES the
// rows past the last used one + 1, so the node shows only the LoRAs you use plus
// one empty "next" slot. Pick a LoRA in the last visible row -> the next appears;
// clear it -> the tail collapses.

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "EricKrea2MultiLoRA";
const MAX = 10;                 // must match EricKrea2MultiLoRA.MAX_SLOTS
const HIDDEN_TYPE = "krea2hidden";

// Collect the (enable, lora, per-stage strengths) widget groups in slot order.
function rowPairs(node) {
    const out = [];
    for (let i = 1; i <= MAX; i++) {
        const ow = (node.widgets || []).find((w) => w.name === `on_${i}`);
        const lw = (node.widgets || []).find((w) => w.name === `lora_${i}`);
        const sws = ["s1", "s2", "s3"]
            .map((s) => (node.widgets || []).find((w) => w.name === `strength_${i}${s}`))
            .filter(Boolean);
        if (lw && sws.length) out.push({ i, ow, lw, sws });
    }
    return out;
}

function hideWidget(w) {
    if (w.__k2hidden) return;
    w.__k2hidden = true;
    w.__k2type = w.type;
    w.__k2cs = w.computeSize;
    w.type = HIDDEN_TYPE;             // unknown type -> litegraph draws nothing
    w.computeSize = () => [0, -4];    // ...and occupies zero vertical space
    w.hidden = true;                  // honored by newer litegraph as an extra guard
}

function showWidget(w) {
    if (!w.__k2hidden) return;
    w.__k2hidden = false;
    w.type = w.__k2type;
    w.computeSize = w.__k2cs;
    w.hidden = false;
}

function relayout(node) {
    const ps = rowPairs(node);
    let lastUsed = 0;
    for (const p of ps) {
        if (p.lw.value && p.lw.value !== "none") lastUsed = p.i;
    }
    const visible = Math.min(MAX, Math.max(1, lastUsed + 1));
    for (const p of ps) {
        if (p.i <= visible) { if (p.ow) showWidget(p.ow); showWidget(p.lw); p.sws.forEach(showWidget); }
        else { if (p.ow) hideWidget(p.ow); hideWidget(p.lw); p.sws.forEach(hideWidget); }
    }
    // Re-fit height, keep the current width.
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Eric.Krea2.MultiLoRA",
    async nodeCreated(node) {
        if (!node || node.comfyClass !== NODE_CLASS) return;
        if (node.__k2ml_init) return;
        node.__k2ml_init = true;

        // Re-evaluate visibility whenever a LoRA slot changes.
        for (const p of rowPairs(node)) {
            const orig = p.lw.callback;
            p.lw.callback = function (v, ...a) {
                const r = orig ? orig.call(this, v, ...a) : undefined;
                relayout(node);
                return r;
            };
        }

        // Re-fit after a saved graph loads its widget values, and once now.
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            setTimeout(() => relayout(node), 0);
            return r;
        };
        setTimeout(() => relayout(node), 0);
    },
});
