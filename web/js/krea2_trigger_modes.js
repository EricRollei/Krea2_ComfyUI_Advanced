// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - plain trigger-word choices (2026-10-16).
// The trigger dropdowns (Apply LoRA / Multi-LoRA Stack add_triggers, Sweep Plan lora_triggers) show four choices:
//   none | trigger | full | saved pick
// The Python lists still hold the older names (off, catalog, recipe, prepend, append) so saved workflows validate;
// here off / catalog / recipe are turned into none / trigger / saved pick when a workflow or preset loads them
// (same meaning), and prepend / append stay visible only on a node that already uses them.

import { app } from "../../../scripts/app.js";

const TARGETS = { EricKrea2ApplyLoRA: "add_triggers", EricKrea2MultiLoRA: "add_triggers",
                  EricKrea2SweepPlan: "lora_triggers" };
const CHOICES = ["none", "trigger", "full", "saved pick"];
const RENAMED = { off: "none", catalog: "trigger", recipe: "saved pick" };
const LEGACY = ["prepend", "append"];

function tidy(w) {
    if (!w) return;
    if (RENAMED[w.value]) w.value = RENAMED[w.value];
    const keep = LEGACY.includes(w.value) ? [w.value] : [];
    if (w.options) w.options.values = [...CHOICES, ...keep];
}

app.registerExtension({
    name: "Eric.Krea2.TriggerModes",
    async nodeCreated(node) {
        const name = TARGETS[node?.comfyClass];
        if (!name || node.__k2tm) return;
        node.__k2tm = true;
        const w = (node.widgets || []).find((x) => x.name === name);
        if (!w) return;
        tidy(w);
        const orig = w.callback;
        w.callback = function (v, ...a) {
            if (RENAMED[v]) { v = RENAMED[v]; w.value = v; }      // a preset carrying an older name
            const r = orig ? orig.call(this, v, ...a) : undefined;
            tidy(w);
            return r;
        };
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            setTimeout(() => { tidy(w); node.setDirtyCanvas(true, true); }, 0);
            return r;
        };
    },
});
