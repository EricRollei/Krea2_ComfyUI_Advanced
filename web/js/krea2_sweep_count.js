// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - Sweep Plan cell counter (2026-10-15).
// Shows under the Sweep Plan node how many cells the panel's current values make, so a sweep
// never surprises you at queue time. The count comes from the server, which builds the plan
// with the SAME code the queue runs (POST /eric_krea2/sweep_count): random LoRA picks, prompt
// lines, baselines, stacks and the inert-duplicate removal are all counted exactly.
// Display only: the readout widget is not saved with the workflow.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "EricKrea2SweepPlan";
const READOUT_H = 26;

function values(node) {
    const out = {};
    for (const w of node.widgets || []) {
        if (!w || !w.name || w === node.__krea2_countWidget) continue;
        out[w.name] = w.value;
    }
    return out;
}

function signature(node) {
    try { return JSON.stringify(values(node)); } catch (e) { return String(Date.now()); }
}

async function refresh(node) {
    node.__krea2_countSig = signature(node);
    node.__krea2_countBusy = true;
    node.setDirtyCanvas(true, false);
    try {
        const resp = await api.fetchApi("/eric_krea2/sweep_count", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ values: values(node) }),
        });
        const j = await resp.json().catch(() => ({}));
        node.__krea2_count = j;
    } catch (e) {
        node.__krea2_count = { ok: false, error: e && e.message ? e.message : String(e) };
    }
    node.__krea2_countBusy = false;
    node.setDirtyCanvas(true, true);
}

function scheduleRefresh(node) {
    if (node.__krea2_countTimer) clearTimeout(node.__krea2_countTimer);
    node.__krea2_countTimer = setTimeout(() => {
        node.__krea2_countTimer = null;
        refresh(node);
    }, 400);
}

function draw(ctx, node, wWidth, y) {
    // values can change without a callback (preset applied, text typed): compare on every draw
    const sig = signature(node);
    if (node.__krea2_countSig !== undefined && node.__krea2_countSig !== sig && !node.__krea2_countTimer) {
        scheduleRefresh(node);
    }
    const pad = 8, x0 = pad, x1 = Math.max(x0 + 30, wWidth - pad);
    ctx.save();
    ctx.fillStyle = "#1c1c1c"; ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
    const h = READOUT_H - 6;
    if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x0, y + 2, x1 - x0, h, 4); ctx.fill(); ctx.stroke(); }
    else { ctx.fillRect(x0, y + 2, x1 - x0, h); ctx.strokeRect(x0, y + 2, x1 - x0, h); }
    ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const ty = y + 2 + h / 2;
    const d = node.__krea2_count;
    let text, color = "#c8c8c8";
    if (!d) {
        text = "counting cells..."; color = "#808080";
    } else if (!d.ok) {
        text = "plan error: " + (d.error || "?"); color = "#e06060";
    } else {
        text = `${d.cells} cell${d.cells === 1 ? "" : "s"}`;
        if (d.over_cap) { text += `  - over max_combos ${d.cap}: the queue will refuse it`; color = "#e06060"; }
        else if (d.cap) { text += `  (cap ${d.cap})`; }
        if (d.note) text += `  - ${d.note}`;
        if (!d.over_cap && d.cells > 0) color = "#dcdc90";
    }
    if (node.__krea2_countBusy) text += "  ...";
    // keep the text inside the box
    let t = text;
    while (t.length > 4 && ctx.measureText(t).width > x1 - x0 - 12) t = t.slice(0, -2);
    if (t !== text) t = t.slice(0, -1) + "…";
    ctx.fillStyle = color;
    ctx.fillText(t, x0 + 6, ty);
    ctx.restore();
}

// Clear button (2026-10-16): every field back to the node's default, except output_folder and
// enabled. sweep_preset goes to "custom"; preset_notes and run_notes are emptied too.
const CLEAR_KEEP = new Set(["output_folder", "enabled", "sweep_cell_count", "clear_sweep", "show_plan",
                            "control_after_generate", "share_sheets", "share_background"]);

function inputDefault(spec) {
    if (!Array.isArray(spec)) return undefined;
    const [type, opts] = spec;
    if (opts && opts.default !== undefined) return opts.default;
    if (Array.isArray(type)) return type[0];             // combo with no default: first choice
    if (type === "STRING") return "";
    if (type === "BOOLEAN") return false;
    if (type === "INT" || type === "FLOAT") return opts && opts.min !== undefined ? opts.min : 0;
    return undefined;
}

function clearSweep(node) {
    const nd = node.constructor && node.constructor.nodeData;
    const inp = (nd && nd.input) || {};
    const specs = Object.assign({}, inp.required || {}, inp.optional || {});
    if (!window.confirm("Clear the Sweep Plan? Every field goes back to its default " +
                        "(output_folder and enabled are kept; the notes are emptied).")) return;
    let n = 0;
    for (const w of node.widgets || []) {
        if (!w || !w.name || CLEAR_KEEP.has(w.name) || w.type === "button") continue;
        let v = w.name === "sweep_preset" ? "custom" : inputDefault(specs[w.name]);
        if (v === undefined || w.value === v) continue;
        w.value = v;
        n++;
    }
    node.setDirtyCanvas(true, true);
    console.log(`[Eric_Krea2] Sweep Plan cleared: ${n} field(s) reset.`);
}

// ---------------------------------------------------------------------------------------------------------------
// Show plan (2026-10-16): every cell the panel would run - settings, LoRAs and weights, and the exact words each
// cell's prompt gets - in a scrollable window, before you queue. POST /eric_krea2/sweep_preview builds the plan with
// the queue's own code; the words come from the same function Ultra V2 uses when it runs the cell.
// ---------------------------------------------------------------------------------------------------------------

const ULTRA = "EricKrea2MultistageUltraV2";
const MODEL_FILE = /\.(safetensors|gguf|sft|ckpt|pt|bin)$/i;

function ultraOf(node) {
    try {
        const outs = node.getOutputNodes ? node.getOutputNodes(0) : null;
        return (outs || []).find((n) => n && n.comfyClass === ULTRA) || null;
    } catch (e) { return null; }
}

function textOf(node, depth = 0) {
    // a prompt widget's text, following a link to the node that feeds it (one or two hops)
    if (!node || depth > 3) return "";
    const w = (node.widgets || []).find((x) => ["prompt", "text", "string", "value"].includes(x.name)
                                         && typeof x.value === "string");
    if (w && w.value) return w.value;
    for (let i = 0; i < (node.inputs || []).length; i++) {
        if (node.inputs[i].link == null) continue;
        const t = textOf(node.getInputNode(i), depth + 1);
        if (t) return t;
    }
    return "";
}

function ultraPrompt(ultra) {
    if (!ultra) return "";
    const i = (ultra.inputs || []).findIndex((x) => x.name === "prompt");
    if (i >= 0 && ultra.inputs[i].link != null) return textOf(ultra.getInputNode(i));
    const w = (ultra.widgets || []).find((x) => x.name === "prompt");
    return w && typeof w.value === "string" ? w.value : "";
}

function checkpointOf(ultra) {
    // walk up from Ultra V2's pipeline input to the loader: the first model-file widget that is not a LoRA
    if (!ultra) return "";
    const seen = new Set();
    const start = (ultra.inputs || []).findIndex((x) => x.name === "krea2_pipeline");
    let queue = start >= 0 && ultra.inputs[start].link != null ? [ultra.getInputNode(start)] : [];
    while (queue.length) {
        const n = queue.shift();
        if (!n || seen.has(n.id)) continue;
        seen.add(n.id);
        const ws = (n.widgets || []).filter((w) => typeof w.value === "string" && MODEL_FILE.test(w.value)
                                               && !/lora/i.test(w.name));
        const pick = ws.find((w) => /transformer|unet|model|checkpoint|ckpt/i.test(w.name)) || ws[0];
        if (pick) return pick.value.split(/[\\/]/).pop().replace(MODEL_FILE, "");
        for (let i = 0; i < (n.inputs || []).length; i++) {
            if (n.inputs[i].link != null) queue.push(n.getInputNode(i));
        }
    }
    return "";
}

const PLAN_CSS = `
.k2sp-back{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:10000;display:flex;align-items:flex-start;
  justify-content:center;padding-top:5vh}
.k2sp{width:min(1250px,96vw);max-height:88vh;display:flex;flex-direction:column;background:#1e1f24;color:#ddd;
  border:1px solid #444;border-radius:10px;box-shadow:0 12px 40px rgba(0,0,0,.6);font:12.5px/1.4 system-ui,Segoe UI,sans-serif}
.k2sp-top{display:flex;gap:10px;align-items:center;padding:10px 12px;border-bottom:1px solid #333;flex-wrap:wrap}
.k2sp-top b{color:#fff}
.k2sp-top input{flex:1;min-width:180px;background:#121317;color:#eee;border:1px solid #555;border-radius:6px;padding:6px 9px}
.k2sp-btn{background:#2b2d35;color:#ccc;border:1px solid #444;border-radius:6px;padding:5px 9px;cursor:pointer}
.k2sp-notes{padding:6px 12px;color:#e0af68;border-bottom:1px solid #333;white-space:pre-wrap}
.k2sp-list{overflow:auto;flex:1}
.k2sp table{border-collapse:collapse;width:100%}
.k2sp th{position:sticky;top:0;background:#26272e;text-align:left;padding:6px 8px;color:#aab;font-weight:600;z-index:1}
.k2sp td{padding:6px 8px;border-top:1px solid #2c2d33;vertical-align:top}
.k2sp td.n{color:#888;width:36px}
.k2sp td.l{width:24%}
.k2sp td.w{width:34%;color:#9ece6a;font-family:Consolas,monospace;font-size:11.5px}
.k2sp td.p{color:#a9abb6}
.k2sp .src{color:#7d8090;font-size:11px;font-family:system-ui}
.k2sp .clash{color:#f7768e;font-size:11px;font-family:system-ui}
.k2sp .set{color:#7aa2f7;font-size:11px}
.k2sp .none{color:#666;font-style:italic;font-family:system-ui}
.k2sp tr.ref td{background:#22232a}
`;

function escHtml(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function planText(d) {
    const out = [`${d.cells} cells | LoRA mode ${d.lora_mode} | words: ${d.triggers || "-"} | checkpoint: ${d.checkpoint || "unknown"}`];
    for (const r of d.rows || []) {
        out.push(`#${r.i}  ${r.settings || ""}`.trim());
        if (r.loras.length) out.push(`   LoRAs: ${r.loras.join(" | ")}`);
        if (r.words) out.push(`   words: ${r.words}`);
        out.push(`   prompt: ${r.prompt}`);
    }
    return out.join("\n");
}

async function showPlan(node) {
    if (!document.getElementById("k2sp-css")) {
        const s = document.createElement("style"); s.id = "k2sp-css"; s.textContent = PLAN_CSS; document.head.appendChild(s);
    }
    const ultra = ultraOf(node);
    const checkpoint = checkpointOf(ultra);
    const prompt = ultraPrompt(ultra);
    const back = document.createElement("div");
    back.className = "k2sp-back";
    back.innerHTML = `<div class="k2sp"><div class="k2sp-top"><b>Sweep plan</b><span class="k2sp-sum">building...</span>
        <input type="text" placeholder="filter rows (LoRA, words, setting...)"/>
        <button class="k2sp-btn k2sp-copy">copy as text</button><button class="k2sp-btn k2sp-close">close</button></div>
        <div class="k2sp-notes" style="display:none"></div><div class="k2sp-list"></div></div>`;
    document.body.appendChild(back);
    const close = () => { back.remove(); document.removeEventListener("keydown", onKey, true); };
    const onKey = (ev) => { if (ev.key === "Escape") { ev.preventDefault(); close(); } ev.stopPropagation(); };
    document.addEventListener("keydown", onKey, true);
    back.addEventListener("mousedown", (ev) => { if (ev.target === back) close(); });
    back.querySelector(".k2sp-close").onclick = close;
    let d;
    try {
        const r = await api.fetchApi("/eric_krea2/sweep_preview", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ values: values(node), checkpoint, prompt }) });
        d = await r.json();
    } catch (e) { d = { ok: false, error: String(e) }; }
    const list = back.querySelector(".k2sp-list");
    if (!d.ok) { back.querySelector(".k2sp-sum").textContent = ""; list.innerHTML = `<div style="padding:14px;color:#f7768e">plan error: ${escHtml(d.error)}</div>`; return; }
    const notes = [...(d.notes || [])];
    if (!ultra) notes.unshift("This Sweep Plan is not wired to an Ultra V2 node: no checkpoint or panel prompt known.");
    else if (!checkpoint) notes.unshift("Checkpoint not found upstream of Ultra V2: 'saved pick' shows the words picked for every checkpoint.");
    if (d.truncated) notes.push(`Showing the first ${d.rows.length} of ${d.cells} cells.`);
    if (notes.length) { const n = back.querySelector(".k2sp-notes"); n.style.display = ""; n.textContent = notes.join("\n"); }
    back.querySelector(".k2sp-sum").innerHTML = `${d.cells} cell${d.cells === 1 ? "" : "s"} &middot; LoRA mode <b>${escHtml(d.lora_mode)}</b>`
        + (d.triggers ? ` &middot; words: <b>${escHtml(d.triggers)}</b>` : "")
        + ` &middot; checkpoint: <b>${escHtml(d.checkpoint || "unknown")}</b>`;
    const rowHtml = (r) => {
        const ref = r.loras.some((x) => /reference/.test(x));
        const words = r.words ? escHtml(r.words) : `<span class="none">${r.loras.length && !ref ? "no words added" : "-"}</span>`;
        return `<tr class="${ref ? "ref" : ""}"><td class="n">${r.i}</td>
          <td class="l">${r.loras.map(escHtml).join("<br>") || '<span class="none">panel</span>'}
            ${r.settings ? `<div class="set">${escHtml(r.settings)}</div>` : ""}</td>
          <td class="w">${words}${(r.sources || []).length ? `<div class="src">${r.sources.map(escHtml).join("<br>")}</div>` : ""}
            ${(r.conflicts || []).length ? `<div class="clash">clash: ${r.conflicts.map(escHtml).join("; ")}</div>` : ""}</td>
          <td class="p">${r.prompt_swept ? "" : '<span class="src">(panel prompt) </span>'}${escHtml(r.prompt)}
            ${(r.tags || []).length ? `<div class="src">prompt LoRA tags: ${r.tags.map(escHtml).join(" ")}</div>` : ""}</td></tr>`;
    };
    const render = (q) => {
        const t = (q || "").toLowerCase();
        const rows = t ? d.rows.filter((r) => JSON.stringify(r).toLowerCase().includes(t)) : d.rows;
        list.innerHTML = `<table><thead><tr><th>#</th><th>LoRAs &middot; settings</th><th>words added to the prompt</th>
          <th>prompt (start)</th></tr></thead><tbody>${rows.map(rowHtml).join("")}</tbody></table>`
          + (rows.length ? "" : `<div style="padding:14px;color:#888">no row matches</div>`);
    };
    render("");
    const input = back.querySelector("input");
    input.addEventListener("input", () => render(input.value));
    back.querySelector(".k2sp-copy").onclick = () => navigator.clipboard?.writeText(planText(d));
}

app.registerExtension({
    name: "Eric.Krea2.SweepCount",
    async nodeCreated(node) {
        if (!node || node.comfyClass !== NODE_CLASS || node.__krea2_countAdded) return;
        node.__krea2_countAdded = true;
        const widget = {
            type: "krea2_sweepcount",
            name: "sweep_cell_count",
            value: "",
            serialize: false,
            serializeValue() { return undefined; },
            computeSize() { return [node.size ? node.size[0] : 300, READOUT_H]; },
            draw(ctx, n, widgetWidth, y) { draw(ctx, n, widgetWidth, y); },
        };
        node.widgets = node.widgets || [];
        node.widgets.push(widget);
        node.__krea2_countWidget = widget;
        // appended last and not saved, so saved widget positions never shift
        const sp = node.addWidget("button", "show_plan", null, () => showPlan(node), { serialize: false });
        if (sp) { sp.label = "Show plan (every cell, its LoRAs and words)"; sp.serialize = false; }
        const btn = node.addWidget("button", "clear_sweep", null, () => clearSweep(node), { serialize: false });
        if (btn) { btn.label = "Clear sweep (reset to defaults)"; btn.serialize = false; }
        const sz = node.computeSize();
        if (node.size[1] < sz[1]) node.setSize([node.size[0], sz[1]]);
        setTimeout(() => refresh(node), 300);
    },
});
