// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - live sigma-curve preview for the "Eric Krea2 Sigmas" node.
// Draws the three per-stage schedules (S1/S2/S3) right on the node so the
// abstract knobs (curve / detail_bias / rho / alpha / beta) become a picture:
// x = step index (0 -> last), y = sigma (1 at top = full noise, 0 = clean).
// The curves are realized server-side by the SAME _sigmas.build_sigmas the Ultra
// node samples with (POST /eric_krea2/preview_sigmas), so the plot never lies.
// Disabled stages are drawn dimmed + dashed (they fall back to the Ultra panel).

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "EricKrea2Sigmas";
const STAGES = ["s1", "s2", "s3"];
const STAGE_COLORS = { s1: "#4fd1e0", s2: "#f0a030", s3: "#7bd66a" };
const HEADER_H = 16;         // header strip above the three rows
const ROW_H = 76;            // per-stage mini-plot height
const PLOT_H = HEADER_H + 3 * ROW_H;
const DEFAULT_STEPS = 24;    // fallback when no Ultra node is connected

// ── read the connected Ultra node's real per-stage run values ────────────────
// The KREA2_SIGMAS output feeds a Multi-Stage Ultra node; walk that link and read
// its sN_steps / sN_start_step / sN_end_step widgets so the plot shows exactly the
// portion of each schedule that will actually sample (no second source of truth).
function findDownstream(node) {
    const out = node.outputs && node.outputs[0];
    if (!out || !out.links || !out.links.length) return null;
    const graph = node.graph || app.graph;
    if (!graph || !graph.links) return null;
    for (const linkId of out.links) {
        const link = graph.links[linkId];
        if (!link) continue;
        const tgt = graph.getNodeById(link.target_id);
        if (tgt && tgt.widgets && tgt.widgets.some((w) => w && /^s[123]_steps$/.test(w.name || "")))
            return tgt;
    }
    return null;
}

function readRun(dn, s) {
    const def = { steps: DEFAULT_STEPS, start: 0, end: DEFAULT_STEPS, connected: false };
    if (!dn || !dn.widgets) return def;
    const val = (name) => {
        const w = dn.widgets.find((w) => w && w.name === name);
        return w ? w.value : undefined;
    };
    let steps = parseInt(val(`${s}_steps`), 10);
    if (!Number.isFinite(steps) || steps < 1) return def;
    let start = parseInt(val(`${s}_start_step`), 10);
    let end = parseInt(val(`${s}_end_step`), 10);
    if (!Number.isFinite(start)) start = 0;
    if (!Number.isFinite(end)) end = steps;
    if (end <= 0) end = steps;                         // s1 "run to end" sentinel
    start = Math.max(0, Math.min(steps - 1, start));
    end = Math.max(start + 1, Math.min(steps, end));
    return { steps, start, end, connected: true };
}

function runSignature(node) {
    const dn = findDownstream(node);
    return STAGES.map((s) => {
        const r = readRun(dn, s);
        return `${r.steps}:${r.start}:${r.end}:${r.connected ? 1 : 0}`;
    }).join("|");
}

function collectBody(node) {
    const body = { steps: DEFAULT_STEPS };
    for (const w of node.widgets || []) {
        if (!w || !w.name) continue;
        if (STAGES.some((s) => w.name.startsWith(s + "_"))) body[w.name] = w.value;
    }
    const dn = findDownstream(node);
    node.__krea2_dnConnected = !!dn;
    for (const s of STAGES) {
        const r = readRun(dn, s);
        body[`${s}_steps`] = r.steps;
        body[`${s}_start`] = r.start;
        body[`${s}_end`] = r.end;
    }
    return body;
}

async function refresh(node) {
    node.__krea2_runSig = runSignature(node);   // remember what we fetched with
    try {
        const resp = await api.fetchApi("/eric_krea2/preview_sigmas", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(collectBody(node)),
        });
        const j = await resp.json().catch(() => ({}));
        if (!resp.ok || !j.ok) throw new Error(j.error || `HTTP ${resp.status}`);
        node.__krea2_sig = j;
        node.__krea2_sigErr = null;
    } catch (e) {
        node.__krea2_sigErr = e && e.message ? e.message : String(e);
    }
    node.setDirtyCanvas(true, true);
}

// Debounced refresh so dragging a slider doesn't spam the endpoint.
function scheduleRefresh(node) {
    if (node.__krea2_sigTimer) clearTimeout(node.__krea2_sigTimer);
    node.__krea2_sigTimer = setTimeout(() => { node.__krea2_sigTimer = null; refresh(node); }, 140);
}

// ── one stage's mini-plot ────────────────────────────────────────────────────
function drawRow(ctx, x0, x1, yTop, yBot, s, st) {
    const color = STAGE_COLORS[s];
    const w = x1 - x0, h = yBot - yTop;

    ctx.fillStyle = "#1c1c1c";
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x0, yTop, w, h, 4); ctx.fill(); ctx.stroke(); }
    else { ctx.fillRect(x0, yTop, w, h); ctx.strokeRect(x0, yTop, w, h); }

    // midline grid (sigma 0.5)
    ctx.strokeStyle = "#2a2a2a";
    ctx.beginPath(); ctx.moveTo(x0, yBot - 0.5 * h); ctx.lineTo(x1, yBot - 0.5 * h); ctx.stroke();

    if (!st || !st.sigmas || !st.sigmas.length) {
        ctx.fillStyle = "#777"; ctx.font = "9px sans-serif";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(`${s.toUpperCase()}: no data`, x0 + 6, (yTop + yBot) / 2);
        return;
    }

    const steps = st.steps || st.sigmas.length;
    const pts = st.sigmas.concat([0]);              // append terminal sigma 0 (clean)
    const N = pts.length;                            // steps + 1
    const start = Math.max(0, Math.min(N - 1, st.start ?? 0));
    const end = Math.max(start + 1, Math.min(N - 1, st.end ?? steps));
    const px = (i) => (N > 1 ? x0 + (i / (N - 1)) * w : x0 + w / 2);
    const py = (v) => yBot - Math.max(0, Math.min(1, v)) * h;
    const dim = st.enabled ? 1.0 : 0.4;

    // ghost = full schedule (the steps that DON'T run this stage)
    ctx.globalAlpha = 0.30 * dim;
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let i = 0; i < N; i++) { const X = px(i), Y = py(pts[i]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke(); ctx.setLineDash([]);

    // solid = the [start, end] window that actually samples
    ctx.globalAlpha = dim;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = start; i <= end; i++) { const X = px(i), Y = py(pts[i]); i === start ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y); }
    ctx.stroke();
    ctx.fillStyle = color;
    for (let i = start; i <= end; i++) { ctx.beginPath(); ctx.arc(px(i), py(pts[i]), 1.8, 0, Math.PI * 2); ctx.fill(); }
    // hollow ring on the start step = the re-noise / injection level
    ctx.strokeStyle = color; ctx.lineWidth = 1.25;
    ctx.beginPath(); ctx.arc(px(start), py(pts[start]), 3.4, 0, Math.PI * 2); ctx.stroke();

    ctx.globalAlpha = 1.0;
    ctx.font = "9px sans-serif"; ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
    ctx.fillStyle = st.enabled ? "#cfcfcf" : "#8a8a8a";
    const tag = st.enabled ? "" : "  (panel)";
    ctx.fillText(
        `${s.toUpperCase()}  ${st.curve}  steps ${start}\u2192${end}/${steps}` +
        `  \u03c3 ${pts[start].toFixed(2)}\u2192${pts[end].toFixed(2)}${tag}`,
        x0 + 5, yTop + 10);
}

function drawPlot(ctx, node, wWidth, y) {
    const pad = 8;
    const x0 = pad, x1 = Math.max(x0 + 30, wWidth - pad);
    ctx.save();

    // auto-refresh when the connected Ultra node's steps/window change
    const sig = runSignature(node);
    if (node.__krea2_runSig !== undefined && node.__krea2_runSig !== sig) scheduleRefresh(node);

    ctx.fillStyle = "#c8c8c8"; ctx.font = "10px sans-serif";
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillText(node.__krea2_dnConnected
        ? "sigma per stage  (solid = steps that run \u00b7 ring = re-noise level \u00b7 ghost = full schedule)"
        : "sigma per stage  (connect the output to an Ultra node to show the real steps)",
        x0, y + 12);

    if (node.__krea2_sigErr) {
        ctx.fillStyle = "#e06060"; ctx.font = "9px sans-serif";
        ctx.fillText("preview error: " + node.__krea2_sigErr, x0, y + 28);
        ctx.restore();
        return;
    }
    const data = node.__krea2_sig;
    if (!data || !data.stages) {
        ctx.fillStyle = "#808080"; ctx.textAlign = "center";
        ctx.fillText("computing preview...", (x0 + x1) / 2, y + PLOT_H / 2);
        ctx.restore();
        return;
    }

    let rowTop = y + HEADER_H;
    for (const s of STAGES) {
        drawRow(ctx, x0, x1, rowTop + 3, rowTop + ROW_H - 5, s, data.stages[s]);
        rowTop += ROW_H;
    }
    ctx.restore();
}

function addPlotWidget(node) {
    const widget = {
        type: "krea2_sigplot",
        name: "sigplot",
        value: "",
        serializeValue() { return undefined; },     // purely a viewer
        computeSize() { return [node.size ? node.size[0] : 300, PLOT_H]; },
        draw(ctx, n, widgetWidth, y) { drawPlot(ctx, n, widgetWidth, y); },
    };
    node.widgets = node.widgets || [];
    node.widgets.push(widget);
    node.__krea2_plot = widget;
}

// Chain a refresh onto every recipe widget's callback so the plot tracks edits,
// including when a preset is applied (which sets many widgets at once).
function hookWidgets(node) {
    for (const w of node.widgets || []) {
        if (!w || !w.name || w === node.__krea2_plot) continue;
        if (w.__krea2_plotHooked) continue;
        w.__krea2_plotHooked = true;
        const orig = w.callback;
        w.callback = function (v, ...a) {
            const r = orig ? orig.call(this, v, ...a) : undefined;
            scheduleRefresh(node);
            return r;
        };
    }
}

app.registerExtension({
    name: "Eric.Krea2.SigmasPreview",
    async nodeCreated(node) {
        if (!node || node.comfyClass !== NODE_CLASS) return;
        if (node.__krea2_plotAdded) return;
        node.__krea2_plotAdded = true;

        addPlotWidget(node);
        hookWidgets(node);

        // grow the node so the plot has room, then fetch the initial curves
        const sz = node.computeSize();
        if (node.size[1] < sz[1]) node.setSize([node.size[0], sz[1]]);
        setTimeout(() => refresh(node), 30);
    },
});
