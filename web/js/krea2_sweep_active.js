// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - "sweep is live" outline on the Sweep Plan node (2026-10-23).
//
// While the node's `enabled` toggle is ON, a bright magenta outline with a soft glow
// is drawn around the whole node (title bar included, also when collapsed), plus a
// small "SWEEP ON" tag above it - so a queued run that will sweep is obvious at a
// glance. Pure onDrawForeground overlay: nothing is added to node.widgets, so saved
// workflows (positional widget values) are untouched. Magenta was picked to stay
// clear of the stage colours (cyan / orange / green) and the lavender groups.

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "EricKrea2SweepPlan";
const COLOR = "#ff3fd2";
const GLOW = "rgba(255, 63, 210, 0.55)";

function isOn(node) {
    const w = (node.widgets || []).find((x) => x && x.name === "enabled");
    return !!(w && w.value === true);
}

function drawActive(node, ctx) {
    if (!isOn(node) || node.mode === 4 /* bypassed */ || node.mode === 2 /* muted */) return;
    const LG = globalThis.LiteGraph || {};
    const titleH = LG.NODE_TITLE_HEIGHT || 30;
    const scale = (app.canvas && app.canvas.ds && app.canvas.ds.scale) || 1;
    const zoom = Math.min(Math.max(1 / scale, 1), 3.5);      // constant-ish screen thickness
    const collapsed = !!(node.flags && node.flags.collapsed);
    const w = collapsed ? (node._collapsed_width || LG.NODE_COLLAPSED_WIDTH || 120) : node.size[0];
    const h = collapsed ? titleH : node.size[1] + titleH;
    const pad = 4 * zoom;
    const x = -pad, y = -titleH - pad, rw = w + 2 * pad, rh = h + 2 * pad;

    ctx.save();
    ctx.strokeStyle = COLOR;
    ctx.lineWidth = 3 * zoom;
    ctx.shadowColor = GLOW;
    ctx.shadowBlur = 12 * zoom;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, rw, rh, 10);
    else ctx.rect(x, y, rw, rh);
    ctx.stroke();

    // "SWEEP ON" tag just above the top-left corner
    ctx.shadowBlur = 0;
    const fs = 11 * zoom;
    ctx.font = `bold ${fs}px sans-serif`;
    const label = "SWEEP ON";
    const tw = ctx.measureText(label).width + 10 * zoom;
    const th = fs + 6 * zoom;
    const tx = x, ty = y - th - 2 * zoom;
    ctx.fillStyle = COLOR;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(tx, ty, tw, th, 4 * zoom);
    else ctx.rect(tx, ty, tw, th);
    ctx.fill();
    ctx.fillStyle = "#1a0016";
    ctx.textBaseline = "middle";
    ctx.fillText(label, tx + 5 * zoom, ty + th / 2 + 0.5);
    ctx.restore();
}

app.registerExtension({
    name: "Eric.Krea2.SweepActive",
    async nodeCreated(node) {
        if (!node || node.comfyClass !== NODE_CLASS) return;
        if (node.__krea2_sweepActive) return;
        node.__krea2_sweepActive = true;
        const orig = node.onDrawForeground;
        node.onDrawForeground = function (ctx, canvas) {
            const r = orig ? orig.apply(this, arguments) : undefined;
            try { drawActive(this, ctx); } catch (_e) { /* never break node drawing */ }
            return r;
        };
        // redraw immediately when the toggle flips
        const w = (node.widgets || []).find((x) => x && x.name === "enabled");
        if (w) {
            const origCb = w.callback;
            w.callback = function (...a) {
                const r = origCb ? origCb.apply(this, a) : undefined;
                node.setDirtyCanvas?.(true, true);
                return r;
            };
        }
    },
});
