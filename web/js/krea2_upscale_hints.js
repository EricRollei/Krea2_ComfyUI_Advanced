// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - upscale-VAE clarity helpers for the Multi-Stage Ultra node.
// The upscale VAE silently overrides the numeric upscale_to_stage2/3 fields
// (a 2x VAE step forces 4x area regardless of what the field says), and mode
// "both" + s1_s2 can cascade to enormous final sizes with no field reflecting
// it. This adds two purely-visual aids so the graph never lies about size:
//   B) relabels the forced upscale_to_stageN fields (widget.label, non-
//      destructive) to say "(VAE 2x -> 4x area, field ignored)" etc.
//   C) draws the real S1->S2->S3->decode resolution chain under the node,
//      realized server-side by the SAME sizing math generate() runs
//      (POST /eric_krea2/resolution_chain), so what you see is what renders.
// Both only light up when an upscale_vae is actually connected.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASSES = ["EricKrea2MultistageUltra", "EricKrea2MultistageUltraV2"];
const READOUT_H = 66;
const WIDGET_KEYS = ["aspect_ratio", "s1_megapixels", "width", "height",
    "upscale_to_stage2", "upscale_to_stage3", "upscale_vae_mode", "s1_s2_upscale_vae"];
// mode -> [s2s3_vae, s2s3_down, final_vae, final_down] (mirrors _UPSCALE_VAE_MODE_FLAGS)
const MODE_FLAGS = {
    "disabled": [0, 0, 0, 0],
    "s2-s3": [1, 0, 0, 0],
    "s2-s3 with downsample": [1, 1, 0, 0],
    "final decode": [0, 0, 1, 0],
    "final decode with downsample": [0, 0, 1, 1],
    "both": [1, 0, 1, 0],
    "both with downsample": [1, 1, 1, 1],
    "both with final decode downsample": [1, 0, 1, 1],
    "inter_stage": [1, 0, 0, 0],
    "final_decode": [0, 0, 1, 0],
};

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w && w.name === name);
}
function wval(node, name) {
    const w = getWidget(node, name);
    return w ? w.value : undefined;
}

// Is the optional upscale_vae input actually wired up?
function vaeConnected(node) {
    const inp = (node.inputs || []).find((i) => i && i.name === "upscale_vae");
    return !!(inp && inp.link != null);
}

// ── B) relabel the forced numeric fields ─────────────────────────────────────
function relabelFields(node) {
    const connected = vaeConnected(node);
    const mode = String(wval(node, "upscale_vae_mode") || "disabled");
    const s1s2 = !!wval(node, "s1_s2_upscale_vae");
    const [s2s3_vae, s2s3_down] = MODE_FLAGS[mode] || [0, 0, 0, 0];
    const do_s2 = Number(wval(node, "upscale_to_stage2")) > 0;
    const do_s3 = do_s2 && Number(wval(node, "upscale_to_stage3")) > 0;

    const setLabel = (name, forced) => {
        const w = getWidget(node, name);
        if (!w) return;
        if (w.__krea2_baseLabel === undefined) w.__krea2_baseLabel = w.label || w.name;
        w.label = forced ? `${w.__krea2_baseLabel}  ${forced}` : w.__krea2_baseLabel;
    };

    setLabel("upscale_to_stage2",
        (connected && s1s2 && do_s2) ? "[VAE 2x -> 4x area, ignored]" : null);
    let s3note = null;
    if (connected && do_s3 && s2s3_vae) {
        s3note = s2s3_down ? "[VAE 2x, downsampled to this]" : "[VAE 2x -> 4x area, ignored]";
    }
    setLabel("upscale_to_stage3", s3note);
}

// ── C) resolution-chain readout ──────────────────────────────────────────────
function collectBody(node) {
    const body = { upscale_vae_connected: vaeConnected(node) };
    for (const k of WIDGET_KEYS) body[k] = wval(node, k);
    return body;
}
function signature(node) {
    return WIDGET_KEYS.map((k) => `${k}=${wval(node, k)}`).join("|") +
        `|vae=${vaeConnected(node) ? 1 : 0}`;
}

async function refresh(node) {
    node.__krea2_chainSig = signature(node);
    try {
        const resp = await api.fetchApi("/eric_krea2/resolution_chain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(collectBody(node)),
        });
        const j = await resp.json().catch(() => ({}));
        if (!resp.ok || !j.ok) throw new Error(j.error || `HTTP ${resp.status}`);
        node.__krea2_chain = j;
        node.__krea2_chainErr = null;
    } catch (e) {
        node.__krea2_chainErr = e && e.message ? e.message : String(e);
    }
    relabelFields(node);
    node.setDirtyCanvas(true, true);
}

function scheduleRefresh(node) {
    if (node.__krea2_chainTimer) clearTimeout(node.__krea2_chainTimer);
    node.__krea2_chainTimer = setTimeout(() => {
        node.__krea2_chainTimer = null;
        refresh(node);
    }, 140);
}

function drawReadout(ctx, node, wWidth, y) {
    const pad = 8, x0 = pad, x1 = Math.max(x0 + 30, wWidth - pad);
    ctx.save();

    // auto-refresh when any relevant widget or the VAE link changes
    const sig = signature(node);
    if (node.__krea2_chainSig !== undefined && node.__krea2_chainSig !== sig) scheduleRefresh(node);

    ctx.fillStyle = "#1c1c1c"; ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
    const h = READOUT_H - 6;
    if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x0, y + 2, x1 - x0, h, 4); ctx.fill(); ctx.stroke(); }
    else { ctx.fillRect(x0, y + 2, x1 - x0, h); ctx.strokeRect(x0, y + 2, x1 - x0, h); }

    ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";

    if (node.__krea2_chainErr) {
        ctx.fillStyle = "#e06060";
        ctx.fillText("resolution preview error: " + node.__krea2_chainErr, x0 + 6, y + 20);
        ctx.restore();
        return;
    }
    const d = node.__krea2_chain;
    if (!d || !d.chain) {
        ctx.fillStyle = "#808080";
        ctx.fillText("computing resolution chain...", x0 + 6, y + 20);
        ctx.restore();
        return;
    }

    ctx.fillStyle = "#c8c8c8";
    ctx.fillText("resolution chain" + (d.connected ? "" : "  (upscale_vae not connected)"),
        x0 + 6, y + 15);

    let ty = y + 30;
    for (const st of d.chain) {
        ctx.fillStyle = "#dcdc90";
        ctx.fillText(`${st.stage} ${st.w}x${st.h} (${st.mp} MP)`, x0 + 6, ty);
        ctx.fillStyle = "#8a8a8a";
        ctx.fillText(st.via, x0 + 168, ty);
        ty += 12;
        if (ty > y + READOUT_H - 6) break;
    }
    if (d.decode && ty <= y + READOUT_H - 4) {
        ctx.fillStyle = "#90c8dc";
        ctx.fillText(`decode ${d.decode.w}x${d.decode.h} (${d.decode.mp} MP)`, x0 + 6, ty);
        ctx.fillStyle = "#8a8a8a";
        ctx.fillText(d.decode.via, x0 + 168, ty);
    }
    ctx.restore();
}

function addReadoutWidget(node) {
    const widget = {
        type: "krea2_reschain",
        name: "reschain",
        value: "",
        serializeValue() { return undefined; },
        computeSize() { return [node.size ? node.size[0] : 300, READOUT_H]; },
        draw(ctx, n, widgetWidth, y) { drawReadout(ctx, n, widgetWidth, y); },
    };
    node.widgets = node.widgets || [];
    node.widgets.push(widget);
    node.__krea2_reschain = widget;
}

// Refresh on every widget edit (incl. preset application, which sets many at once).
function hookWidgets(node) {
    for (const w of node.widgets || []) {
        if (!w || !w.name || w === node.__krea2_reschain) continue;
        if (w.__krea2_hintHooked) continue;
        w.__krea2_hintHooked = true;
        const orig = w.callback;
        w.callback = function (v, ...a) {
            const r = orig ? orig.call(this, v, ...a) : undefined;
            scheduleRefresh(node);
            return r;
        };
    }
}

app.registerExtension({
    name: "Eric.Krea2.UpscaleHints",
    async nodeCreated(node) {
        if (!node || !NODE_CLASSES.includes(node.comfyClass)) return;
        if (node.__krea2_hintsAdded) return;
        node.__krea2_hintsAdded = true;

        addReadoutWidget(node);
        hookWidgets(node);

        // refresh when the upscale_vae input is (dis)connected too
        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
            const r = origConn ? origConn.apply(this, args) : undefined;
            scheduleRefresh(node);
            return r;
        };

        const sz = node.computeSize();
        if (node.size[1] < sz[1]) node.setSize([node.size[0], sz[1]]);
        setTimeout(() => refresh(node), 30);
    },
});
