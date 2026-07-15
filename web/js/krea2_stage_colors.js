// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - per-stage color coding for widget-heavy nodes.
// A pure onDrawForeground overlay (nothing is added to node.widgets, so
// widgets_values serialization and saved workflows are untouched):
//   * every colored group gets a translucent color WASH over its whole block
//     (the pills are painted by litegraph before this pass, so we can't recolor
//     their fill - but a low-alpha overlay tints them, and reads as a colored
//     region even fully zoomed out),
//   * a frame + gutter bar (with a small vertical tag) around each group that
//     spans 2+ rows; single-row groups keep just a gutter tick (keeps the
//     Multi-LoRA per-slot s1/s2/s3 strengths from becoming a wall of boxes),
//   * line widths are zoom-compensated (thicker in graph units as you zoom out,
//     up to a cap) so frames stay visible at overview zoom,
//   * stage-transition fields (upscale_to_stage2 / upscale_to_stage3 /
//     s1_s2_upscale_vae) keep their bright inner focus ring in the color of the
//     stage they feed INTO,
//   * INACTIVE stages dim on the Ultra nodes (upscale_to_stage2 == 0 -> S2+S3
//     off; upscale_to_stage3 == 0 -> S3 off; S3 requires S2, mirroring
//     generate()): desaturated/darker color + "S2 OFF" tag,
//   * beneath the stage hues, two muted NEUTRAL families cover the rest of the
//     Ultra panel: general/setup (slate) and conditioning/experimental
//     (lavender). Preset dropdowns and the prompt boxes stay bare on purpose -
//     visual rest, and DOM widgets can't be tinted from the canvas anyway.
// Stage palette matches krea2_sigmas_preview.js exactly (S1 cyan, S2 orange,
// S3 green) so the whole package speaks one color language.

import { app } from "../../../scripts/app.js";

// ── palette / config (tune here) ─────────────────────────────────────────────
const STAGE_COLORS = { s1: "#4fd1e0", s2: "#f0a030", s3: "#7bd66a" };  // == sigmas preview

const NEUTRALS = {
    gen: { color: "#7e93ad", label: "GEN" },   // general / setup / routing
    cnd: { color: "#a48ae0", label: "CND" },   // conditioning / experimental
};

// Ultra-panel fields -> neutral family (applied on Ultra classes only).
const NEUTRAL_FIELDS = {
    seed: "gen", control_after_generate: "gen", seed_mode: "gen",
    crop_bottom: "gen", crop_overgen: "gen", init_match_size: "gen",
    width: "gen", height: "gen", aspect_ratio: "gen",
    turbo_guidance: "gen", upscale_vae_mode: "gen", preview_stages: "gen",
    cond_preset: "cnd", cond_rebalance: "cnd", cond_multiplier: "cnd",
    cond_layer_weights: "cnd", cond_jitter: "cnd", cond_jitter_seed: "cnd",
};

// prominence per kind (alphas)
const STYLE = {
    stage:   { wash: 0.07, frame: 0.60, bar: 0.85, tag: 1.0 },
    neutral: { wash: 0.045, frame: 0.30, bar: 0.50, tag: 0.75 },
};
const INACTIVE_FADE = 0.55;          // extra alpha pull-down for OFF stages
const MIN_FRAME_HEIGHT = 34;         // px (graph units): below this, no frame/tag

const NODE_CLASSES = new Set([
    "EricKrea2MultistageUltra",
    "EricKrea2MultistageUltraV2",
    "EricKrea2Sigmas",
    "EricKrea2MultiLoRA",
]);
// Ultra nodes: upscale_to_stageN activity semantics + neutral field families.
const ULTRA_CLASSES = new Set(["EricKrea2MultistageUltra", "EricKrea2MultistageUltraV2"]);

// Fields colored by the stage they FEED, not their name prefix, plus the
// focus-ring flag for the transition fields. Checked before the regexes
// (note s1_s2_upscale_vae would otherwise regex-match S1).
const OVERRIDES = {
    upscale_to_stage2: { key: "s2", kind: "stage", ring: true },
    upscale_to_stage3: { key: "s3", kind: "stage", ring: true },
    s1_s2_upscale_vae: { key: "s2", kind: "stage", ring: true },
};

function sectionOf(node, w) {
    const name = w && w.name;
    if (!name) return null;
    if (OVERRIDES[name]) return OVERRIDES[name];
    let m = /^s([123])_/.exec(name);                      // s1_steps, s2_cfg, s3_curve ...
    if (m) return { key: "s" + m[1], kind: "stage", ring: false };
    m = /^strength_\d+s([123])$/.exec(name);              // MultiLoRA strength_{i}s{n}
    if (m) return { key: "s" + m[1], kind: "stage", ring: false };
    if (ULTRA_CLASSES.has(node.comfyClass) && NEUTRAL_FIELDS[name]) {
        return { key: NEUTRAL_FIELDS[name], kind: "neutral", ring: false };
    }
    return null;
}

function rowHeight(w, nodeWidth) {
    try {
        if (w.computeSize) return w.computeSize(nodeWidth)[1];
    } catch (_e) { /* fall through */ }
    return (window.LiteGraph && LiteGraph.NODE_WIDGET_HEIGHT) || 20;
}

// ── stage activity (Ultra nodes) ─────────────────────────────────────────────
function widgetValue(node, name) {
    const w = (node.widgets || []).find((x) => x && x.name === name);
    return w ? w.value : undefined;
}

// { s1, s2, s3: bool }. S3 requires S2 (same rule generate() applies).
function stageActivity(node) {
    if (!ULTRA_CLASSES.has(node.comfyClass)) return { s1: true, s2: true, s3: true };
    const up2 = Number(widgetValue(node, "upscale_to_stage2"));
    const up3 = Number(widgetValue(node, "upscale_to_stage3"));
    const s2 = Number.isFinite(up2) ? up2 > 0 : true;
    const s3 = s2 && (Number.isFinite(up3) ? up3 > 0 : true);
    return { s1: true, s2, s3 };
}

// ── colors ───────────────────────────────────────────────────────────────────
// Desaturated + darkened variant of a stage color for inactive stages
// (blend toward mid-gray, then scale down).
function dimColor(hex) {
    const n = parseInt(hex.slice(1), 16);
    let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    const mix = 0.65, gray = 110, dark = 0.65;
    r = Math.round((r * (1 - mix) + gray * mix) * dark);
    g = Math.round((g * (1 - mix) + gray * mix) * dark);
    b = Math.round((b * (1 - mix) + gray * mix) * dark);
    return `rgb(${r},${g},${b})`;
}

function drawOverlay(node, ctx) {
    if (node.flags && node.flags.collapsed) return;
    const width = node.size ? node.size[0] : 0;
    if (!width || !node.widgets || !node.widgets.length) return;

    // zoom compensation: lines/bars get thicker in graph units as you zoom out
    // (constant-ish screen thickness), capped so far zoom doesn't explode.
    const scale = (app.canvas && app.canvas.ds && app.canvas.ds.scale) || 1;
    const zoom = Math.min(Math.max(1 / scale, 1), 3.5);

    // Collect visible colored rows in draw order; null entries break groups.
    const rows = [];
    for (const w of node.widgets) {
        if (!w || w.hidden || w.type === "krea2hidden") continue;   // MultiLoRA collapsed slots
        if (w.element) continue;                                    // DOM widgets sit above the canvas
        if (w.last_y == null) continue;                             // not drawn yet (first frame)
        const info = sectionOf(node, w);
        if (!info) { rows.push(null); continue; }
        const h = rowHeight(w, width);
        if (h <= 0) continue;
        rows.push({ info, y: w.last_y, h });
    }

    // Group consecutive rows of the same section.
    const groups = [];
    let cur = null;
    for (const r of rows) {
        if (!r) { cur = null; continue; }
        if (cur && cur.key === r.info.key) {
            cur.y1 = r.y + r.h;
            cur.rings.push(r.info.ring ? r : null);
        } else {
            cur = { key: r.info.key, kind: r.info.kind, y0: r.y, y1: r.y + r.h,
                    rings: [r.info.ring ? r : null] };
            groups.push(cur);
        }
    }
    if (!groups.length) return;

    const activity = stageActivity(node);

    ctx.save();
    for (const g of groups) {
        const isStage = g.kind === "stage";
        const base = isStage ? STAGE_COLORS[g.key] : (NEUTRALS[g.key] && NEUTRALS[g.key].color);
        if (!base) continue;
        const st = isStage ? STYLE.stage : STYLE.neutral;
        const active = !isStage || activity[g.key] !== false;
        const color = active ? base : dimColor(base);
        const fade = active ? 1.0 : INACTIVE_FADE;
        const tall = (g.y1 - g.y0) >= MIN_FRAME_HEIGHT;

        // color wash over the whole block (tints the pills; survives any zoom)
        const fx = 11.5, fy = g.y0 - 2.5, fw = width - 23, fh = (g.y1 - 1) - fy;
        ctx.globalAlpha = st.wash * fade;
        ctx.fillStyle = color;
        if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(fx, fy, fw, fh, 7); ctx.fill(); }
        else ctx.fillRect(fx, fy, fw, fh);

        // frame: full outline around the group (multi-row groups only)
        if (tall) {
            ctx.globalAlpha = st.frame * fade;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.2 * zoom;
            if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(fx, fy, fw, fh, 7); ctx.stroke(); }
            else ctx.strokeRect(fx, fy, fw, fh);
        }

        // gutter bar / tick along the group (left of the pills at x=15)
        ctx.globalAlpha = st.bar * fade;
        ctx.fillStyle = color;
        const barW = 5 * Math.min(zoom, 2.2);
        const barY0 = g.y0, barY1 = g.y1 - 3;
        if (ctx.roundRect) {
            ctx.beginPath(); ctx.roundRect(4, barY0, barW, barY1 - barY0, 2.5); ctx.fill();
        } else {
            ctx.fillRect(4, barY0, barW, barY1 - barY0);
        }

        // vertical tag centered on the bar (tall groups, readable zoom only)
        if (tall && scale >= 0.6) {
            ctx.save();                            // isolate the rotate/translate
            ctx.globalAlpha = st.tag * fade;
            ctx.font = "bold 9px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.translate(6.5, (barY0 + barY1) / 2);
            ctx.rotate(-Math.PI / 2);
            // knock out a bit of bar behind the letters so the tag reads
            const label = isStage
                ? (active ? g.key.toUpperCase() : g.key.toUpperCase() + " OFF")
                : NEUTRALS[g.key].label;
            const half = label.length > 3 ? 20 : 11;
            ctx.fillStyle = "#181818";
            ctx.fillRect(-half, -5, half * 2, 10);
            ctx.fillStyle = color;
            ctx.fillText(label, 0, 0.5);
            ctx.restore();
        }

        // inner focus rings on the stage-transition rows
        for (const r of g.rings) {
            if (!r) continue;
            const x = 13.5, y = r.y - 1.5, w = width - 27, h = r.h + 3;
            ctx.globalAlpha = 0.10 * fade;
            ctx.fillStyle = color;
            if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.fill(); }
            else ctx.fillRect(x, y, w, h);
            ctx.globalAlpha = 0.9 * fade;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5 * zoom;
            if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.stroke(); }
            else ctx.strokeRect(x, y, w, h);
        }
    }
    ctx.restore();
}

app.registerExtension({
    name: "Eric.Krea2.StageColors",
    async nodeCreated(node) {
        if (!node || !NODE_CLASSES.has(node.comfyClass)) return;
        if (node.__krea2_stageColors) return;
        node.__krea2_stageColors = true;

        const orig = node.onDrawForeground;
        node.onDrawForeground = function (ctx, canvas) {
            const r = orig ? orig.apply(this, arguments) : undefined;
            try { drawOverlay(this, ctx); } catch (_e) { /* never break node drawing */ }
            return r;
        };
    },
});
