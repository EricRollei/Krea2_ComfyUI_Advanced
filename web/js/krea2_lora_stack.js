// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - growable Multi-LoRA Stack node + LoRA search popup.
//
// 1) Growable rows: the Python node declares MAX_SLOTS rows (lora_i / strength_i)
//    so serialization and the native LoRA dropdowns are 100% standard. This
//    extension just HIDES the rows past the last used one + 1.
//
// 2) Search (2026-10-07): clicking the MIDDLE of a LoRA dropdown opens a search
//    popup instead of the plain 500-item list. It searches filenames, folders,
//    trigger words AND LoRA Catalog cards (name, role, effect text, artists,
//    depicts, facet tags), from GET /eric_krea2/lora_search_index.
//      - the dropdown's arrow zones still step prev/next
//      - Shift+click opens the native list
//      - right-click the node -> "Search LoRA ..." per slot (fallback)
//    NO widgets are added: the popup only sets the existing combo's value, so
//    saved workflows (positional widgets_values) are untouched.
//    Applies to the Multi-LoRA Stack, Apply LoRA and Diagnose LoRA nodes.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "EricKrea2MultiLoRA";
const SEARCH_NODES = { EricKrea2MultiLoRA: /^lora_\d+$/, EricKrea2ApplyLoRA: /^lora_name$/,
                       EricKrea2DiagnoseLoRA: /^lora_name$/ };
const MAX = 10;                 // must match EricKrea2MultiLoRA.MAX_SLOTS
const HIDDEN_TYPE = "krea2hidden";

// -------------------------------------------------------------------------
// growable rows (unchanged behaviour)
// -------------------------------------------------------------------------

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
    w.type = HIDDEN_TYPE;
    w.computeSize = () => [0, -4];
    w.hidden = true;
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
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas(true, true);
}

// -------------------------------------------------------------------------
// search index (fetched once per session; refresh button in the popup)
// -------------------------------------------------------------------------

let INDEX = null;
let INDEX_META = {};

async function loadIndex(force = false) {
    if (INDEX && !force) return INDEX;
    const r = await api.fetchApi(`/eric_krea2/lora_search_index${force ? `?t=${Date.now()}` : ""}`);
    const j = await r.json();
    if (j.error) console.warn("[EricKrea2] LoRA search index:", j.error);
    INDEX = (j.entries || []).map((e) => {
        const fields = {
            name: (e.n || "").toLowerCase(),
            path: (e.v || "").toLowerCase().replace(/\\/g, "/"),
            trig: (e.t || []).join(" | ").toLowerCase(),
            artist: (e.artists || []).join(" | ").toLowerCase(),
            role: `${e.role || ""} ${e.kind || ""}`.toLowerCase(),
            tags: (e.tags || []).join(" | ").toLowerCase(),
            effect: `${e.effect || ""} ${(e.depicts || []).join(" ")}`.toLowerCase(),
        };
        return { ...e, _f: fields };
    });
    INDEX_META = { catalog: j.catalog, carded: j.carded || 0, note: j.note || j.error || "" };
    return INDEX;
}

// AND across query terms; each term must hit some field. Field weights favour
// names/paths/triggers over free text. role:xxx restricts the role/kind field.
const WEIGHTS = { name: 6, path: 4, trig: 4, artist: 4, role: 3, tags: 2, effect: 1 };

function scoreEntry(e, terms) {
    let total = 0;
    for (const t of terms) {
        let best = 0;
        if (t.field) {
            if (e._f[t.field].includes(t.text)) best = WEIGHTS[t.field] + 2;
        } else {
            for (const [f, w] of Object.entries(WEIGHTS)) {
                const s = e._f[f];
                const i = s.indexOf(t.text);
                if (i < 0) continue;
                const wordStart = i === 0 || /[\s/_\-|.,(]/.test(s[i - 1]);
                best = Math.max(best, w + (wordStart ? 1 : 0));
            }
        }
        if (!best) return -1;
        total += best;
    }
    return total + (e.fav ? 1.5 : 0) + (e.rating || 0) * 0.4;
}

function parseQuery(q) {
    const terms = [];
    for (const raw of q.toLowerCase().split(/\s+/).filter(Boolean)) {
        const m = raw.match(/^(role|artist|tag|trig):(.+)$/);
        if (m) {
            const field = { role: "role", artist: "artist", tag: "tags", trig: "trig" }[m[1]];
            terms.push({ field, text: m[2] });
        } else terms.push({ text: raw });
    }
    return terms;
}

// -------------------------------------------------------------------------
// popup
// -------------------------------------------------------------------------

const CSS = `
.k2ls-back{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:10000;display:flex;
  align-items:flex-start;justify-content:center;padding-top:8vh}
.k2ls{width:min(860px,94vw);max-height:78vh;display:flex;flex-direction:column;background:#1e1f24;
  color:#ddd;border:1px solid #444;border-radius:10px;box-shadow:0 12px 40px rgba(0,0,0,.6);
  font:13px/1.35 system-ui,Segoe UI,sans-serif}
.k2ls-top{display:flex;gap:8px;padding:10px;border-bottom:1px solid #333;align-items:center}
.k2ls-top input{flex:1;background:#121317;color:#eee;border:1px solid #555;border-radius:6px;
  padding:8px 10px;font-size:14px;outline:none}
.k2ls-top input:focus{border-color:#7aa2f7}
.k2ls-btn{background:#2b2d35;color:#ccc;border:1px solid #444;border-radius:6px;padding:6px 9px;
  cursor:pointer;white-space:nowrap}
.k2ls-btn.on{background:#3d4b73;color:#fff;border-color:#7aa2f7}
.k2ls-chips{display:flex;flex-wrap:wrap;gap:5px;padding:6px 10px;border-bottom:1px solid #333}
.k2ls-chip{font-size:11px;padding:2px 8px;border-radius:10px;background:#2b2d35;border:1px solid #444;
  cursor:pointer;color:#bbb}
.k2ls-chip.on{background:#3d4b73;color:#fff;border-color:#7aa2f7}
.k2ls-list{overflow:auto;flex:1}
.k2ls-row{padding:7px 12px;border-bottom:1px solid #2a2b31;cursor:pointer}
.k2ls-row.sel{background:#2c3550}
.k2ls-row.cur{box-shadow:inset 3px 0 0 #9ece6a}
.k2ls-l1{display:flex;gap:8px;align-items:baseline}
.k2ls-name{font-weight:600;color:#fff}
.k2ls-name mark{background:#5b4a1a;color:#ffd479;border-radius:2px}
.k2ls-role{font-size:10.5px;padding:1px 6px;border-radius:8px;background:#34394a;color:#aab8e8}
.k2ls-stars{color:#e0af68;font-size:11px;letter-spacing:1px}
.k2ls-dir{color:#7d8090;font-size:11px;margin-left:auto;max-width:45%;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;direction:rtl;text-align:left}
.k2ls-trig{font:11px Consolas,monospace;color:#9ece6a;margin-top:2px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.k2ls-eff{color:#a9abb6;font-size:12px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.k2ls-foot{padding:6px 12px;color:#7d8090;font-size:11px;border-top:1px solid #333;display:flex;gap:12px}
`;

function ensureCss() {
    if (document.getElementById("k2ls-css")) return;
    const s = document.createElement("style");
    s.id = "k2ls-css";
    s.textContent = CSS;
    document.head.appendChild(s);
}

function esc(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function highlight(text, terms) {
    let h = esc(text);
    for (const t of terms) {
        if (t.field || t.text.length < 2) continue;
        const re = new RegExp(t.text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig");
        h = h.replace(re, (m) => `<mark>${m}</mark>`);
    }
    return h;
}

function setComboValue(widget, node, value) {
    const canvas = app.canvas;
    if (typeof widget.setValue === "function") {
        widget.setValue(value, { e: null, node, canvas });
    } else {
        widget.value = value;
        widget.callback?.(value, canvas, node);
    }
    node.setDirtyCanvas(true, true);
}

async function openSearch(node, widget) {
    ensureCss();
    const back = document.createElement("div");
    back.className = "k2ls-back";
    back.innerHTML = `
      <div class="k2ls" role="dialog">
        <div class="k2ls-top">
          <input type="text" placeholder="Search name, folder, trigger, artist, tag, effect...  (role:style  artist:vallejo  tag:watercolor)"/>
          <button class="k2ls-btn k2ls-fav" title="Favourites only">★ fav</button>
          <button class="k2ls-btn k2ls-none" title="Clear this slot">none</button>
          <button class="k2ls-btn k2ls-ref" title="Rebuild the index (new files / edited cards)">↻</button>
        </div>
        <div class="k2ls-chips"></div>
        <div class="k2ls-list"><div style="padding:14px;color:#888">loading index...</div></div>
        <div class="k2ls-foot"><span class="k2ls-count"></span><span class="k2ls-meta"></span>
          <span style="margin-left:auto">↑↓ move · Enter pick · Esc close · Shift+click dropdown = plain list</span></div>
      </div>`;
    document.body.appendChild(back);
    const input = back.querySelector("input");
    const list = back.querySelector(".k2ls-list");
    const chipsEl = back.querySelector(".k2ls-chips");
    const favBtn = back.querySelector(".k2ls-fav");
    let favOnly = false;
    const roles = new Set();
    let results = [];
    let sel = 0;

    const close = () => { back.remove(); document.removeEventListener("keydown", onKey, true); };
    const pick = (v) => { setComboValue(widget, node, v); close(); };

    function render() {
        const terms = parseQuery(input.value.trim());
        let pool = INDEX || [];
        if (favOnly) pool = pool.filter((e) => e.fav);
        if (roles.size) pool = pool.filter((e) => roles.has(e.role || "(no card)"));
        if (terms.length) {
            results = pool.map((e) => [scoreEntry(e, terms), e]).filter((x) => x[0] >= 0)
                .sort((a, b) => b[0] - a[0] || a[1].n.localeCompare(b[1].n)).map((x) => x[1]);
        } else {
            results = [...pool].sort((a, b) => (b.v === widget.value) - (a.v === widget.value)
                || (b.fav - a.fav) || ((b.rating || 0) - (a.rating || 0)) || a.n.localeCompare(b.n));
        }
        sel = Math.min(sel, Math.max(0, results.length - 1));
        const shown = results.slice(0, 250);
        list.innerHTML = shown.map((e, i) => {
            const stars = e.rating ? "★".repeat(e.rating) : "";
            const role = e.role ? `<span class="k2ls-role">${esc(e.role)}</span>` : "";
            const trig = (e.t || []).length ? `<div class="k2ls-trig">${highlight(e.t.join("  ·  "), terms)}</div>` : "";
            const effTxt = [e.effect, (e.artists || []).length ? "artists: " + e.artists.join(", ") : "",
                            (e.tags || []).slice(0, 10).join(", ")].filter(Boolean).join("  —  ");
            const eff = effTxt ? `<div class="k2ls-eff" title="${esc(effTxt)}">${highlight(effTxt, terms)}</div>` : "";
            return `<div class="k2ls-row${i === sel ? " sel" : ""}${e.v === widget.value ? " cur" : ""}" data-i="${i}">
                <div class="k2ls-l1">${e.fav ? "♥" : ""}<span class="k2ls-name">${highlight(e.n, terms)}</span>
                ${role}<span class="k2ls-stars">${stars}</span><span class="k2ls-dir" title="${esc(e.v)}">${esc(e.d)}</span></div>
                ${trig}${eff}</div>`;
        }).join("") || `<div style="padding:14px;color:#888">no match</div>`;
        back.querySelector(".k2ls-count").textContent =
            `${results.length} match${results.length === 1 ? "" : "es"}${results.length > 250 ? " (first 250 shown)" : ""}`;
        const sr = list.querySelector(".k2ls-row.sel");
        sr?.scrollIntoView({ block: "nearest" });
    }

    function renderChips() {
        const counts = {};
        for (const e of INDEX || []) { const r = e.role || "(no card)"; counts[r] = (counts[r] || 0) + 1; }
        chipsEl.innerHTML = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([r, n]) =>
            `<span class="k2ls-chip${roles.has(r) ? " on" : ""}" data-r="${esc(r)}">${esc(r)} ${n}</span>`).join("");
        const m = INDEX_META;
        back.querySelector(".k2ls-meta").textContent = m.catalog
            ? `${m.carded} with catalog cards` : (m.note || "no LoRA Catalog found");
    }

    function onKey(ev) {
        if (!document.body.contains(back)) return;
        if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); close(); }
        else if (ev.key === "ArrowDown") { ev.preventDefault(); sel = Math.min(sel + 1, results.length - 1); render(); }
        else if (ev.key === "ArrowUp") { ev.preventDefault(); sel = Math.max(sel - 1, 0); render(); }
        else if (ev.key === "Enter") { ev.preventDefault(); if (results[sel]) pick(results[sel].v); }
        ev.stopPropagation();   // keep ComfyUI shortcuts out while typing
    }

    back.addEventListener("mousedown", (ev) => { if (ev.target === back) close(); });
    list.addEventListener("click", (ev) => {
        const row = ev.target.closest(".k2ls-row");
        if (row) pick(results[+row.dataset.i].v);
    });
    chipsEl.addEventListener("click", (ev) => {
        const c = ev.target.closest(".k2ls-chip");
        if (!c) return;
        const r = c.dataset.r;
        roles.has(r) ? roles.delete(r) : roles.add(r);
        sel = 0; renderChips(); render();
    });
    favBtn.addEventListener("click", () => { favOnly = !favOnly; favBtn.classList.toggle("on", favOnly); sel = 0; render(); });
    back.querySelector(".k2ls-none").addEventListener("click", () => pick("none"));
    back.querySelector(".k2ls-ref").addEventListener("click", async () => {
        list.innerHTML = `<div style="padding:14px;color:#888">rebuilding index...</div>`;
        await loadIndex(true); renderChips(); render();
    });
    input.addEventListener("input", () => { sel = 0; render(); });
    document.addEventListener("keydown", onKey, true);
    input.focus();

    try {
        await loadIndex();
    } catch (e) {
        list.innerHTML = `<div style="padding:14px;color:#f7768e">index failed: ${esc(e)}</div>`;
        return;
    }
    renderChips();
    render();
}

// Route a click on the MIDDLE of a LoRA combo to the search popup. Arrow zones
// keep stepping; Shift+click falls back to the native list. Instance-level
// override of ComboWidget.onClick (frontend >= 1.2x widget classes); on a
// frontend without onClick the right-click menu entries still work.
function hookCombo(node, widget) {
    if (!widget || widget.__k2search) return;
    widget.__k2search = true;
    const orig = widget.onClick;
    if (typeof orig !== "function") return;
    widget.onClick = function (opts) {
        try {
            const e = opts?.e;
            const n = opts?.node || node;
            const x = e ? e.canvasX - n.pos[0] : -1;
            const w = this.width || n.size[0];
            if (e && !e.shiftKey && x >= 40 && x <= w - 40) {
                openSearch(n, this);
                return;
            }
        } catch (err) {
            console.warn("[EricKrea2] LoRA search hook:", err);
        }
        return orig.call(this, opts);
    };
}

app.registerExtension({
    name: "Eric.Krea2.MultiLoRA",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const pat = SEARCH_NODES[nodeData?.name];
        if (!pat) return;
        const origMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = origMenu ? origMenu.apply(this, arguments) : undefined;
            const ws = (this.widgets || []).filter((w) => pat.test(w.name) && !w.__k2hidden);
            ws.forEach((w) => options.push({
                content: `🔍 Search LoRA${ws.length > 1 ? ` (${w.name.replace("lora_", "slot ")})` : ""}...`,
                callback: () => openSearch(this, w),
            }));
            return r;
        };
    },

    async nodeCreated(node) {
        const pat = SEARCH_NODES[node?.comfyClass];
        if (pat) {
            for (const w of node.widgets || []) if (pat.test(w.name)) hookCombo(node, w);
        }
        if (!node || node.comfyClass !== NODE_CLASS) return;
        if (node.__k2ml_init) return;
        node.__k2ml_init = true;

        for (const p of rowPairs(node)) {
            const orig = p.lw.callback;
            p.lw.callback = function (v, ...a) {
                const r = orig ? orig.call(this, v, ...a) : undefined;
                relayout(node);
                return r;
            };
        }

        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            setTimeout(() => relayout(node), 0);
            return r;
        };
        setTimeout(() => relayout(node), 0);
    },
});
