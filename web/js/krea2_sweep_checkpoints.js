// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - check-box picker for the Sweep Checkpoints node (2026-11-08).
//
// Adds a "Checkpoints: N of M" button to the node. It opens a list of every checkpoint the
// Component Loader can see under models/diffusion_models/krea2/ with check boxes, a filter
// box and All / None. Ticking writes the names, one per line and in pick-list order, into the
// node's `checkpoints` text widget (which is what is saved and run). Names ticked earlier that
// are no longer on disk stay listed, greyed, so they can be unticked. The button is not
// serialised, and the node is new, so no saved widget order is affected.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "EricKrea2SweepCheckpoints";

const norm = (s) => String(s || "").replace(/\\/g, "/").trim().toLowerCase();
const textWidget = (node) => (node.widgets || []).find((w) => w && w.name === "checkpoints");
const ticked = (node) => {
    const w = textWidget(node);
    return w ? String(w.value || "").split(/\r?\n/).map((s) => s.trim())
        .filter((s) => s && !s.startsWith("#")) : [];
};

async function fetchItems() {
    try {
        const r = await api.fetchApi("/eric_krea2/krea2_checkpoints");
        const j = await r.json();
        if (Array.isArray(j.items)) return j.items;
        return Array.isArray(j.names) ? j.names.map((n) => ({ name: n })) : [];
    } catch (e) {
        return [];
    }
}
async function fetchNames() {
    return (await fetchItems()).map((i) => i.name);
}
const fmtSize = (b) => (b == null ? "" : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
const fmtDate = (s) => {
    if (s == null) return "";
    const d = new Date(s * 1000);
    const z = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())}`;
};
const splitPath = (n) => {
    const s = String(n).replace(/\\/g, "/");
    const i = s.lastIndexOf("/");
    return i < 0 ? ["", s] : [s.slice(0, i), s.slice(i + 1)];
};

function setLabel(node, btn, total) {
    const n = ticked(node).length;
    const txt = total == null ? `Checkpoints: ${n} ticked  \u25BE` : `Checkpoints: ${n} of ${total}  \u25BE`;
    btn.name = txt;
    btn.label = txt;
    node.setDirtyCanvas(true, true);
}

function el(tag, style, text) {
    const e = document.createElement(tag);
    if (style) Object.assign(e.style, style);
    if (text != null) e.textContent = text;
    return e;
}

async function openPicker(node, btn) {
    // Wide window (2026-11-10): one checkpoint per line - file name, its folder, size, date -
    // sortable by clicking a column header. Display sorting never changes the RUN order, which
    // is always the pick-list order.
    const items = await fetchItems();
    const names = items.map((i) => i.name);
    const tw = textWidget(node);
    if (!tw) return;
    const cur = new Set(ticked(node).map(norm));
    const onDisk = new Set(names.map(norm));
    const missing = ticked(node).filter((t) => !onDisk.has(norm(t)));

    const overlay = el("div", { position: "fixed", inset: "0", background: "rgba(0,0,0,0.45)", zIndex: "10000" });
    const box = el("div", {
        position: "fixed", left: "50%", top: "4vh", transform: "translateX(-50%)",
        width: "min(1280px, 96vw)", height: "90vh", display: "flex", flexDirection: "column",
        gap: "8px", padding: "14px", borderRadius: "8px", fontSize: "13px",
        background: "var(--comfy-menu-bg, #222)", color: "var(--fg-color, #ddd)",
        border: "1px solid var(--border-color, #555)", boxShadow: "0 8px 30px rgba(0,0,0,.5)",
    });
    box.appendChild(el("div", { fontWeight: "bold", fontSize: "15px" }, "Checkpoints to sweep (Turbo only)"));
    const filter = el("input", { padding: "5px 8px", background: "var(--comfy-input-bg, #111)",
        color: "inherit", border: "1px solid var(--border-color, #555)", borderRadius: "4px" });
    filter.placeholder = "filter by name or folder...";
    box.appendChild(filter);
    const bar = el("div", { display: "flex", gap: "6px", alignItems: "center" });
    const mkBtn = (t) => el("button", { padding: "3px 10px", cursor: "pointer" }, t);
    const bAll = mkBtn("Tick shown"), bNone = mkBtn("Untick shown"), bClose = mkBtn("Done");
    const status = el("span", { marginLeft: "auto", opacity: "0.85" });
    bar.append(bAll, bNone, status, bClose);
    box.appendChild(bar);

    const cols = "28px minmax(220px, 2fr) minmax(160px, 3fr) 80px 96px";
    const cell = { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" };
    const head = el("div", { display: "grid", gridTemplateColumns: cols, gap: "8px", padding: "4px 6px",
        fontWeight: "bold", borderBottom: "1px solid var(--border-color, #444)", userSelect: "none" });
    const hdr = (label, key) => {
        const h = el("div", Object.assign({ cursor: key ? "pointer" : "default" }, cell), label);
        if (key) h.addEventListener("click", () => sortBy(key));
        return h;
    };
    head.append(hdr("", null), hdr("File", "file"), hdr("Folder", "folder"),
        hdr("Size", "size"), hdr("Date", "mtime"));
    box.appendChild(head);
    const list = el("div", { overflowY: "auto", overflowX: "hidden", flex: "1" });
    box.appendChild(list);

    const rows = [];
    const addRow = (it, isMissing) => {
        const [folder, file] = splitPath(it.name);
        const row = el("label", { display: "grid", gridTemplateColumns: cols, gap: "8px", alignItems: "center",
            padding: "3px 6px", cursor: "pointer", opacity: isMissing ? "0.5" : "1",
            borderBottom: "1px solid rgba(128,128,128,0.12)" });
        row.title = it.name;
        const cb = el("input");
        cb.type = "checkbox";
        cb.checked = cur.has(norm(it.name));
        row.append(cb,
            el("span", Object.assign({}, cell), isMissing ? `${file}  (not on disk)` : file),
            el("span", Object.assign({ opacity: "0.6" }, cell), folder),
            el("span", Object.assign({ textAlign: "right", opacity: "0.85" }, cell), fmtSize(it.size)),
            el("span", Object.assign({ opacity: "0.85" }, cell), fmtDate(it.mtime)));
        list.appendChild(row);
        rows.push({ name: it.name, file: file.toLowerCase(), folder: folder.toLowerCase(),
            size: it.size ?? -1, mtime: it.mtime ?? -1, cb, row, isMissing });
        cb.addEventListener("change", write);
    };
    items.forEach((it) => addRow(it, false));
    missing.forEach((n) => addRow({ name: n }, true));
    if (!items.length) list.appendChild(el("div", { opacity: "0.7", padding: "6px" },
        "No checkpoints found in models/diffusion_models/krea2/."));

    let sortKey = null, sortDir = 1;
    function sortBy(key) {
        sortDir = sortKey === key ? -sortDir : (key === "size" || key === "mtime" ? -1 : 1);
        sortKey = key;
        const sorted = rows.slice().sort((a, b) => (a[key] > b[key] ? 1 : a[key] < b[key] ? -1 : 0) * sortDir);
        sorted.forEach((r) => list.appendChild(r.row));
    }
    function write() {
        // run order = pick-list order, whatever the display sort
        const picked = names.filter((n) => rows.find((r) => r.name === n && r.cb.checked))
            .concat(rows.filter((r) => r.isMissing && r.cb.checked).map((r) => r.name));
        tw.value = picked.join("\n");
        if (tw.callback) { try { tw.callback(tw.value); } catch (e) { /* ignore */ } }
        status.textContent = `${picked.length} of ${names.length} ticked`;
        setLabel(node, btn, names.length);
    }
    const shown = () => rows.filter((r) => r.row.style.display !== "none");
    filter.addEventListener("input", () => {
        const q = filter.value.trim().toLowerCase();
        rows.forEach((r) => { r.row.style.display = !q || r.name.toLowerCase().includes(q) ? "grid" : "none"; });
    });
    bAll.addEventListener("click", () => { shown().forEach((r) => { if (!r.isMissing) r.cb.checked = true; }); write(); });
    bNone.addEventListener("click", () => { shown().forEach((r) => { r.cb.checked = false; }); write(); });
    const close = () => { document.removeEventListener("keydown", onKey); overlay.remove(); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    bClose.addEventListener("click", close);
    overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", onKey);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    status.textContent = `${rows.filter((r) => r.cb.checked).length} of ${names.length} ticked`;
    filter.focus();
}

app.registerExtension({
    name: "Eric.Krea2.SweepCheckpoints",
    // Same hook + addWidget form as the pack's other buttons (Show plan, Save Preset), which
    // render on the current front-end; the earlier beforeRegisterNodeDef/onNodeCreated form did not.
    async nodeCreated(node) {
        if (!node || node.comfyClass !== NODE_CLASS || node._k2CkBtn) return;
        const btn = node.addWidget("button", "pick_checkpoints", null,
            () => openPicker(node, btn), { serialize: false });
        if (!btn) return;
        btn.serialize = false;
        node._k2CkBtn = btn;
        setLabel(node, btn, null);
        const refresh = () => fetchNames().then((n) => setLabel(node, btn, n.length));
        refresh();
        // a saved workflow fills the text widget after creation - recount once it has
        setTimeout(refresh, 500);
        const sz = node.computeSize?.();
        if (sz && node.size && node.size[1] < sz[1]) node.setSize?.([node.size[0], sz[1]]);
    },
});
