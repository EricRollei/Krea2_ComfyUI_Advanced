// Copyright (c) 2026 Eric Hiss. All rights reserved.
// Licensed under the terms in LICENSE.md.
//
// Eric_Krea2 - "★ Save recipe → presets" button for the Settings-from-Image node.
// Reads the selected image widget, asks the server to extract the embedded
// KREA2_SETTINGS recipe and save each present section (loader / ultra / lora)
// into its preset library under a name you choose. This is the image → named-
// preset bridge: any master PNG becomes selectable in the node panels.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function findWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

app.registerExtension({
    name: "Eric.Krea2.SettingsFromImage",
    async nodeCreated(node) {
        if (!node || node.comfyClass !== "EricKrea2SettingsFromImage") return;
        if (node.__krea2_saveRecipe_added) return;
        node.__krea2_saveRecipe_added = true;

        node.addWidget("button", "★ Save recipe → presets", null, async () => {
            const iw = findWidget(node, "image");
            const image = iw && iw.value;
            if (!image) { alert("Pick an image first."); return; }

            let name = window.prompt("Save this image's recipe to presets as:", "");
            if (name === null) return;          // cancelled
            name = name.trim();
            if (!name) { alert("Preset name cannot be empty."); return; }

            try {
                const resp = await api.fetchApi("/eric_krea2/save_image_presets", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image, name }),
                });
                const j = await resp.json().catch(() => ({}));
                if (!resp.ok || !j.ok) throw new Error(j.error || `HTTP ${resp.status}`);
                const saved = (j.saved || []).join(", ") || "(none)";
                alert(`Saved recipe "${j.name}" to preset libraries: ${saved}.\n` +
                      `Reload the graph to see it in the ${saved} preset dropdown(s).`);
            } catch (e) {
                alert(`Save failed: ${e && e.message ? e.message : e}`);
            }
        });
    },
});
