# Spec: checkpoint profiles (+ shift-mu offset)

Status: DRAFT for Eric's review - nothing built. 2026-11-12.

## Why

Sweeps showed each Turbo fine-tune has its own preferences: sampler, schedule, steps, hybrid split and
shift mu. At 2 MP the preferred mu ranges from ~1.35 to ~1.65 across checkpoints, around the
`resolution` rule's 1.30. One global setting can't serve them all. A **checkpoint profile** stores what a
checkpoint likes, and it is applied automatically whenever that checkpoint runs - including per cell in a
checkpoint sweep, so checkpoints can be compared each at its own best.

## Decisions (Eric, 2026-11-12)

1. A **Checkpoint Profile node**, resolved at run time (works headless and per cell in sweeps) - not
   values written into the panel.
2. Precedence **sweep axis > profile > panel**, and it must be **visible** which fields are being
   overridden, and by what.
3. Styles, kept simple: **general**, **realistic** (photo / detailed), **illustration** (anime / clean).
4. Helper LoRAs in profiles: **not now** - on the to-do list.

## Part A - shift mu: measured values per size (revised 2026-11-12, Eric's call)

A profile stores the **mu values you measured**, per image size - no offsets. Example for one checkpoint,
realistic style: 2 MP 1.55, 3 MP 1.65, 5 MP 2.25. (Those don't follow the `resolution` rule's shape -
1.30 / 1.71 / 2.53 - so a single offset couldn't have represented them anyway.)

```json
"mu_s1":     {"2": 1.55, "3": 1.65, "5": 2.25},
"mu_refine": {"4.5": 2.1}
```

- Keys are megapixels; `mu_s1` is looked up at Stage 1's size, `mu_refine` at the Stage 2 / Stage 3 size.
- **Curve through the measured points (Eric, 2026-11-12):** at a measured size the stored value is used
  exactly; between measured sizes, straight segments connect the points; outside the measured range the
  curve continues along the best-fit trend of ALL the points (so one noisy pick doesn't swing the
  extrapolation). With a single point there is no trend yet, so it continues along the `resolution`
  rule's slope.
- The Save Profile dialog draws it: your points, the curve the profile will use, and the `resolution`
  rule for comparison.
- When a profile supplies a table it decides that stage's mu, whatever the panel's `distilled_shift` says.
  A stage with no table uses the panel's own setting.
- The sweep's existing `mus` field is how the values get measured; nothing new is needed on the Ultra panel.

## Part B - the profile file

`checkpoint_profiles.json` in the pack folder (the user's own file, git-ignored; an example ships in
`defaults/`). Keyed by the checkpoint's name as the loader lists it (`Krea2\...\file.safetensors`):

```json
{
  "Krea2\\2949523-godago\\3357324-v2_0\\3357324_godago_v20_3244980.safetensors": {
    "general":      {"s1_sampler": "lcm_hybrid2", "sigma_profile": "EXP7", "s1_steps": 12,
                     "s1_hybrid_steps": 6,
                     "mu_s1": {"2": 1.45, "3": 1.7}},
    "realistic":    {"mu_s1": {"2": 1.55, "3": 1.65, "5": 2.25}, "mu_refine": {"4.5": 2.1}},
    "illustration": {"s1_eta": 0.0},
    "notes": "likes more shift; wet look below 1.45 at 2 MP"
  }
}
```

- A style's fields are layered on top of `general` (realistic/illustration only list what differs).
- Any Ultra widget key is allowed (validated against the Ultra panel, unknown keys warned and skipped),
  plus `sigma_profile` (a saved ★ curve, same mechanism as the sweep's).
- Phase 1 key set offered by Save Profile: per stage sampler, schedule, steps, hybrid steps, eta, noise,
  cfg; `sigma_profile`; `turbo_guidance`; the mu tables (entered as size -> value rows).

## Part C - the Eric Krea2 Checkpoint Profile node

- Wiring: **Component Loader -> (LoRA stack) -> Checkpoint Profile -> Ultra**. It passes the pipeline
  through and attaches the profile policy to it - no new Ultra input, and checkpoint-sweep switching keeps
  it (the switch copies the pipeline's extra keys).
- Widgets: `enabled`, `style` (general / realistic / illustration).
- At run time Ultra looks up the profile for **the checkpoint actually loaded** (per cell in a checkpoint
  sweep), layers style over default, and applies its fields over the panel values - except fields a sweep
  cell sets, which win. A checkpoint with no profile runs on the panel (noted, not an error).
- **★ Save Profile** button: opens a check-box list of the candidate fields with the current panel values,
  for the checkpoint selected in the loader and the node's style. Tick what to store, Save. No JSON typing.
  Saving an existing profile shows old -> new values first.

## Part D - showing what is overridden (decision 2)

Live, before you queue, updated when the loader's checkpoint, the profile style, or the Sweep Plan changes:
- **Profile node:** a read-out of the resolved profile for the loaded checkpoint + style -
  e.g. `s1_sampler lcm_hybrid2 (profile)`, `s1_steps 12 -> swept`, or "no profile for this checkpoint".
- **Ultra node:** each overridden widget gets a coloured marker and a suffix on its label -
  teal `<- profile`, pink `<- sweep` (the sweep's existing colour). The widget's own value stays visible
  so you can see what the profile/sweep replaces.
- Source of truth: a server route resolves profile + style for a checkpoint name; the swept field list
  comes from the Sweep Plan's existing plan-preview endpoint (extended to return the swept keys), so it
  covers grid fields and combos (text or file) alike.
- After a run, each PNG chunk gets a `profile` section (checkpoint, style, fields applied, fields the sweep
  overrode), and sweeps get a `profile` column in the CSV / manifest - durable, not console-only.

## Phasing

1. Mu tables + profile file + node (runtime application, per-cell in sweeps) + Save Profile + Profile-node
   read-out + PNG/CSV records.
2. Ultra widget markers (Part D, Ultra side); "promote sweep cell to profile"; VAE in profiles (needs the
   profile node to load the chosen decode VAE); profile-aware checkpoint sweeps report.
3. To-do: helper LoRAs in profiles; the sweep report suggesting each checkpoint's mu values from ratings.

## Open questions

None open (2026-11-12: style named `general`; mu stored as measured values per size, not offsets).
