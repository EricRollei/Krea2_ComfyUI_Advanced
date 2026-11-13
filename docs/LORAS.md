# LoRAs: search, prompt tags & trigger words

[<- Back to the README](../README.md) | [All docs](../README.md#documentation)

## LoRA search & suggestions

- **🔍 Search popup** - click the **middle** of a LoRA dropdown on the Multi-LoRA Stack,
  Apply LoRA or Diagnose LoRA node: a search box over filenames, folders, trigger words and
  **LoRA Catalog** cards (name, role, effect text, artists, depicts, facet tags). The arrow
  zones still step prev / next, Shift+click opens the native list, and right-click the node
  → "Search LoRA ..." per slot is a fallback. No widgets are added - saved workflows are
  untouched. Catalog location: env `ERIC_LORA_CATALOG`, else video_prompter's config,
  else `L:/Models/loras/_lora_catalog.sqlite`; a missing catalog just means fewer fields.
- **`suggestions` socket** on the Multi-LoRA Stack - LoRAs picked by the LoRA Suggester
  (video_prompter) are added after the panel's own slots with their per-stage weights and
  trigger words; a file already in the stack is not added twice. The Prompt LoRA Tags node
  feeds the same socket.

### Prompt LoRA tags

The **Eric Krea2 Prompt LoRA Tags** node lets you load LoRAs from the prompt text, the way
A1111-style prompts do:

```
a woman in a red coat <lora:Shouji_Ink_KR2:0.66>, rain
```

| Tag | Weight |
|---|---|
| `<lora:name>` | 1.0 on all stages |
| `<lora:name:0.66>` | 0.66 on all three stages |
| `<lora:name:1/0.7/0.4>` | per stage S1 / S2 / S3 |
| `<lora:name:0.8:0.5>` (A1111 model:text-encoder) | the first number |

**Wiring:** prompt in -> the node's `prompt` output (tags removed, stray commas tidied) goes
on to the Multi-LoRA Stack's `prompt` input or the text encoder; its `loras` output goes into
the Multi-LoRA Stack's `suggestions` socket. The stack adds those LoRAs after its own slots,
never twice, with their trigger words (the LoRA Catalog card trigger first) in the
`add_triggers` merge. The `summary` output lists what was loaded.

**Names** are found like everywhere else in the pack: a full path, a loras-folder path
(extension optional), or the bare file name / stem in any subfolder. Civitai download names
are forgiven: `Shouji_Ink_KR2` finds `3317450_shouji_ink_kr2_bf16.safetensors` (the leading
version id and a `_bf16` / `_fp16` / `_fp8` suffix are ignored), but only when exactly one file
matches. An unknown or ambiguous name stops the run with the reason and the candidates;
`on_missing = skip` drops it with a console note instead.

## LoRA trigger words: base model vs fine-tunes (measured Oct 2026)

How much a LoRA needs its trigger words depends on the **checkpoint**. A trigger test (LoRA Catalog
`run.py triggers --plan-test`, details in the catalog's `docs/TRIGGERS.md`) rendered five Krea 2 style
LoRAs with each prompt variant - no words, the trigger token, the official trained words, trigger + style
phrase - plus a words-only cell without the LoRA, on three checkpoints. Score = CSD style gain toward the
LoRA's own look over that checkpoint's plain no-LoRA render (one seed, one neutral subject, card weights):

| LoRA | variant | Lox (fine-tune) | FinalCut (fine-tune) | base turbo |
|---|---|---|---|---|
| Michael Parkes | no words / trigger / official | +0.00 / +0.01 / +0.55 | +0.03 / +0.02 / +0.52 | **+0.44** / +0.24 / +0.57 |
| Liquid Chalk | no words / trigger / trigger+phrase | +0.03 / +0.02 / +0.43 | +0.06 / +0.46 / +0.42 | **+0.60** / +0.60 / +0.69 |
| Ukiyo-e | no words / trigger / official | -0.02 / +0.06 / +0.39 | -0.03 / +0.04 / +0.36 | +0.03 / +0.38 / +0.49 |
| Ballpoint | no words / trigger | +0.02 / +0.80 | +0.17 / +0.82 | +0.08 / +0.70 |
| zy cinematic | no words / trigger / trigger+phrase | +0.00 / +0.01 / +0.18 | +0.04 / +0.20 / +0.33 | +0.08 / +0.14 / +0.35 |

- **Base turbo:** two of the five (Parkes, Liquid Chalk) show strongly with no words at all, and the bare
  trigger token works far better than on the fine-tunes (Ukiyo-e +0.38 vs +0.06 on Lox). Stacks can stay
  lean - a trigger, not a paragraph - leaving more of the 507-token budget for the subject and more LoRAs.
- **Fine-tunes:** most LoRAs do almost nothing on a neutral prompt, and the bare trigger token does not
  change that. The official trained words or trigger + a style phrase switch them on. Budget their words.
- **Sometimes the words alone do it:** on Lox, Ukiyo-e and zy cinematic scored nearly the same from the
  words without the LoRA - the checkpoint already knows that look.
- **One trigger, two behaviours:** Ballpoint and Liquid Chalk share `n0vuschr0ma style`; on Lox it switches
  Ballpoint fully on and does nothing for Liquid Chalk.

Practical consequence for the trigger modes (`add_triggers` on the LoRA / Multi-LoRA Stack nodes,
`lora_triggers` on Sweep Plan): `catalog` sends one short trigger, which is enough on the base model and
often not enough on a fine-tune; `prepend` sends every Civitai trigger word. When testing a LoRA on a
fine-tune, test it with its words.

**The `recipe` trigger mode (2026-10-15)** puts these findings to work. On Apply LoRA, the
Multi-LoRA Stack (`add_triggers`) and the Sweep Plan (`lora_triggers`), `recipe` prepends each
LoRA's *prompt words* from the LoRA Catalog, chosen for the checkpoint that is loaded (the
transformer file's name, the same name a sweep folder carries):

1. your pick for this checkpoint (card editor "Prompt words", or "use words" on the Sweeps page),
2. a trigger test's pick for this checkpoint (`run.py triggers --check`),
3. your pick for every checkpoint, else the newest test pick,
4. the recipe's trigger + style phrase (untested),
5. the card's first trigger; no catalog words at all: as `catalog`.

A pick of the *no words* variant adds nothing on purpose (some LoRAs show best without words on a
base checkpoint). The console line says where each LoRA's words came from.

**Merging a stack's words (2026-10-16, `_prompt_merge.py`).** With two or more LoRAs in
`recipe` mode (Multi-LoRA Stack, Sweep Plan stacks), the words are merged once for the whole
stack instead of pasted end to end:

1. *Repeats* - a phrase said twice is kept once; a phrase whose words all sit in a longer
   phrase of another LoRA is dropped ("watercolor washes" inside "delicate layered watercolor
   washes"); phrases the prompt already says are dropped.
2. *Picks* - a LoRA whose words are a pick (yours or a trigger test's) keeps all of them.
3. *Order and length* - each LoRA's first phrase (its trigger) comes first, in stack order,
   then phrases two LoRAs share, then the rest. The other LoRAs share a budget of about 60
   words by weight; each keeps its first two phrases at least.
4. *Conflicts* - opposite words between LoRAs, or a LoRA and the prompt ("muted" vs
   "vibrant", "photorealistic" vs "anime", "dark" vs "airy" ...), are printed in the console
   and, in a sweep, recorded per cell (`word_conflicts`). Nothing is removed for a conflict;
   that choice is yours.

A single LoRA's words are used as they are.
