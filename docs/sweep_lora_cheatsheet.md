# Sweep Plan - LoRA cheat sheet

**Always:** `enabled` ON (magenta outline) · press **Show plan** before queueing · upstream Multi-LoRA Stack: `add_triggers` none (or `stack_enabled` off) so its trigger words don't leak into every cell.

---

## 1 · Compare single LoRAs (bakeoff)
`lora_mode` **bakeoff** · `include_baseline` ON (adds a no-LoRA reference) · `keep_panel_stack` OFF · `lora_strengths` **card**
```
comic_book_v2
Frazetta Style
L:\Models\loras\Krea2\Illustration\KreaFortiche.safetensors
```
`lora_strengths`: `card` = card weight · `0.6, 0.8, 1.0` = every LoRA at each · `card, 1.0` = both

## 2 · Random / all LoRAs (bakeoff)
```
random: graphic-art x6          6 from the loras/Krea2/graphic-art folder
all: Illustration/Comic         every LoRA in that folder
random: catalog watercolor x4   4 whose card mentions watercolor
random: graphic-art x6 seed=7   another repeatable draw  (seed=new = fresh every queue)
```

## 3 · Catalog filters (combine freely)
```
random: catalog creator=Adel_Ai x4            by creator (author= works too)
random: catalog creator=Adel_Ai tag=watercolor
all: catalog tag="art nouveau"                quotes for phrases
all: catalog kind=style_medium ink
random: catalog role=style noir x3
all: catalog creator=Adel_Ai arch=Flux        another base (default Krea2; arch=any = all)
```

## 4 · One LoRA for each prompt (bakeoff + prompts)
`lora_mode` **bakeoff** · `prompts` = several lines (or a file, see 9) · the per-prompt line must be the **only** line
```
random: catalog creator=Adel_Ai per prompt       1 LoRA per prompt
random: catalog creator=Adel_Ai x2 per prompt    2 single-LoRA cells per prompt
```

## 5 · Stacks
`lora_mode` **stacks** · `include_baseline` ON · `stack_parts` **each LoRA alone** (best for rating - lets the catalog compute synergy)
```
comic_book_v2 @0.8 | V0id @0.6                    a fixed stack
random: graphic-art x4 @0.8 | V0id @0.6           4 stacks, new pick each
Parkes | all: illustration                        Parkes with every illustration LoRA
cross: random: illustration x4 | random: graphic-art x4    all 16 combinations
```
`stack_parts`: each LoRA alone · reference only (fast screening) · build up (A, A+B, A+B+C) · leave one out
`lora_strengths` (stacks): blank = weights as written · `0.8, 1.0` = every stack scaled by each

## 6 · One stack per prompt (stacks + prompts)
`lora_mode` **stacks** · one `per prompt` covers the whole line
```
random: comic | random: illustration per prompt         1 comic + 1 illustration per prompt
random: catalog creator=Adel_Ai x2 per prompt @0.8      a 2-LoRA stack per prompt
random: catalog comic per prompt | V0id @0.6            + a fixed LoRA in every stack
```

## 7 · Proven partners (stacks)
```
random: catalog comic x4 | partner: best      each pick + its best-rated partner
Frazetta Style | partner: random              any of its good partners
random: catalog comic per prompt | partner: best @0.6
```
From your rated stacks (catalog synergy), else what Civitai users stack it with. No partner known = slot dropped.

## 8 · Word options (cards with several modes)
```
cinematic_grade #teal orange @0.8    that option's words
cinematic_grade #each                one cell per option
cinematic_grade #random              one option at random
random: catalog comic x2 per prompt #random   each pick its own option
```

## 9 · Prompts from a file (`prompts` field)
```
file: D:\prompts\portraits.txt                every line
random: D:\prompts\portraits.txt x20          20 at random  (seed=7 / seed=new)
A lighthouse at dawn, oil on canvas           typed prompts mix in
```

## 10 · Other modes
- **On top of your own stack:** `keep_panel_stack` ON - swept LoRA/stack is added to the upstream stack (baseline = your stack alone).
- **Strength ladder:** `lora_mode` **strength_ladder** · `lora_target` `1` (or part of a name) · `lora_strengths` `0.4, 0.6, 0.8, 1.0` - varies one entry of your upstream stack.
- **No LoRA sweep:** `lora_mode` **none** - every cell uses the upstream stack unchanged.

## Trigger words (`lora_triggers`)
**saved pick** (default, best) · full (whole recipe) · trigger (main trigger only) · none. Baseline cells get none.
