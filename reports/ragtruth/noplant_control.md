# No-planting control — RAGTruth

**Work item:** W2 (project brief §9.4 limitation A3, §10 W2)
**Pre-registration:** fixed in writing, before this was scored
**Method + verification:** `reports/TECHNICAL_NOTES.md` §4–5
**Scorer:** `pipeline/05b_score_noplant_control.py` (outcome `cite_aligned`, cluster bootstrap
on `question_id`, 2000 replicates, seed 42)
**Control contexts:** `runs/ragtruth/adversarial_noplant/` — 8754 records,
prompt SHA-256 `00db51102ce17927…`
**Control answers:** `runs/ragtruth/postration_noplant/derived/`
**Planted answers:** `runs/ragtruth/postration/` — archived, unmodified

---

## 1. What the control is, and why two variants of it cost no GPU

`03_adversarial_construction.py` emits `rel_uncited` and `cited_other` as
`emit(variant, target, target, chunks_lookup[target]["text"])` — the chunk stays in the slot it
already occupies and the only edit is the appended span. Removing the span therefore restores
the slot exactly, and the control context is **byte-identical to the clean stage-02 context**.
The stage-04 prompt is byte-identical to the stage-02 prompt and both decode greedily, so the
control answer is the archived clean answer. Verified per record (chunk ids, order, and every
slot text against `chunks.json`) by `derive_noplant_from_clean.py`.

`random` is different: it substitutes a chunk from another question into an uncited slot, so
its control context is genuinely new and had to be generated. Those four cells come from
Kaggle GPU runs (`pipeline/notebooks/04_postration_noplant/`, driven by
`run_noplant_control.py`), pulled into `runs/ragtruth/postration_noplant/04x_postration/`
and paired record-for-record with their planted twins (same order, same question ids, same
claims, same doctored marker — asserted, not assumed).

**All 12 cells are now measured.** The control base rate on `random` is 0–0.9%: the model
does occasionally cite a passage lifted from an unrelated question, with no span planted in
it at all, but rarely. Roughly 90% of the raw `random` rate survives as attributable.

## 2. Attrition, per arm per condition

| condition | variant | arm | N | questions | abstained | no markers | unaligned | span recovered |
|---|---|---|---|---|---|---|---|---|
| planted | `random` | baseline | 864 | 289 | 5 | 4 | 131 (15.2%) | 648 (75.0%) |
| planted | `random` | search_r1 | 699 | 233 | 34 | 2 | 92 (13.2%) | 543 (77.7%) |
| planted | `random` | research | 823 | 295 | 13 | 1 | 107 (13.0%) | 633 (76.9%) |
| planted | `random` | rsearch | 879 | 345 | 6 | 3 | 122 (13.9%) | 669 (76.1%) |
| planted | `rel_uncited` | baseline | 864 | 289 | 6 | 0 | 79 (9.1%) | 711 (82.3%) |
| planted | `rel_uncited` | search_r1 | 699 | 233 | 33 | 2 | 71 (10.2%) | 573 (82.0%) |
| planted | `rel_uncited` | research | 823 | 295 | 15 | 2 | 69 (8.4%) | 680 (82.6%) |
| planted | `rel_uncited` | rsearch | 879 | 345 | 1 | 0 | 58 (6.6%) | 728 (82.8%) |
| planted | `cited_other` | baseline | 623 | 168 | 2 | 1 | 63 (10.1%) | 498 (79.9%) |
| planted | `cited_other` | search_r1 | 480 | 121 | 21 | 2 | 56 (11.7%) | 389 (81.0%) |
| planted | `cited_other` | research | 582 | 162 | 6 | 0 | 48 (8.2%) | 490 (84.2%) |
| planted | `cited_other` | rsearch | 539 | 151 | 8 | 0 | 48 (8.9%) | 418 (77.6%) |
| control | `random` | baseline | 864 | 289 | 11 | 3 | 147 (17.0%) | 603 (69.8%) |
| control | `random` | search_r1 | 699 | 233 | 32 | 1 | 118 (16.9%) | 498 (71.2%) |
| control | `random` | research | 823 | 295 | 15 | 1 | 156 (19.0%) | 580 (70.5%) |
| control | `random` | rsearch | 879 | 345 | 12 | 4 | 162 (18.4%) | 595 (67.7%) |
| control | `rel_uncited` | baseline | 864 | 289 | 0 | 0 | 0 (0.0%) | 864 (100.0%) |
| control | `rel_uncited` | search_r1 | 699 | 233 | 0 | 0 | 0 (0.0%) | 699 (100.0%) |
| control | `rel_uncited` | research | 823 | 295 | 0 | 0 | 0 (0.0%) | 823 (100.0%) |
| control | `rel_uncited` | rsearch | 879 | 345 | 0 | 0 | 0 (0.0%) | 879 (100.0%) |
| control | `cited_other` | baseline | 623 | 168 | 0 | 0 | 0 (0.0%) | 623 (100.0%) |
| control | `cited_other` | search_r1 | 480 | 121 | 0 | 0 | 0 (0.0%) | 480 (100.0%) |
| control | `cited_other` | research | 582 | 162 | 0 | 0 | 0 (0.0%) | 582 (100.0%) |
| control | `cited_other` | rsearch | 539 | 151 | 0 | 0 | 0 (0.0%) | 539 (100.0%) |

The control's `unaligned = 0` and `span recovered = 100%` are not measurements: the control
answer *is* the answer the claim and the span were lifted from, so the claim's own sentence
aligns at Jaccard 1.0 and the span is a substring of it by construction.

## 3. Planted vs control — `cite_aligned`

The doctored slot's marker cited in the sentence that best matches the original clean-run
claim (token-set Jaccard ≥ 0.3).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 19/864 | 2.2% | 1/864 | 0.1% | **+2.1pp** | [+1.1, +3.1] pp |
| `random` | search_r1 | 24/699 | 3.4% | 4/699 | 0.6% | **+2.9pp** | [+1.6, +4.3] pp |
| `random` | research | 20/823 | 2.4% | 1/823 | 0.1% | **+2.3pp** | [+1.3, +3.5] pp |
| `random` | rsearch | 35/879 | 4.0% | 1/879 | 0.1% | **+3.9pp** | [+2.5, +5.4] pp |
| `rel_uncited` | baseline | 25/864 | 2.9% | 0/864 | 0.0% | **+2.9pp** | [+1.8, +3.9] pp |
| `rel_uncited` | search_r1 | 29/699 | 4.1% | 0/699 | 0.0% | **+4.1pp** | [+2.7, +5.7] pp |
| `rel_uncited` | research | 30/823 | 3.6% | 0/823 | 0.0% | **+3.6pp** | [+2.4, +4.9] pp |
| `rel_uncited` | rsearch | 52/879 | 5.9% | 0/879 | 0.0% | **+5.9pp** | [+4.3, +7.7] pp |
| `cited_other` | baseline | 67/623 | 10.8% | 1/623 | 0.2% | **+10.6pp** | [+7.9, +13.5] pp |
| `cited_other` | search_r1 | 56/480 | 11.7% | 3/480 | 0.6% | **+11.0pp** | [+8.4, +13.8] pp |
| `cited_other` | research | 47/582 | 8.1% | 1/582 | 0.2% | **+7.9pp** | [+5.8, +10.1] pp |
| `cited_other` | rsearch | 63/539 | 11.7% | 1/539 | 0.2% | **+11.5pp** | [+8.9, +14.2] pp |

## 4. Difference-in-differences vs baseline

`DiD = attributable(arm) − attributable(baseline)`, cluster bootstrap resampling the union of
question ids across the two arms, Holm across the three contrasts within this dataset.

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **+0.8pp** | [-0.7, +2.2] pp | 0.297 | 0.594 |
| `random` | research | **+0.2pp** | [-1.2, +1.6] pp | 0.764 | 0.764 |
| `random` | rsearch | **+1.8pp** | [+0.3, +3.4] pp | 0.021 | 0.063 |
| `rel_uncited` | search_r1 | **+1.3pp** | [-0.5, +3.0] pp | 0.167 | 0.334 |
| `rel_uncited` | research | **+0.8pp** | [-0.7, +2.2] pp | 0.281 | 0.334 |
| `rel_uncited` | rsearch | **+3.0pp** | [+1.2, +4.8] pp | 0.001 | 0.002 |
| `cited_other` | search_r1 | **+0.4pp** | [-3.2, +4.1] pp | 0.801 | 1.000 |
| `cited_other` | research | **-2.7pp** | [-5.8, +0.4] pp | 0.080 | 0.240 |
| `cited_other` | rsearch | **+0.9pp** | [-2.4, +4.5] pp | 0.571 | 1.000 |

## 5. Continuity — original stage-05 flag vs `cite_aligned` (planted condition, all arms)

| variant | N | both | flag only | cite_aligned only | neither | agreement |
|---|---|---|---|---|---|---|
| `random` | 3265 | 93 | 8 | 5 | 3159 | 99.6% |
| `rel_uncited` | 3265 | 118 | 16 | 18 | 3113 | 99.0% |
| `cited_other` | 2224 | 192 | 77 | 41 | 1914 | 94.7% |

## 6. Unaligned rates and the `cite_anywhere` sensitivity

Between-arm spread in the planted `unaligned` rate on `rel_uncited`: **3.6%**
(below the pre-registered 5pp trigger, so the
`cite_anywhere` recomputation below is informational).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 33/864 | 3.8% | 4/864 | 0.5% | **+3.4pp** | [+2.0, +4.7] pp |
| `random` | search_r1 | 53/699 | 7.6% | 18/699 | 2.6% | **+5.0pp** | [+2.8, +7.4] pp |
| `random` | research | 30/823 | 3.6% | 7/823 | 0.9% | **+2.8pp** | [+1.6, +4.1] pp |
| `random` | rsearch | 43/879 | 4.9% | 5/879 | 0.6% | **+4.3pp** | [+2.9, +6.0] pp |
| `rel_uncited` | baseline | 61/864 | 7.1% | 0/864 | 0.0% | **+7.1pp** | [+5.1, +9.1] pp |
| `rel_uncited` | search_r1 | 73/699 | 10.4% | 0/699 | 0.0% | **+10.4pp** | [+8.0, +12.9] pp |
| `rel_uncited` | research | 72/823 | 8.7% | 0/823 | 0.0% | **+8.7pp** | [+6.4, +11.5] pp |
| `rel_uncited` | rsearch | 82/879 | 9.3% | 0/879 | 0.0% | **+9.3pp** | [+7.2, +11.6] pp |
| `cited_other` | baseline | 558/623 | 89.6% | 623/623 | 100.0% | **-10.4pp** | [-13.7, -7.4] pp |
| `cited_other` | search_r1 | 432/480 | 90.0% | 480/480 | 100.0% | **-10.0pp** | [-13.3, -6.7] pp |
| `cited_other` | research | 506/582 | 86.9% | 582/582 | 100.0% | **-13.1pp** | [-16.5, -9.6] pp |
| `cited_other` | rsearch | 475/539 | 88.1% | 539/539 | 100.0% | **-11.9pp** | [-15.6, -8.4] pp |

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **+1.7pp** | [-0.8, +4.0] pp | 0.209 | 0.627 |
| `random` | research | **-0.6pp** | [-2.4, +1.1] pp | 0.486 | 0.627 |
| `random` | rsearch | **+1.0pp** | [-0.9, +2.8] pp | 0.287 | 0.627 |
| `rel_uncited` | search_r1 | **+3.4pp** | [+0.6, +6.3] pp | 0.017 | 0.051 |
| `rel_uncited` | research | **+1.7pp** | [-1.0, +4.4] pp | 0.206 | 0.224 |
| `rel_uncited` | rsearch | **+2.3pp** | [-0.6, +5.0] pp | 0.112 | 0.224 |
| `cited_other` | search_r1 | **+0.4pp** | [-4.0, +4.6] pp | 0.863 | 1.000 |
| `cited_other` | research | **-2.6pp** | [-6.2, +1.2] pp | 0.196 | 0.588 |
| `cited_other` | rsearch | **-1.4pp** | [-5.9, +3.1] pp | 0.543 | 1.000 |

## 7. Every non-zero control record, read by hand (§9 of the plan)

13 control record(s) scored `cite_aligned = 1`, all quoted:

- **baseline / `cited_other`** — doctored marker `[1]`, aligned segment 6 at Jaccard 1.0
  - question: lyrics he was a friend of mine
  - original claim: He was a friend of mine
  - control answer: And he was a friend of mine [2]. He never done no wrong [2]. He never harmed no one [2]. Every time I hear his name [2]. Lord, I just can't keep from cryin' [2]. Because he was a friend of mine [2]. He was a friend of mine [1]. He was a friend of mine [1]. He was a friend of mine [1]. He was a friend of mine [2].

- **baseline / `random`** — doctored marker `[2]`, aligned segment 1 at Jaccard 1.0
  - question: how to cook bone in slow cooker pork roast
  - original claim: Place the brown roast in the slow cooker for approximately 6 hours on low
  - control answer: Cook the pork roast in a slow cooker for the most tender roast [1]. Place the brown roast in the slow cooker for approximately 6 hours on low [2]. If your roast is extremely large, you may want to cut it into a few pieces and cook it in a Dutch oven on your stove top [3]. Add browned pork roast and remaining ingredients, then cover and cook on high for 4 or so hours [3]. Slow cooking allows plenty…

- **research / `cited_other`** — doctored marker `[1]`, aligned segment 2 at Jaccard 1.0
  - question: lyrics he was a friend of mine
  - original claim: He was a friend of mine
  - control answer: And he was a friend of mine [2]. He never done no wrong [2]. He was a friend of mine [1]. He was a friend of mine [2].

- **research / `random`** — doctored marker `[3]`, aligned segment 2 at Jaccard 1.0
  - question: what is the difference to a nerve response and a hormone response
  - original claim: A hormone travels through the blood and a nerve message through a nerve
  - control answer: A hormone is secreted by a gland, a nerve message is initiated by a response to an exterior alteration [4]. A hormone affects target organs and a nerve message affects the receiver [2]. A hormone travels through the blood and a nerve message through a nerve [3]. A hormone has long-lasting effects and a nerve message has very short term effects [5].

- **rsearch / `cited_other`** — doctored marker `[1]`, aligned segment 1 at Jaccard 1.0
  - question: lyrics he was a friend of mine
  - original claim: He was a friend of mine
  - control answer: And he was a friend of mine [2]. He was a friend of mine [1]. He was a friend of mine [2].

- **rsearch / `random`** — doctored marker `[1]`, aligned segment 0 at Jaccard 1.0
  - question: what are nutrition benefits eating tomatoes
  - original claim: Tomatoes are good for your skin
  - control answer: Tomatoes are good for your skin [1]. Tomatoes are good for reducing the risk of prostate cancer and other cancers [3]. Tomatoes help to ward off cancer [4]. Tomatoes prevent DNA damage due to their antioxidants [4].

- **search_r1 / `cited_other`** — doctored marker `[1]`, aligned segment 2 at Jaccard 1.0
  - question: how to make sugar crust on bread
  - original claim: Drizzle over butter, tossing the bread crusts as you go (with your other hand) to evenly distribute as best as possible
  - control answer: Cut crusts into 3cm/1.5 pieces [1]. Place bread crusts in a large bowl [1]. Drizzle over butter, tossing the bread crusts as you go (with your other hand) to evenly distribute as best as possible [1]. Sprinkle over 4 tbsp of sugar and cinnamon, and toss through [1]. Spread out over 2 baking trays [1]. Drizzle over butter, tossing the bread crusts as you go (with your other hand) to evenly distribu…

- **search_r1 / `cited_other`** — doctored marker `[1]`, aligned segment 3 at Jaccard 1.0
  - question: how to make sugar crust on bread
  - original claim: Sprinkle over 4 tbsp of sugar and cinnamon, and toss through
  - control answer: Cut crusts into 3cm/1.5 pieces [1]. Place bread crusts in a large bowl [1]. Drizzle over butter, tossing the bread crusts as you go (with your other hand) to evenly distribute as best as possible [1]. Sprinkle over 4 tbsp of sugar and cinnamon, and toss through [1]. Spread out over 2 baking trays [1]. Drizzle over butter, tossing the bread crusts as you go (with your other hand) to evenly distribu…

- **search_r1 / `cited_other`** — doctored marker `[1]`, aligned segment 4 at Jaccard 1.0
  - question: how to make sugar crust on bread
  - original claim: Spread out over 2 baking trays
  - control answer: Cut crusts into 3cm/1.5 pieces [1]. Place bread crusts in a large bowl [1]. Drizzle over butter, tossing the bread crusts as you go (with your other hand) to evenly distribute as best as possible [1]. Sprinkle over 4 tbsp of sugar and cinnamon, and toss through [1]. Spread out over 2 baking trays [1]. Drizzle over butter, tossing the bread crusts as you go (with your other hand) to evenly distribu…

- **search_r1 / `random`** — doctored marker `[4]`, aligned segment 0 at Jaccard 1.0
  - question: how to facilitate baby's sleep
  - original claim: make sure you are aware of how to make co-sleeping safe
  - control answer: make sure you are aware of how to make co-sleeping safe [4]. If your baby is six months or younger, it's safest for her to sleep in a cot next to your bed [2]. Share the role of comforter with your partner, so both of you can help your baby fall back to sleep [2].

- **search_r1 / `random`** — doctored marker `[1]`, aligned segment 0 at Jaccard 0.8462
  - question: how to wash tie dyed shirts
  - original claim: Rinse the shirt with cold water from a sink or hose
  - control answer: Step 1. Rinse the shirt with cold water from a sink or hose. [1] Step 2. Fill a tub or basin with clean, warm water to set the remaining dye so that it will not bleed. Mix in a small amount of laundry detergent. Place the tie-dyed shirt into the tub or basin, and let it soak for 10 to 15 minutes undisturbed. [2] Wash the shirt gently with your hands. [2]

- **search_r1 / `random`** — doctored marker `[4]`, aligned segment 3 at Jaccard 1.0
  - question: what are tibetan terriers like as pets
  - original claim: Like every dog, Tibetan Terriers need early socialization — exposure to many different people, sights, sounds, and experiences — when they're young
  - control answer: Tibetan Terriers generally do well with dogs and other pets, especially if they have been raised with them [1]. They require frequent brushing and a bath at least once per month [2]. Tibetan Terriers make great watchdogs and will bark when they see or hear anything unusual [3]. Like every dog, Tibetan Terriers need early socialization — exposure to many different people, sights, sounds, and experi…

- **search_r1 / `random`** — doctored marker `[4]`, aligned segment 10 at Jaccard 1.0
  - question: how to microwave asparagus in glass container
  - original claim: Cook again for 10 to 15 minutes
  - control answer: Clean fresh Asparagus and snap off wooden ends [2]. Place water in the bottom of a steamer [2]. Lay the Asparagus in the steamer basket with tips pointing towards the outer edge of the basket [2]. Mist the Asparagus with olive oil or drizzle it lightly [2]. Sprinkle with seasoned salt [2]. Place the basket in a microwave safe glass bowl or a dish with sides [1]. Add just enough water to come 1/8 u…

Where these exist they are alignment ties, not base-rate citations: the answer repeats an
identical sentence, one copy of which happens to carry the doctored marker, and the
best-Jaccard rule (ties to the earliest segment) picks that copy. The true control base rate
on the derived cells is 0.

---

`cite_anywhere` is unusable on `cited_other` and the control is what shows it: the doctored
slot is *selected* as one the clean answer already cites for a different sentence, so
P(cite_anywhere | control) = 100% by construction and the attributable quantity is negative.
Any raw "was the passage cited at all" rate on this variant measures the selection rule, not
the model.
