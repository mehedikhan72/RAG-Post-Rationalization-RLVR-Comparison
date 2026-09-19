# No-planting control — HAGRID

**Work item:** W2 (project brief §9.4 limitation A3, §10 W2)
**Pre-registration:** fixed in writing, before this was scored
**Method + verification:** `reports/TECHNICAL_NOTES.md` §4–5
**Scorer:** `pipeline/05b_score_noplant_control.py` (outcome `cite_aligned`, cluster bootstrap
on `question_id`, 2000 replicates, seed 42)
**Control contexts:** `runs/hagrid/adversarial_noplant/` — 3575 records,
prompt SHA-256 `00db51102ce17927…`
**Control answers:** `runs/hagrid/postration_noplant/derived/`
**Planted answers:** `runs/hagrid/postration/` — archived, unmodified

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
`run_noplant_control.py`), pulled into `runs/hagrid/postration_noplant/04x_postration/`
and paired record-for-record with their planted twins (same order, same question ids, same
claims, same doctored marker — asserted, not assumed).

**All 12 cells are now measured.** The control base rate on `random` is 0–0.9%: the model
does occasionally cite a passage lifted from an unrelated question, with no span planted in
it at all, but rarely. Roughly 90% of the raw `random` rate survives as attributable.

## 2. Attrition, per arm per condition

| condition | variant | arm | N | questions | abstained | no markers | unaligned | span recovered |
|---|---|---|---|---|---|---|---|---|
| planted | `random` | baseline | 433 | 377 | 6 | 1 | 42 (9.7%) | 358 (82.7%) |
| planted | `random` | search_r1 | 383 | 326 | 17 | 0 | 42 (11.0%) | 309 (80.7%) |
| planted | `random` | research | 434 | 382 | 10 | 0 | 47 (10.8%) | 354 (81.6%) |
| planted | `random` | rsearch | 430 | 395 | 11 | 0 | 45 (10.5%) | 337 (78.4%) |
| planted | `rel_uncited` | baseline | 433 | 377 | 10 | 1 | 29 (6.7%) | 373 (86.1%) |
| planted | `rel_uncited` | search_r1 | 383 | 326 | 12 | 1 | 24 (6.3%) | 331 (86.4%) |
| planted | `rel_uncited` | research | 434 | 382 | 4 | 0 | 30 (6.9%) | 386 (88.9%) |
| planted | `rel_uncited` | rsearch | 430 | 395 | 5 | 0 | 26 (6.0%) | 375 (87.2%) |
| planted | `cited_other` | baseline | 61 | 25 | 0 | 0 | 9 (14.8%) | 47 (77.0%) |
| planted | `cited_other` | search_r1 | 51 | 19 | 5 | 0 | 7 (13.7%) | 39 (76.5%) |
| planted | `cited_other` | research | 61 | 26 | 0 | 0 | 6 (9.8%) | 47 (77.0%) |
| planted | `cited_other` | rsearch | 42 | 15 | 0 | 0 | 4 (9.5%) | 35 (83.3%) |
| control | `random` | baseline | 433 | 377 | 9 | 1 | 56 (12.9%) | 321 (74.1%) |
| control | `random` | search_r1 | 383 | 326 | 16 | 0 | 51 (13.3%) | 279 (72.8%) |
| control | `random` | research | 434 | 382 | 9 | 1 | 54 (12.4%) | 316 (72.8%) |
| control | `random` | rsearch | 430 | 395 | 13 | 0 | 57 (13.3%) | 307 (71.4%) |
| control | `rel_uncited` | baseline | 433 | 377 | 0 | 0 | 0 (0.0%) | 433 (100.0%) |
| control | `rel_uncited` | search_r1 | 383 | 326 | 0 | 0 | 0 (0.0%) | 383 (100.0%) |
| control | `rel_uncited` | research | 434 | 382 | 0 | 0 | 0 (0.0%) | 434 (100.0%) |
| control | `rel_uncited` | rsearch | 430 | 395 | 0 | 0 | 0 (0.0%) | 430 (100.0%) |
| control | `cited_other` | baseline | 61 | 25 | 0 | 0 | 0 (0.0%) | 61 (100.0%) |
| control | `cited_other` | search_r1 | 51 | 19 | 0 | 0 | 0 (0.0%) | 51 (100.0%) |
| control | `cited_other` | research | 61 | 26 | 0 | 0 | 0 (0.0%) | 61 (100.0%) |
| control | `cited_other` | rsearch | 42 | 15 | 0 | 0 | 0 (0.0%) | 42 (100.0%) |

The control's `unaligned = 0` and `span recovered = 100%` are not measurements: the control
answer *is* the answer the claim and the span were lifted from, so the claim's own sentence
aligns at Jaccard 1.0 and the span is a substring of it by construction.

## 3. Planted vs control — `cite_aligned`

The doctored slot's marker cited in the sentence that best matches the original clean-run
claim (token-set Jaccard ≥ 0.3).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 37/433 | 8.5% | 4/433 | 0.9% | **+7.6pp** | [+4.9, +10.4] pp |
| `random` | search_r1 | 33/383 | 8.6% | 0/383 | 0.0% | **+8.6pp** | [+5.8, +11.6] pp |
| `random` | research | 46/434 | 10.6% | 0/434 | 0.0% | **+10.6pp** | [+7.9, +13.6] pp |
| `random` | rsearch | 54/430 | 12.6% | 3/430 | 0.7% | **+11.9pp** | [+8.9, +15.2] pp |
| `rel_uncited` | baseline | 64/433 | 14.8% | 0/433 | 0.0% | **+14.8pp** | [+11.8, +18.2] pp |
| `rel_uncited` | search_r1 | 53/383 | 13.8% | 0/383 | 0.0% | **+13.8pp** | [+10.4, +17.6] pp |
| `rel_uncited` | research | 59/434 | 13.6% | 0/434 | 0.0% | **+13.6pp** | [+10.5, +17.0] pp |
| `rel_uncited` | rsearch | 80/430 | 18.6% | 0/430 | 0.0% | **+18.6pp** | [+15.0, +22.5] pp |
| `cited_other` | baseline | 12/61 | 19.7% | 0/61 | 0.0% | **+19.7pp** | [+10.8, +31.3] pp |
| `cited_other` | search_r1 | 7/51 | 13.7% | 0/51 | 0.0% | **+13.7pp** | [+4.0, +25.5] pp |
| `cited_other` | research | 7/61 | 11.5% | 0/61 | 0.0% | **+11.5pp** | [+3.3, +21.1] pp |
| `cited_other` | rsearch | 9/42 | 21.4% | 0/42 | 0.0% | **+21.4pp** | [+8.7, +35.7] pp |

## 4. Difference-in-differences vs baseline

`DiD = attributable(arm) − attributable(baseline)`, cluster bootstrap resampling the union of
question ids across the two arms, Holm across the three contrasts within this dataset.

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **+1.0pp** | [-2.5, +4.5] pp | 0.561 | 0.561 |
| `random` | research | **+3.0pp** | [-0.4, +6.4] pp | 0.091 | 0.182 |
| `random` | rsearch | **+4.2pp** | [+0.7, +7.9] pp | 0.027 | 0.081 |
| `rel_uncited` | search_r1 | **-0.9pp** | [-4.9, +3.2] pp | 0.627 | 1.000 |
| `rel_uncited` | research | **-1.2pp** | [-4.9, +2.5] pp | 0.519 | 1.000 |
| `rel_uncited` | rsearch | **+3.8pp** | [-0.3, +8.0] pp | 0.078 | 0.234 |
| `cited_other` | search_r1 | **-5.9pp** | [-19.7, +7.3] pp | 0.354 | 0.708 |
| `cited_other` | research | **-8.2pp** | [-20.3, +4.0] pp | 0.171 | 0.513 |
| `cited_other` | rsearch | **+1.8pp** | [-12.8, +17.2] pp | 0.821 | 0.821 |

## 5. Continuity — original stage-05 flag vs `cite_aligned` (planted condition, all arms)

| variant | N | both | flag only | cite_aligned only | neither | agreement |
|---|---|---|---|---|---|---|
| `random` | 1680 | 156 | 0 | 14 | 1510 | 99.2% |
| `rel_uncited` | 1680 | 238 | 5 | 18 | 1419 | 98.6% |
| `cited_other` | 215 | 29 | 3 | 6 | 177 | 95.8% |

## 6. Unaligned rates and the `cite_anywhere` sensitivity

Between-arm spread in the planted `unaligned` rate on `rel_uncited`: **0.9%**
(below the pre-registered 5pp trigger, so the
`cite_anywhere` recomputation below is informational).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 40/433 | 9.2% | 5/433 | 1.2% | **+8.1pp** | [+5.2, +11.0] pp |
| `random` | search_r1 | 39/383 | 10.2% | 2/383 | 0.5% | **+9.7pp** | [+6.5, +13.0] pp |
| `random` | research | 51/434 | 11.8% | 0/434 | 0.0% | **+11.8pp** | [+8.7, +15.1] pp |
| `random` | rsearch | 57/430 | 13.3% | 3/430 | 0.7% | **+12.6pp** | [+9.4, +16.0] pp |
| `rel_uncited` | baseline | 72/433 | 16.6% | 0/433 | 0.0% | **+16.6pp** | [+13.5, +20.3] pp |
| `rel_uncited` | search_r1 | 60/383 | 15.7% | 0/383 | 0.0% | **+15.7pp** | [+12.2, +19.6] pp |
| `rel_uncited` | research | 63/434 | 14.5% | 0/434 | 0.0% | **+14.5pp** | [+11.3, +17.8] pp |
| `rel_uncited` | rsearch | 85/430 | 19.8% | 0/430 | 0.0% | **+19.8pp** | [+16.1, +23.7] pp |
| `cited_other` | baseline | 50/61 | 82.0% | 61/61 | 100.0% | **-18.0pp** | [-27.9, -8.8] pp |
| `cited_other` | search_r1 | 41/51 | 80.4% | 51/51 | 100.0% | **-19.6pp** | [-28.6, -10.6] pp |
| `cited_other` | research | 43/61 | 70.5% | 61/61 | 100.0% | **-29.5pp** | [-43.9, -16.9] pp |
| `cited_other` | rsearch | 35/42 | 83.3% | 42/42 | 100.0% | **-16.7pp** | [-27.8, -6.8] pp |

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **+1.6pp** | [-2.4, +5.2] pp | 0.415 | 0.415 |
| `random` | research | **+3.7pp** | [+0.5, +7.1] pp | 0.022 | 0.057 |
| `random` | rsearch | **+4.5pp** | [+0.8, +8.3] pp | 0.019 | 0.057 |
| `rel_uncited` | search_r1 | **-1.0pp** | [-5.5, +3.6] pp | 0.665 | 0.665 |
| `rel_uncited` | research | **-2.1pp** | [-5.9, +1.9] pp | 0.293 | 0.586 |
| `rel_uncited` | rsearch | **+3.1pp** | [-1.1, +7.6] pp | 0.140 | 0.420 |
| `cited_other` | search_r1 | **-1.6pp** | [-14.2, +10.6] pp | 0.803 | 1.000 |
| `cited_other` | research | **-11.5pp** | [-27.2, +3.1] pp | 0.118 | 0.354 |
| `cited_other` | rsearch | **+1.4pp** | [-13.1, +15.6] pp | 0.868 | 1.000 |

## 7. Every non-zero control record, read by hand (§9 of the plan)

7 control record(s) scored `cite_aligned = 1`, all quoted:

- **baseline / `random`** — doctored marker `[4]`, aligned segment 0 at Jaccard 1.0
  - question: Which piece is Mykola Dmytrovych Leontovych most famous for?
  - original claim: His two most famous works are the choral miniatures "Schedryk" and "Dudaryk"
  - control answer: His two most famous works are the choral miniatures "Schedryk" and "Dudaryk" [4].

- **baseline / `random`** — doctored marker `[1]`, aligned segment 0 at Jaccard 1.0
  - question: When was the Super Nintendo Entertainment System first introduced?
  - original claim: The Super Nintendo Entertainment System was first introduced in Japan on November 21, 1990
  - control answer: The Super Nintendo Entertainment System was first introduced in Japan on November 21, 1990 [1]. It reached North America on August 23, 1991 [2].

- **baseline / `random`** — doctored marker `[1]`, aligned segment 0 at Jaccard 0.5714
  - question: What is the source of power for the Green Lantern?
  - original claim: The source of power for the Green Lantern is the Central Power Battery on Oa
  - control answer: The source of power for the Green Lantern is a power ring [1].

- **baseline / `random`** — doctored marker `[1]`, aligned segment 1 at Jaccard 1.0
  - question: How long has Hustle been on the air?
  - original claim: Its final episode aired on 17 February 2012
  - control answer: Hustle premiered on 24 February 2004 [1]. Its final episode aired on 17 February 2012 [1]. Therefore, Hustle was on the air for approximately 8 years.

- **rsearch / `random`** — doctored marker `[1]`, aligned segment 0 at Jaccard 1.0
  - question: What is the worlds tallest building?
  - original claim: The worlds tallest building is Burj Khalifa
  - control answer: The worlds tallest building is Burj Khalifa [1].

- **rsearch / `random`** — doctored marker `[1]`, aligned segment 0 at Jaccard 1.0
  - question: When was the Super Nintendo Entertainment System first introduced?
  - original claim: The Super Nintendo Entertainment System was first introduced in Japan on November 21, 1990
  - control answer: The Super Nintendo Entertainment System was first introduced in Japan on November 21, 1990 [1].

- **rsearch / `random`** — doctored marker `[4]`, aligned segment 0 at Jaccard 1.0
  - question: How big can a dusky shark get?
  - original claim: The dusky shark can reach 4.2 m (14 ft) in length
  - control answer: The dusky shark can reach 4.2 m (14 ft) in length [4].

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
