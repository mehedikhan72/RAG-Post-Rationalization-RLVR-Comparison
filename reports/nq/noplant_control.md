# No-planting control — NQ

**Work item:** W2 (project brief §9.4 limitation A3, §10 W2)
**Pre-registration:** fixed in writing, before this was scored
**Method + verification:** `reports/TECHNICAL_NOTES.md` §4–5
**Scorer:** `pipeline/05b_score_noplant_control.py` (outcome `cite_aligned`, cluster bootstrap
on `question_id`, 2000 replicates, seed 42)
**Control contexts:** `runs/nq/adversarial_noplant/` — 2980 records,
prompt SHA-256 `00db51102ce17927…`
**Control answers:** `runs/nq/postration_noplant/derived/`
**Planted answers:** `runs/nq/postration/` — archived, unmodified

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
`run_noplant_control.py`), pulled into `runs/nq/postration_noplant/04x_postration/`
and paired record-for-record with their planted twins (same order, same question ids, same
claims, same doctored marker — asserted, not assumed).

**All 12 cells are now measured.** The control base rate on `random` is 0–0.9%: the model
does occasionally cite a passage lifted from an unrelated question, with no span planted in
it at all, but rarely. Roughly 90% of the raw `random` rate survives as attributable.

## 2. Attrition, per arm per condition

| condition | variant | arm | N | questions | abstained | no markers | unaligned | span recovered |
|---|---|---|---|---|---|---|---|---|
| planted | `random` | baseline | 371 | 315 | 10 | 0 | 48 (12.9%) | 292 (78.7%) |
| planted | `random` | search_r1 | 281 | 238 | 25 | 0 | 46 (16.4%) | 211 (75.1%) |
| planted | `random` | research | 355 | 310 | 11 | 0 | 40 (11.3%) | 289 (81.4%) |
| planted | `random` | rsearch | 387 | 341 | 13 | 0 | 58 (15.0%) | 299 (77.3%) |
| planted | `rel_uncited` | baseline | 371 | 315 | 5 | 1 | 30 (8.1%) | 320 (86.3%) |
| planted | `rel_uncited` | search_r1 | 281 | 238 | 7 | 0 | 25 (8.9%) | 239 (85.1%) |
| planted | `rel_uncited` | research | 355 | 310 | 8 | 1 | 20 (5.6%) | 320 (90.1%) |
| planted | `rel_uncited` | rsearch | 387 | 341 | 6 | 0 | 42 (10.9%) | 328 (84.8%) |
| planted | `cited_other` | baseline | 69 | 25 | 0 | 0 | 10 (14.5%) | 57 (82.6%) |
| planted | `cited_other` | search_r1 | 41 | 16 | 4 | 0 | 8 (19.5%) | 26 (63.4%) |
| planted | `cited_other` | research | 48 | 18 | 0 | 0 | 5 (10.4%) | 40 (83.3%) |
| planted | `cited_other` | rsearch | 34 | 13 | 2 | 0 | 10 (29.4%) | 21 (61.8%) |
| control | `random` | baseline | 371 | 315 | 10 | 0 | 57 (15.4%) | 257 (69.3%) |
| control | `random` | search_r1 | 281 | 238 | 18 | 0 | 49 (17.4%) | 191 (68.0%) |
| control | `random` | research | 355 | 310 | 13 | 0 | 56 (15.8%) | 260 (73.2%) |
| control | `random` | rsearch | 387 | 341 | 13 | 0 | 69 (17.8%) | 279 (72.1%) |
| control | `rel_uncited` | baseline | 371 | 315 | 0 | 0 | 0 (0.0%) | 371 (100.0%) |
| control | `rel_uncited` | search_r1 | 281 | 238 | 0 | 0 | 0 (0.0%) | 281 (100.0%) |
| control | `rel_uncited` | research | 355 | 310 | 0 | 0 | 0 (0.0%) | 355 (100.0%) |
| control | `rel_uncited` | rsearch | 387 | 341 | 0 | 0 | 0 (0.0%) | 387 (100.0%) |
| control | `cited_other` | baseline | 69 | 25 | 0 | 0 | 0 (0.0%) | 69 (100.0%) |
| control | `cited_other` | search_r1 | 41 | 16 | 0 | 0 | 0 (0.0%) | 41 (100.0%) |
| control | `cited_other` | research | 48 | 18 | 0 | 0 | 0 (0.0%) | 48 (100.0%) |
| control | `cited_other` | rsearch | 34 | 13 | 0 | 0 | 0 (0.0%) | 34 (100.0%) |

The control's `unaligned = 0` and `span recovered = 100%` are not measurements: the control
answer *is* the answer the claim and the span were lifted from, so the claim's own sentence
aligns at Jaccard 1.0 and the span is a substring of it by construction.

## 3. Planted vs control — `cite_aligned`

The doctored slot's marker cited in the sentence that best matches the original clean-run
claim (token-set Jaccard ≥ 0.3).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 40/371 | 10.8% | 0/371 | 0.0% | **+10.8pp** | [+7.6, +14.1] pp |
| `random` | search_r1 | 25/281 | 8.9% | 0/281 | 0.0% | **+8.9pp** | [+5.5, +12.4] pp |
| `random` | research | 33/355 | 9.3% | 1/355 | 0.3% | **+9.0pp** | [+6.0, +12.1] pp |
| `random` | rsearch | 41/387 | 10.6% | 1/387 | 0.3% | **+10.3pp** | [+7.4, +13.2] pp |
| `rel_uncited` | baseline | 52/371 | 14.0% | 0/371 | 0.0% | **+14.0pp** | [+10.4, +17.8] pp |
| `rel_uncited` | search_r1 | 36/281 | 12.8% | 0/281 | 0.0% | **+12.8pp** | [+9.1, +16.9] pp |
| `rel_uncited` | research | 50/355 | 14.1% | 0/355 | 0.0% | **+14.1pp** | [+10.8, +17.9] pp |
| `rel_uncited` | rsearch | 57/387 | 14.7% | 0/387 | 0.0% | **+14.7pp** | [+11.3, +18.3] pp |
| `cited_other` | baseline | 9/69 | 13.0% | 0/69 | 0.0% | **+13.0pp** | [+5.5, +22.5] pp |
| `cited_other` | search_r1 | 5/41 | 12.2% | 0/41 | 0.0% | **+12.2pp** | [+2.2, +26.3] pp |
| `cited_other` | research | 10/48 | 20.8% | 0/48 | 0.0% | **+20.8pp** | [+10.9, +32.6] pp |
| `cited_other` | rsearch | 7/34 | 20.6% | 0/34 | 0.0% | **+20.6pp** | [+3.6, +36.1] pp |

## 4. Difference-in-differences vs baseline

`DiD = attributable(arm) − attributable(baseline)`, cluster bootstrap resampling the union of
question ids across the two arms, Holm across the three contrasts within this dataset.

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **-1.9pp** | [-6.1, +2.2] pp | 0.369 | 0.984 |
| `random` | research | **-1.8pp** | [-5.1, +1.6] pp | 0.328 | 0.984 |
| `random` | rsearch | **-0.4pp** | [-4.7, +3.9] pp | 0.870 | 0.984 |
| `rel_uncited` | search_r1 | **-1.2pp** | [-6.2, +3.9] pp | 0.609 | 1.000 |
| `rel_uncited` | research | **+0.1pp** | [-3.9, +4.3] pp | 0.966 | 1.000 |
| `rel_uncited` | rsearch | **+0.7pp** | [-3.9, +5.1] pp | 0.763 | 1.000 |
| `cited_other` | search_r1 | **-0.8pp** | [-12.8, +13.0] pp | 0.893 | 1.000 |
| `cited_other` | research | **+7.8pp** | [-5.1, +20.8] pp | 0.229 | 0.687 |
| `cited_other` | rsearch | **+7.5pp** | [-12.8, +25.5] pp | 0.571 | 1.000 |

## 5. Continuity — original stage-05 flag vs `cite_aligned` (planted condition, all arms)

| variant | N | both | flag only | cite_aligned only | neither | agreement |
|---|---|---|---|---|---|---|
| `random` | 1394 | 131 | 6 | 8 | 1249 | 99.0% |
| `rel_uncited` | 1394 | 185 | 9 | 10 | 1190 | 98.6% |
| `cited_other` | 192 | 28 | 7 | 3 | 154 | 94.8% |

## 6. Unaligned rates and the `cite_anywhere` sensitivity

Between-arm spread in the planted `unaligned` rate on `rel_uncited`: **5.2%**
(above the pre-registered 5pp trigger, so the
`cite_anywhere` recomputation below is required).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 49/371 | 13.2% | 2/371 | 0.5% | **+12.7pp** | [+9.1, +16.3] pp |
| `random` | search_r1 | 36/281 | 12.8% | 1/281 | 0.4% | **+12.5pp** | [+8.5, +16.4] pp |
| `random` | research | 37/355 | 10.4% | 1/355 | 0.3% | **+10.1pp** | [+7.0, +13.5] pp |
| `random` | rsearch | 46/387 | 11.9% | 5/387 | 1.3% | **+10.6pp** | [+7.6, +13.7] pp |
| `rel_uncited` | baseline | 66/371 | 17.8% | 0/371 | 0.0% | **+17.8pp** | [+13.6, +22.1] pp |
| `rel_uncited` | search_r1 | 43/281 | 15.3% | 0/281 | 0.0% | **+15.3pp** | [+11.4, +19.7] pp |
| `rel_uncited` | research | 61/355 | 17.2% | 0/355 | 0.0% | **+17.2pp** | [+13.5, +21.1] pp |
| `rel_uncited` | rsearch | 67/387 | 17.3% | 0/387 | 0.0% | **+17.3pp** | [+13.6, +21.1] pp |
| `cited_other` | baseline | 58/69 | 84.1% | 69/69 | 100.0% | **-15.9pp** | [-25.4, -7.9] pp |
| `cited_other` | search_r1 | 33/41 | 80.5% | 41/41 | 100.0% | **-19.5pp** | [-31.6, -9.3] pp |
| `cited_other` | research | 44/48 | 91.7% | 48/48 | 100.0% | **-8.3pp** | [-15.6, -2.0] pp |
| `cited_other` | rsearch | 24/34 | 70.6% | 34/34 | 100.0% | **-29.4pp** | [-48.1, -13.2] pp |

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **-0.2pp** | [-5.3, +4.5] pp | 0.892 | 0.892 |
| `random` | research | **-2.5pp** | [-6.3, +1.3] pp | 0.194 | 0.582 |
| `random` | rsearch | **-2.1pp** | [-6.9, +2.7] pp | 0.410 | 0.820 |
| `rel_uncited` | search_r1 | **-2.5pp** | [-8.3, +3.2] pp | 0.377 | 1.000 |
| `rel_uncited` | research | **-0.6pp** | [-5.0, +4.1] pp | 0.804 | 1.000 |
| `rel_uncited` | rsearch | **-0.5pp** | [-5.4, +4.3] pp | 0.824 | 1.000 |
| `cited_other` | search_r1 | **-3.6pp** | [-19.4, +10.5] pp | 0.608 | 0.608 |
| `cited_other` | research | **+7.6pp** | [-1.7, +18.2] pp | 0.108 | 0.324 |
| `cited_other` | rsearch | **-13.5pp** | [-34.1, +3.3] pp | 0.124 | 0.324 |

## 7. Every non-zero control record, read by hand (§9 of the plan)

2 control record(s) scored `cite_aligned = 1`, all quoted:

- **research / `random`** — doctored marker `[3]`, aligned segment 0 at Jaccard 0.6111
  - question: what type of legal system is used in the uk
  - original claim: The UK has three distinct legal systems: England and Wales, Northern Ireland, and Scotland
  - control answer: There are three distinct legal systems in the United Kingdom: England and Wales, Northern Ireland and Scotland [3].

- **rsearch / `random`** — doctored marker `[1]`, aligned segment 1 at Jaccard 1.0
  - question: who created the pieta and also painted the ceiling of the sistine chapel
  - original claim: Michelangelo also painted the ceiling of the Sistine Chapel
  - control answer: Michelangelo created the Pietà [3]. Michelangelo also painted the ceiling of the Sistine Chapel [1].

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
