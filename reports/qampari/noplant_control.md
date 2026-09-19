# No-planting control — QAMPARI

**Work item:** W2 (project brief §9.4 limitation A3, §10 W2)
**Pre-registration:** fixed in writing, before this was scored
**Method + verification:** `reports/TECHNICAL_NOTES.md` §4–5
**Scorer:** `pipeline/05b_score_noplant_control.py` (outcome `cite_aligned`, cluster bootstrap
on `question_id`, 2000 replicates, seed 42)
**Control contexts:** `runs/qampari/adversarial_noplant/` — 3036 records,
prompt SHA-256 `00db51102ce17927…`
**Control answers:** `runs/qampari/postration_noplant/derived/`
**Planted answers:** `runs/qampari/postration/` — archived, unmodified

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
`run_noplant_control.py`), pulled into `runs/qampari/postration_noplant/04x_postration/`
and paired record-for-record with their planted twins (same order, same question ids, same
claims, same doctored marker — asserted, not assumed).

**All 12 cells are now measured.** The control base rate on `random` is 0–0.9%: the model
does occasionally cite a passage lifted from an unrelated question, with no span planted in
it at all, but rarely. Roughly 90% of the raw `random` rate survives as attributable.

## 2. Attrition, per arm per condition

| condition | variant | arm | N | questions | abstained | no markers | unaligned | span recovered |
|---|---|---|---|---|---|---|---|---|
| planted | `random` | baseline | 319 | 239 | 17 | 0 | 43 (13.5%) | 238 (74.6%) |
| planted | `random` | search_r1 | 283 | 186 | 23 | 0 | 55 (19.4%) | 204 (72.1%) |
| planted | `random` | research | 332 | 254 | 16 | 0 | 46 (13.9%) | 247 (74.4%) |
| planted | `random` | rsearch | 352 | 283 | 15 | 1 | 56 (15.9%) | 282 (80.1%) |
| planted | `rel_uncited` | baseline | 319 | 239 | 5 | 0 | 29 (9.1%) | 274 (85.9%) |
| planted | `rel_uncited` | search_r1 | 283 | 186 | 14 | 0 | 27 (9.5%) | 232 (82.0%) |
| planted | `rel_uncited` | research | 332 | 254 | 10 | 0 | 26 (7.8%) | 279 (84.0%) |
| planted | `rel_uncited` | rsearch | 352 | 283 | 8 | 0 | 26 (7.4%) | 309 (87.8%) |
| planted | `cited_other` | baseline | 123 | 42 | 1 | 0 | 11 (8.9%) | 101 (82.1%) |
| planted | `cited_other` | search_r1 | 128 | 43 | 13 | 0 | 17 (13.3%) | 95 (74.2%) |
| planted | `cited_other` | research | 123 | 38 | 2 | 0 | 9 (7.3%) | 101 (82.1%) |
| planted | `cited_other` | rsearch | 90 | 27 | 2 | 0 | 9 (10.0%) | 79 (87.8%) |
| control | `random` | baseline | 319 | 239 | 22 | 0 | 65 (20.4%) | 205 (64.3%) |
| control | `random` | search_r1 | 283 | 186 | 25 | 0 | 77 (27.2%) | 169 (59.7%) |
| control | `random` | research | 332 | 254 | 20 | 0 | 67 (20.2%) | 201 (60.5%) |
| control | `random` | rsearch | 352 | 283 | 15 | 0 | 66 (18.8%) | 256 (72.7%) |
| control | `rel_uncited` | baseline | 319 | 239 | 0 | 0 | 0 (0.0%) | 319 (100.0%) |
| control | `rel_uncited` | search_r1 | 283 | 186 | 0 | 0 | 0 (0.0%) | 283 (100.0%) |
| control | `rel_uncited` | research | 332 | 254 | 0 | 0 | 0 (0.0%) | 332 (100.0%) |
| control | `rel_uncited` | rsearch | 352 | 283 | 0 | 0 | 0 (0.0%) | 352 (100.0%) |
| control | `cited_other` | baseline | 123 | 42 | 0 | 0 | 0 (0.0%) | 123 (100.0%) |
| control | `cited_other` | search_r1 | 128 | 43 | 0 | 0 | 0 (0.0%) | 128 (100.0%) |
| control | `cited_other` | research | 123 | 38 | 0 | 0 | 0 (0.0%) | 123 (100.0%) |
| control | `cited_other` | rsearch | 90 | 27 | 0 | 0 | 0 (0.0%) | 90 (100.0%) |

The control's `unaligned = 0` and `span recovered = 100%` are not measurements: the control
answer *is* the answer the claim and the span were lifted from, so the claim's own sentence
aligns at Jaccard 1.0 and the span is a substring of it by construction.

## 3. Planted vs control — `cite_aligned`

The doctored slot's marker cited in the sentence that best matches the original clean-run
claim (token-set Jaccard ≥ 0.3).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 14/319 | 4.4% | 0/319 | 0.0% | **+4.4pp** | [+2.3, +6.8] pp |
| `random` | search_r1 | 16/283 | 5.7% | 0/283 | 0.0% | **+5.7pp** | [+3.2, +8.5] pp |
| `random` | research | 26/332 | 7.8% | 0/332 | 0.0% | **+7.8pp** | [+5.1, +10.7] pp |
| `random` | rsearch | 30/352 | 8.5% | 1/352 | 0.3% | **+8.2pp** | [+4.9, +11.8] pp |
| `rel_uncited` | baseline | 29/319 | 9.1% | 0/319 | 0.0% | **+9.1pp** | [+6.2, +12.0] pp |
| `rel_uncited` | search_r1 | 38/283 | 13.4% | 0/283 | 0.0% | **+13.4pp** | [+9.3, +17.9] pp |
| `rel_uncited` | research | 25/332 | 7.5% | 0/332 | 0.0% | **+7.5pp** | [+4.6, +10.5] pp |
| `rel_uncited` | rsearch | 43/352 | 12.2% | 0/352 | 0.0% | **+12.2pp** | [+8.8, +15.7] pp |
| `cited_other` | baseline | 16/123 | 13.0% | 0/123 | 0.0% | **+13.0pp** | [+7.3, +19.7] pp |
| `cited_other` | search_r1 | 20/128 | 15.6% | 0/128 | 0.0% | **+15.6pp** | [+9.8, +21.9] pp |
| `cited_other` | research | 15/123 | 12.2% | 0/123 | 0.0% | **+12.2pp** | [+6.0, +19.4] pp |
| `cited_other` | rsearch | 16/90 | 17.8% | 0/90 | 0.0% | **+17.8pp** | [+8.8, +27.8] pp |

## 4. Difference-in-differences vs baseline

`DiD = attributable(arm) − attributable(baseline)`, cluster bootstrap resampling the union of
question ids across the two arms, Holm across the three contrasts within this dataset.

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **+1.3pp** | [-1.6, +4.3] pp | 0.379 | 0.379 |
| `random` | research | **+3.4pp** | [+0.2, +6.7] pp | 0.038 | 0.114 |
| `random` | rsearch | **+3.8pp** | [-0.1, +7.9] pp | 0.054 | 0.114 |
| `rel_uncited` | search_r1 | **+4.3pp** | [-0.5, +9.7] pp | 0.089 | 0.267 |
| `rel_uncited` | research | **-1.6pp** | [-5.1, +2.3] pp | 0.415 | 0.415 |
| `rel_uncited` | rsearch | **+3.1pp** | [-1.4, +7.3] pp | 0.162 | 0.324 |
| `cited_other` | search_r1 | **+2.6pp** | [-6.0, +10.7] pp | 0.543 | 1.000 |
| `cited_other` | research | **-0.8pp** | [-9.1, +7.8] pp | 0.809 | 1.000 |
| `cited_other` | rsearch | **+4.8pp** | [-6.0, +15.5] pp | 0.415 | 1.000 |

## 5. Continuity — original stage-05 flag vs `cite_aligned` (planted condition, all arms)

| variant | N | both | flag only | cite_aligned only | neither | agreement |
|---|---|---|---|---|---|---|
| `random` | 1286 | 82 | 6 | 4 | 1194 | 99.2% |
| `rel_uncited` | 1286 | 128 | 13 | 7 | 1138 | 98.4% |
| `cited_other` | 464 | 53 | 22 | 14 | 375 | 92.2% |

## 6. Unaligned rates and the `cite_anywhere` sensitivity

Between-arm spread in the planted `unaligned` rate on `rel_uncited`: **2.2%**
(below the pre-registered 5pp trigger, so the
`cite_anywhere` recomputation below is informational).

| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |
|---|---|---|---|---|---|---|---|
| `random` | baseline | 21/319 | 6.6% | 0/319 | 0.0% | **+6.6pp** | [+4.0, +9.4] pp |
| `random` | search_r1 | 27/283 | 9.5% | 8/283 | 2.8% | **+6.7pp** | [+3.0, +10.5] pp |
| `random` | research | 32/332 | 9.6% | 3/332 | 0.9% | **+8.7pp** | [+5.5, +12.1] pp |
| `random` | rsearch | 40/352 | 11.4% | 5/352 | 1.4% | **+9.9pp** | [+6.2, +14.2] pp |
| `rel_uncited` | baseline | 42/319 | 13.2% | 0/319 | 0.0% | **+13.2pp** | [+9.5, +16.6] pp |
| `rel_uncited` | search_r1 | 56/283 | 19.8% | 0/283 | 0.0% | **+19.8pp** | [+15.1, +25.2] pp |
| `rel_uncited` | research | 38/332 | 11.4% | 0/332 | 0.0% | **+11.4pp** | [+7.9, +15.0] pp |
| `rel_uncited` | rsearch | 58/352 | 16.5% | 0/352 | 0.0% | **+16.5pp** | [+12.4, +20.6] pp |
| `cited_other` | baseline | 97/123 | 78.9% | 123/123 | 100.0% | **-21.1pp** | [-30.3, -13.2] pp |
| `cited_other` | search_r1 | 107/128 | 83.6% | 128/128 | 100.0% | **-16.4pp** | [-23.8, -9.6] pp |
| `cited_other` | research | 90/123 | 73.2% | 123/123 | 100.0% | **-26.8pp** | [-36.6, -18.0] pp |
| `cited_other` | rsearch | 71/90 | 78.9% | 90/90 | 100.0% | **-21.1pp** | [-31.4, -12.9] pp |

| variant | arm | DiD | 95% CI | boot p | Holm p |
|---|---|---|---|---|---|
| `random` | search_r1 | **+0.1pp** | [-4.0, +4.6] pp | 0.926 | 0.926 |
| `random` | research | **+2.2pp** | [-1.9, +6.4] pp | 0.298 | 0.596 |
| `random` | rsearch | **+3.4pp** | [-1.1, +7.7] pp | 0.153 | 0.459 |
| `rel_uncited` | search_r1 | **+6.6pp** | [+0.7, +12.8] pp | 0.034 | 0.102 |
| `rel_uncited` | research | **-1.7pp** | [-6.1, +2.7] pp | 0.473 | 0.473 |
| `rel_uncited` | rsearch | **+3.3pp** | [-2.1, +8.7] pp | 0.230 | 0.460 |
| `cited_other` | search_r1 | **+4.7pp** | [-4.1, +13.8] pp | 0.329 | 0.777 |
| `cited_other` | research | **-5.7pp** | [-16.3, +4.4] pp | 0.259 | 0.777 |
| `cited_other` | rsearch | **+0.0pp** | [-10.8, +10.4] pp | 0.986 | 0.986 |

## 7. Every non-zero control record, read by hand (§9 of the plan)

1 control record(s) scored `cite_aligned = 1`, all quoted:

- **rsearch / `random`** — doctored marker `[3]`, aligned segment 0 at Jaccard 1.0
  - question: What written form of media was published by Synapse Software?
  - original claim: "Encounter!"
  - control answer: "Encounter!" [3]

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
