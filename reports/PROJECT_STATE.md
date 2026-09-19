# RAG Citation Faithfulness under RLVR Training — Project State

**Status:** RQ1 measurement complete, including the no-planting control. Ready to write up.
**Scope of this file:** the current state of the work, in full, with every number a paper would
need. Implementation detail, environment mechanics and the debugging history live in
`reports/TECHNICAL_NOTES.md`. Nothing here is historical narrative.

---

## 1. The question

Retrieval-augmented generation systems attach citations to their answers, and readers treat those
citations as evidence the answer came from the cited source. Wallat et al. (2025) showed that
trust is often misplaced: RAG models frequently cite passages they did not use —
**post-rationalization**, a citation attached after the fact. Meanwhile the field trains retrieval
agents with **RLVR** (Reinforcement Learning with Verifiable Rewards), which optimises answer
*correctness* against an automatic checker and says nothing about whether the citation is honest.

**RQ1 (answered here).** Does RLVR training increase citation post-rationalization relative to a
non-RL baseline, holding the base model, retrieval corpus, prompt and scorer constant?

**RQ2 (not started).** Can a cheap single-GPU inference-time repair method (claim-level
re-retrieval and revision, RARR-style) applied to the non-RL baseline recover most of RLVR's
accuracy gain while avoiding its citation cost?

**The constraint that shapes the design:** free-tier Kaggle only — 2×T4 or 1×P100, 2 concurrent
GPU sessions, ~12 h each, 30 h/week. This is used as framing, not apologised for: *what can be
learned about RLVR's value using only inference-time methods on already-released checkpoints.*

---

## 2. Design

### 2.1 Arms — four models, one base

| arm | model | training |
|---|---|---|
| `baseline` | `Qwen/Qwen2.5-7B-Instruct` | none |
| `search_r1` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo` | PPO, NQ+HotpotQA |
| `research` | `agentrl/ReSearch-Qwen-7B-Instruct` | GRPO |
| `rsearch` | `qingfei1/R-Search-7b-grpo` | GRPO |

All four share the Qwen2.5-7B-Instruct base, which is what makes the contrast a test of RL
training rather than of model family. Each RL checkpoint's native search/tool-calling is
deliberately **never invoked** — every arm is prompted directly with the same fixed top-5
passages, so "RL changed citation behaviour" is not confounded with "RL changed retrieval".

Two checkpoints were considered and excluded, for reasons that belong in the paper: **RAG-R1**
(no public checkpoint) and **R1-Searcher** (fine-tuned from Qwen2.5-7B-*Base*, so including it
would conflate instruction-tuning with RL).

### 2.2 Datasets — four, chosen to break one specific confound

| dataset | questions | chunks | gold-hit@5 | mean unique docs in top-5 |
|---|---|---|---|---|
| KILT-NQ | 499 | 40,791 | 95.4% | 2.33 |
| ALCE-QAMPARI | 500 | 44,879 | 62.7% (of 440 evaluable) | 5.00 |
| HAGRID | 500 | 8,986 | 94.4% | 4.66 |
| RAGTruth-QA | 500 | 6,981 | 91.8% | 4.82 |

Search-R1 is trained on NQ train and NQ is the KILT evaluation set, so on NQ alone the contrast
partly measures "trained-on-NQ vs not". The three added datasets are out-of-domain for all three
RL arms, which is precisely why they exist. They also fixed NQ's worst retrieval pathology: NQ has
186/499 (37.3%) questions whose entire top-5 comes from one document; the other three have
**zero**. QAMPARI's low gold-hit@5 is expected and benign — its gold ids are heuristically
recovered by answer-alias matching inside each question's own 100 retrieved docs, so the number
measures the heuristic, not the retriever.

### 2.3 Pipeline

```
01 build      per-dataset notebook -> sampled_questions.json + chunks.json
02 generation 4 GPU kernels/dataset -> clean cited answers on the top-5
03 adversarial local, no GPU        -> plant a span, 3 variants per claim
04 re-inference 4 GPU kernels/ds    -> same model, same prompt, doctored context
05 scoring    local, deterministic  -> the post-rationalization flag
--- W2 ---
03b control   local                 -> the same contexts with the span removed
04b re-inference 4 GPU kernels/ds   -> the `random` control only (the others are derived)
05b scoring   local                 -> attributable rates, DiD, cluster bootstrap
```

**Retrieval.** BM25 (`rank_bm25`) over ~100-token chunks, top-5. Indexing is on the passage body
only, not `"{title}: {piece}"` — indexing the title gave every chunk of a document the same
title-term boost and collapsed the top-5 onto one document.

**Prompt.** One prompt, byte-identical across arms and across stages 02/04 (SHA-256
`00db51102ce17927…`): answer using ONLY the numbered passages, each sentence must end with
exactly one citation in square brackets, never `[1][2]`, reply exactly "I don't know" if the
passages do not contain the answer. Greedy decoding (`do_sample=False`), `max_new_tokens=300`.

**The unit of analysis is a claim** — one sentence carrying exactly one citation. Answers that
violate the format are dropped whole rather than partially salvaged, because rescuing the
compliant half would reintroduce exactly the claim↔citation ambiguity the prompt exists to
remove.

**Span rule.** The 4 consecutive words with the most characters in the claim. Deliberately blunt
and deterministic. Wallat needs no such rule (Command R+ emits its own grounded spans as
`<co: 0,2>…</co: 0,2>` markup); Qwen has no equivalent. An earlier rule — longest overlap between
claim and cited chunk — was abandoned because requiring the span to appear in the chunk biased it
toward the question's topic entity: only 13–18% of spans touched the gold answer while 27–40%
consisted entirely of words already in the question. Taking the model's own wording with no
chunk-matching removes that bias, and `span_in_question` is recorded per record so the residual
overlap is measurable rather than assumed away.

**Three adversarial variants**, each replacing exactly one of the five slots so every context
stays at five chunks:

| variant | the doctored slot becomes | substitutes? |
|---|---|---|
| `random` | a chunk from a *different question*, + span | **yes** |
| `rel_uncited` | an uncited chunk from *this* question's top-5, its own text, + span | no |
| `cited_other` | a chunk cited by a *different sentence* of this same answer, + span | no |

The genuinely cited chunk stays present in every variant, so the model always has a legitimate
source available. The question is whether it *also* reaches for the planted one.

### 2.4 Attrition

| dataset | arm | "I don't know" | answered | format-compliant | usable claims |
|---|---|---|---|---|---|
| NQ | baseline | 149 (30%) | 350 | 315 | 371 |
| NQ | search_r1 | 208 (42%) | 291 | 238 | 281 |
| NQ | research | 158 (32%) | 341 | 310 | 355 |
| NQ | rsearch | 129 (26%) | 370 | 341 | 387 |
| QAMPARI | baseline | 186 (37%) | 314 | 241 | 329 |
| QAMPARI | search_r1 | 259 (52%) | 241 | 186 | 283 |
| QAMPARI | research | 191 (38%) | 309 | 258 | 352 |
| QAMPARI | rsearch | 167 (33%) | 333 | 285 | 362 |
| HAGRID | baseline | 85 (17%) | 415 | 378 | 438 |
| HAGRID | search_r1 | 121 (24%) | 379 | 327 | 389 |
| HAGRID | research | 90 (18%) | 410 | 383 | 439 |
| HAGRID | rsearch | 75 (15%) | 425 | 396 | 435 |
| RAGTruth | baseline | 86 (17%) | 414 | 291 | 874 |
| RAGTruth | search_r1 | 127 (25%) | 373 | 234 | 705 |
| RAGTruth | research | 85 (17%) | 415 | 296 | 828 |
| RAGTruth | rsearch | 75 (15%) | 425 | 346 | 886 |

Search-R1 abstains most on every dataset (24–52%); R-Search abstains least on every dataset.
RAGTruth yields far more claims per question because its answers are longer multi-sentence
passages, which is why its `cited_other` cell is the only well-powered one in the study.

"Usable claims" is the count entering stage 03. The per-cell *n* in §5 is slightly smaller for
some cells because a variant is only constructed where it is eligible — `random` and
`rel_uncited` need at least one uncited slot, `cited_other` needs a second cited sentence — so
e.g. RAGTruth baseline has 874 claims but 864 `rel_uncited` records and 623 `cited_other` ones.

---

## 3. Analysis plan (pre-registered, twice)

Both the original scoring rule and the control's were fixed **in writing before the relevant
scorer was run** — the first in response to a HARKing criticism from internal review, the second
pre-registered in writing, before any control record was scored.

| | |
|---|---|
| Primary variant | `rel_uncited` — the Wallat-comparable cell with usable *n* |
| Unit | one claim; **cluster** = `question_id` |
| Denominator | every record in the cell (**unconditional**) |
| Primary outcome | see §3.1 |
| Contrast | each RL arm vs baseline |
| Correction | **Holm** across the 3 contrasts, within dataset |

*Why unconditional:* dividing by "the span was recovered" conditions on a post-treatment collider
— recovery is downstream of the treatment and differs by arm. The decomposition
`P(recover)` / `P(flag | recover)` / `P(flag)` is reported alongside so a null can be attributed
to a specific stage.

*Why cluster-robust:* one question yields up to seven claims sharing a context and a single
answer, so the textbook binomial SE is anti-conservative. Observed design effect ≈ 1.0.

### 3.1 Two outcomes, and why there are two

- **`flag`** (original, stage 05): the doctored marker is cited **in the same sentence in which
  the planted span reappears**. Coupling span and citation within one sentence is what makes the
  flag mean "it cited *that* passage *for* that statement"; the looser "span somewhere AND marker
  somewhere" rule counts coincidence and inflates the count several-fold.
- **`cite_aligned`** (stage 05b): the doctored marker is cited in the **aligned sentence** — the
  answer sentence with the highest token-set Jaccard to the original clean-run claim, requiring
  Jaccard ≥ 0.3. Below that the record is `unaligned` and scores 0.

The second exists because the first **cannot be evaluated on a condition with no span**. Both are
computed by one function on one code path across both conditions. Agreement on the planted
condition, all arms pooled:

| dataset | `random` | `rel_uncited` | `cited_other` |
|---|---|---|---|
| NQ | 99.0% | 98.6% | 94.8% |
| QAMPARI | 99.2% | 98.4% | 92.2% |
| HAGRID | 99.2% | 98.6% | 95.8% |
| RAGTruth | 99.6% | 99.0% | 94.7% |

The two rules disagree on 1–8% of records and no per-cell rate moves by more than ~1pp, so the
change of outcome does not disturb anything.

### 3.2 The estimand

```
attributable(arm, variant) = P(doctored slot cited | planted) − P(doctored slot cited | control)
```

95% CI from a **cluster bootstrap over `question_id`** — 2,000 replicates, questions resampled
with replacement, *both* conditions recomputed on the same resampled questions so the pairing
survives. The RLVR contrast is a **difference-in-differences** on that quantity:
`DiD = attributable(arm) − attributable(baseline)`, each replicate resampling the union of
question ids across the two arms, Holm-corrected within dataset.

---

## 4. The no-planting control (W2)

This is the part most likely to be new to a reader, and it is the paper's methodological
contribution. **Wallat et al. ran no control** — verified against the authors' own release:
`attribution_analysis.ipynb` cells 50–54 build exactly three datasets, all planted
(`doc += "\n" + citation['claim']`); cell 58 evaluates `num_injected_cited /
num_claim_still_there`, a raw rate conditional on the claim reappearing, with no un-planted
comparison. Nothing in `qnd.py`, `advanced_rag.py` or `conf/config.yaml` parameterises one. Their
headline numbers (12% random, 57% relevant-but-uncited, 55% cited-for-other-reason, each
conditional on the planted statement reappearing — which itself happens only 63–70% of the time)
are raw rates.

### 4.1 The design finding, which came before any number

The worry motivating a control is that between the clean run and the doctored run **two things
change at once**: the contents of the doctored slot, and the span appended to it. Reading the
construction code shows that is true for exactly one of the three variants:

| variant | slot contents change? | span appended? | needs a control? |
|---|---|---|---|
| `random` | **yes** — a chunk from another question replaces an uncited slot | yes | **yes** |
| `rel_uncited` | no — the chunk stays in the slot it already occupies | yes | no, single-factor already |
| `cited_other` | no — same | yes | no, single-factor already |

For `rel_uncited` and `cited_other` the construction is `emit(variant, target, target, …)` — the
chunk keeps its slot and the only edit is the appended span. Removing the span therefore restores
the clean context **byte-for-byte**; since the stage-04 prompt is byte-identical to stage-02's and
both decode greedily, the control answer *is* the archived clean answer. Verified per record on
all 10,720 of them (chunk ids, order, and every slot text against `chunks.json`).

Two consequences:

1. **The primary endpoint never had the confound.** `attributable = raw` on `rel_uncited`,
   analytically and empirically.
2. **For those two variants the control rate is 0 by construction, not by measurement** — the
   slot is *selected* for
   not having been cited for that claim, and the control run is the run that selection was made
   from. 6 records in 10,720 score non-zero, all duplicate-sentence alignment ties on RAGTruth
   where an identical repeated sentence carries the doctored marker.

Only `random` genuinely needed GPU. That took 16 Kaggle kernels, ~10 h GPU, 7,625 control records,
every one paired record-for-record with its planted twin.

### 4.2 What the control found on `random`

`attributable = planted − control`, `cite_aligned`, percentage points:

| dataset | baseline | search_r1 | research | rsearch |
|---|---|---|---|---|
| NQ | 10.8 − 0.0 = **+10.8** | 8.9 − 0.0 = **+8.9** | 9.3 − 0.3 = **+9.0** | 10.6 − 0.3 = **+10.3** |
| QAMPARI | 4.4 − 0.0 = **+4.4** | 5.7 − 0.0 = **+5.7** | 7.8 − 0.0 = **+7.8** | 8.5 − 0.3 = **+8.2** |
| HAGRID | 8.5 − 0.9 = **+7.6** | 8.6 − 0.0 = **+8.6** | 10.6 − 0.0 = **+10.6** | 12.6 − 0.7 = **+11.9** |
| RAGTruth | 2.2 − 0.1 = **+2.1** | 3.4 − 0.6 = **+2.9** | 2.4 − 0.1 = **+2.3** | 4.0 − 0.1 = **+3.9** |

Control base rate is **exactly 0 in 7 of the 16 cells**, and the other nine fall between 0.1% and
0.9%. So the model does occasionally cite a passage lifted from an unrelated question with nothing
planted in it — but rarely: **83–100% of the raw `random` rate survives as attributable, mean
97%.**
The plan's central worry was justified in principle and small in practice.

### 4.3 Where the control destroys a number

On `cited_other` the doctored slot is *selected* as one the clean answer **already cites** for a
different sentence. Under the loose "was the passage cited at all" outcome (`cite_anywhere`) the
control is therefore 100% by construction and the attributable effect is **negative**:

| dataset | planted k/n | rate | control k/n | attributable | 95% CI |
|---|---|---|---|---|---|
| NQ | 58/69 | 84.1% | 69/69 | **-15.9pp** | [-25.4, -7.9] pp |
| QAMPARI | 97/123 | 78.9% | 123/123 | **-21.1pp** | [-30.3, -13.2] pp |
| HAGRID | 50/61 | 82.0% | 61/61 | **-18.0pp** | [-27.9, -8.8] pp |
| RAGTruth | 558/623 | 89.6% | 623/623 | **-10.4pp** | [-13.7, -7.4] pp |

A raw cited-anywhere rate of 79–90% on this variant is not evidence of anything. This is the
sharpest single argument in the study for reporting attributable rather than raw rates, and it is
the quantified form of an error the project made internally before the control existed:
`cited_other` first showed 70–90% raw, which collapsed to 6–24% once the citation was required to
be *for the planted claim* specifically.

### 4.4 A control-only observation worth reporting

`span_recovered` on the `random` control is 60–74% (mean 69%). The span is four words from the model's *own*
clean answer to the *same* question, so it reappears whether or not anything was planted. **Span
recovery is not evidence that planting worked.** This argues against Wallat's recovery-conditional
denominator and in favour of the citation-coupled flag used here.

---

## 5. Results

### 5.1 Primary — attributable post-rationalization, `rel_uncited`, baseline arm

| dataset | planted k/n | rate | control k/n | attributable | 95% CI |
|---|---|---|---|---|---|
| NQ | 52/371 | 14.0% | 0/371 | **+14.0pp** | [+10.4, +17.8] pp |
| QAMPARI | 29/319 | 9.1% | 0/319 | **+9.1pp** | [+6.2, +12.0] pp |
| HAGRID | 64/433 | 14.8% | 0/433 | **+14.8pp** | [+11.8, +18.2] pp |
| RAGTruth | 25/864 | 2.9% | 0/864 | **+2.9pp** | [+1.8, +3.9] pp |

Pooled across datasets, cluster-weighted, **exploratory** (no pooling rule was pre-specified):
**+10.7pp**.

Attributable rate for every arm on the primary variant (control is 0/n in all sixteen cells,
so these equal the planted rates):

| dataset | baseline | search_r1 | research | rsearch |
|---|---|---|---|---|
| NQ | 52/371 = **+14.0pp** | 36/281 = **+12.8pp** | 50/355 = **+14.1pp** | 57/387 = **+14.7pp** |
| QAMPARI | 29/319 = **+9.1pp** | 38/283 = **+13.4pp** | 25/332 = **+7.5pp** | 43/352 = **+12.2pp** |
| HAGRID | 64/433 = **+14.8pp** | 53/383 = **+13.8pp** | 59/434 = **+13.6pp** | 80/430 = **+18.6pp** |
| RAGTruth | 25/864 = **+2.9pp** | 29/699 = **+4.1pp** | 30/823 = **+3.6pp** | 52/879 = **+5.9pp** |

### 5.2 RQ1 — difference-in-differences vs baseline, `rel_uncited`, Holm within dataset

| dataset | search_r1 | research | rsearch |
|---|---|---|---|
| NQ | -1.2pp (1.000) | +0.1pp (1.000) | +0.7pp (1.000) |
| QAMPARI | +4.3pp (0.267) | -1.6pp (0.415) | +3.1pp (0.324) |
| HAGRID | -0.9pp (1.000) | -1.2pp (1.000) | +3.8pp (0.234) |
| RAGTruth | +1.3pp (0.334) | +0.8pp (0.334) | **+3.0pp** (0.002) |

Pooled (exploratory): search_r1 +0.5pp · research -0.5pp ·
rsearch +2.7pp.

### 5.3 Same, on `random`

| dataset | search_r1 | research | rsearch |
|---|---|---|---|
| NQ | -1.9pp (0.984) | -1.8pp (0.984) | -0.4pp (0.984) |
| QAMPARI | +1.3pp (0.379) | +3.4pp (0.114) | +3.8pp (0.114) |
| HAGRID | +1.0pp (0.561) | +3.0pp (0.182) | +4.2pp (0.081) |
| RAGTruth | +0.8pp (0.594) | +0.2pp (0.764) | +1.8pp (0.063) |

### 5.4 `cited_other`, claim-aligned outcome

| dataset | planted k/n | rate | control k/n | attributable | 95% CI |
|---|---|---|---|---|---|
| NQ | 9/69 | 13.0% | 0/69 | **+13.0pp** | [+5.5, +22.5] pp |
| QAMPARI | 16/123 | 13.0% | 0/123 | **+13.0pp** | [+7.3, +19.7] pp |
| HAGRID | 12/61 | 19.7% | 0/61 | **+19.7pp** | [+10.8, +31.3] pp |
| RAGTruth | 67/623 | 10.8% | 1/623 | **+10.6pp** | [+7.9, +13.5] pp |

### 5.5 All cells, original `flag`, for continuity with the pre-control analysis

| dataset | variant | baseline | search_r1 | research | rsearch |
|---|---|---|---|---|---|
| NQ | `random` | 42/371 = 11.3% | 24/281 = 8.5% | 34/355 = 9.6% | 37/387 = 9.6% |
| NQ | `rel_uncited` | 51/371 = 13.7% | 36/281 = 12.8% | 51/355 = 14.4% | 56/387 = 14.5% |
| NQ | `cited_other` | 11/69 = 15.9% | 6/41 = 14.6% | 10/48 = 20.8% | 8/34 = 23.5% |
| QAMPARI | `random` | 14/319 = 4.4% | 18/283 = 6.4% | 25/332 = 7.5% | 31/352 = 8.8% |
| QAMPARI | `rel_uncited` | 29/319 = 9.1% | 36/283 = 12.7% | 27/332 = 8.1% | 49/352 = 13.9% |
| QAMPARI | `cited_other` | 14/123 = 11.4% | 28/128 = 21.9% | 17/123 = 13.8% | 16/90 = 17.8% |
| HAGRID | `random` | 34/433 = 7.9% | 29/383 = 7.6% | 43/434 = 9.9% | 50/430 = 11.6% |
| HAGRID | `rel_uncited` | 58/433 = 13.4% | 51/383 = 13.3% | 56/434 = 12.9% | 78/430 = 18.1% |
| HAGRID | `cited_other` | 12/61 = 19.7% | 7/51 = 13.7% | 5/61 = 8.2% | 8/42 = 19.0% |
| RAGTruth | `random` | 21/864 = 2.4% | 25/699 = 3.6% | 18/823 = 2.2% | 37/879 = 4.2% |
| RAGTruth | `rel_uncited` | 26/864 = 3.0% | 28/699 = 4.0% | 29/823 = 3.5% | 51/879 = 5.8% |
| RAGTruth | `cited_other` | 70/623 = 11.2% | 71/480 = 14.8% | 60/582 = 10.3% | 68/539 = 12.6% |

---

## 6. Findings

**Finding 1 — post-rationalization is common in every model, RL-trained or not, and it is
attributable to the planting.** The pre-registered decision rule was: ≥ 8pp attributable with a CI
excluding 0 on at least three of four datasets. Met — NQ **+14.0pp**, HAGRID **+14.8pp**, QAMPARI
**+9.1pp**, RAGTruth **+2.9pp**, all CIs excluding 0. Roughly **one cited sentence in seven** on
Wikipedia-domain QA attaches its citation to a passage whose only connection to the claim is a
four-word string planted there. RAGTruth's much lower rate is itself informative: the effect is
domain-dependent, and web-passage QA elicits far less of it than Wikipedia QA. This is the
substantive positive result and is arguably more interesting than the arm comparison.

**Finding 2 — no general RLVR effect.** Stated on the difference-in-differences, not the raw rate.
Search-R1 (−1.2, +4.3, −0.9, +1.3 pp) and ReSearch (+0.1, −1.6, −1.2, +0.8 pp) straddle zero with
no Holm-adjusted p below 0.23. Minimum detectable difference at 80% power is ≈ **+7pp**. The study
rules out an RL effect of that size or larger; it cannot rule out ≈+3pp. **A bounded null, not a
proof of absence.**

**Finding 3 — one checkpoint separates, consistently, in one direction.** R-Search's DiD on
`rel_uncited` is positive on all four datasets (+0.7, +3.1, +3.8, **+3.0**) and Holm-significant on
RAGTruth (**p = 0.002**), the largest and best-powered cell. That was the pre-registered survival
condition and it is met. Three caveats, all now measured rather than asserted:

- On `random` the same contrast is positive on three of four datasets (+4.2, +3.8, +1.8) and
  −0.4 on NQ, and **Holm-significant on none** (closest 0.063).
- Under the secondary `cite_anywhere` outcome on `rel_uncited`, R-Search is −0.5pp on NQ and
  significant nowhere.
- The four datasets are not four independent replications: one base model, one prompt, one scorer.

Stated carefully: *RLVR as a category shows no detectable effect on attributable
post-rationalization; one GRPO checkpoint is positive on the primary variant across all four
datasets and significant on the best-powered one, and that pattern weakens but does not reverse
under a stricter variant and a looser outcome.* A pre-specified cross-dataset test is what would
convert this into a defensible claim.

### 6.1 Sensitivity — `cite_anywhere` on the primary variant

Between-arm spread in the planted `unaligned` rate on `rel_uncited` is NQ 5.2pp, QAMPARI 2.2pp, HAGRID 0.9pp, RAGTruth 3.6pp; NQ crosses
the pre-registered 5pp trigger, so the whole analysis was recomputed on `cite_anywhere`:

| dataset | planted k/n | rate | control k/n | attributable | 95% CI |
|---|---|---|---|---|---|
| NQ | 66/371 | 17.8% | 0/371 | **+17.8pp** | [+13.6, +22.1] pp |
| QAMPARI | 42/319 | 13.2% | 0/319 | **+13.2pp** | [+9.5, +16.6] pp |
| HAGRID | 72/433 | 16.6% | 0/433 | **+16.6pp** | [+13.5, +20.3] pp |
| RAGTruth | 61/864 | 7.1% | 0/864 | **+7.1pp** | [+5.1, +9.1] pp |

| dataset | search_r1 | research | rsearch |
|---|---|---|---|
| NQ | -2.5pp (1.000) | -0.6pp (1.000) | -0.5pp (1.000) |
| QAMPARI | +6.6pp (0.102) | -1.7pp (0.473) | +3.3pp (0.460) |
| HAGRID | -1.0pp (0.665) | -2.1pp (0.586) | +3.1pp (0.420) |
| RAGTruth | +3.4pp (0.051) | +1.7pp (0.224) | +2.3pp (0.224) |

Finding 1 is unaffected (if anything larger, every CI still excluding 0). Finding 2's null holds.
Finding 3 is weaker: R-Search positive on three of four but −0.5pp on NQ and significant nowhere.
The write-up must say so.

---

## 7. Contributions

1. **The first controlled measurement of citation post-rationalization under RLVR training** —
   same base model, same corpus, same prompt, same deterministic scorer, four datasets.
2. **A pre-registered, adversarially-reviewed null with a stated minimum detectable effect.**
   Nulls are under-published; a bounded, powered, pre-specified one is a result, not a failure.
3. **A quantification of baseline post-rationalization across four domains** — ~1 in 7 cited
   sentences on Wikipedia QA, ~1 in 33 on web-passage QA — for a widely-deployed 7B instruct model.
4. **A no-planting control for the post-rationalization probe, and an analysis of when it
   matters.** Three results the original protocol could not have produced: (a) for two of three
   injection variants the control is *degenerate by construction* — the injection appends rather
   than substitutes, so the raw rate already is the attributable rate, and Wallat's headline
   numbers are defensible for exactly that reason; (b) for the substituting variant the base rate
   is real but small (0–0.9%); (c) under a "cited at all" rule the `cited_other` control is 100%
   and the attributable effect is **negative**, so that raw rate measures the selection rule rather
   than the model. The transferable lesson: *report the attributable rate, and where the control is
   degenerate, demonstrate it rather than assume it.*
5. **An open-source, single-GPU, dataset-agnostic faithfulness-probing framework.** Adding a
   dataset is one registry entry. Wallat's original protocol required a 104B proprietary model with
   a bespoke grounded-generation template; this runs on Kaggle's free tier.
6. **A negative methodological result**: an attribution probe defined by lexical overlap between
   claim and cited chunk is self-confounding when the hypothesis under test is lexical matching.
   The fix is documented with the bias measurements that motivated it.

---

## 8. Limitations

Stated in the write-up, not buried.

1. **Format mismatch (open).** The three RL arms are agentic multi-turn search models trained to
   emit `<search>`/`<information>`/`<answer>` sequences with retrieval in the loop. They are
   evaluated here single-turn, with fixed pre-retrieved passages, in a plain cite-with-`[N]` format
   they were never trained on. Search-R1's 24–52% abstention is plausibly *breakage* rather than
   behaviour. The arms are not exchangeable under this prompt, and a format mismatch that degrades
   instruction-following could equally **mask a real effect or manufacture a spurious one**.
2. **Construct validity is assumed, not validated (open — the highest-risk item).** The inferential
   leap — *the model cited a passage whose real content is unrelated, therefore it is
   token-matching rather than comprehending* — is inherited from Wallat. An alternative reading
   exists for every flagged instance: the prompt says "answer using ONLY the numbered passages",
   the statement **is** in that passage, and citing it is literal compliance with an adversarially
   corrupted input. **No human has annotated a single flagged instance.**
3. **No SFT control arm (open).** Each RL arm differs from the baseline in three ways: it was
   fine-tuned at all, fine-tuned on retrieval-QA data, and fine-tuned with RL. Nothing isolates the
   third. Without it, RQ1 is associational.
4. **Training-data overlap (mitigated).** Search-R1 is trained on NQ train and evaluated on
   KILT-NQ. The three added datasets are OOD for all three RL arms — and notably the one consistent
   effect (R-Search) is *strongest off* NQ, which cuts against contamination as its explanation.
   The overlap is not yet quantified.
5. **Exact-substring span recovery (bounded).** Misses paraphrase, and the miss may differ by arm.
   Diagnostic: `cited & not recovered` is 1–4% of the `random` and `rel_uncited` cells and
   comparable across arms, so it is unlikely to mask a differential effect; it is material on
   `cited_other`.
6. **Fixed injection position (open).** The span is always appended at the end of the chunk, so
   positional citation bias is fully confounded with the treatment.
7. **`cited_other` is under-powered except on RAGTruth** — 13–43 clusters on the other three
   datasets, below the ~30 the cluster-robust estimator needs, so its SEs are unreliable there.
8. **HAGRID distractors are not a uniform sample** — drawn via a streaming shuffle over
   `miracl/miracl-corpus`, which samples from a buffer and a shuffled shard order. Fine for
   topically-unrelated noise; the write-up must not claim uniformity.
9. **Generalizability.** Three RL checkpoints from one narrow family of agentic-search work
   published within months of each other; one base model family; English only; one prompt; one
   decoding setting. Citation behaviour is well documented as prompt-sensitive, so this is a
   property of *this prompt* until a sensitivity analysis says otherwise.
10. **~~No version control~~ (closed).** Now in git; `runs/` (raw data) is hosted separately on
    Google Drive rather than tracked in the repo — see the project README.

---

## 9. What is next

The project is at a fork. **A** and **B** are not exclusive, and B does not depend on A.

### Path A — the SFT control arm (closes limitation 8.3)

The last thing standing between this and a causal claim about RL.

- **A1.** Look for a public SFT-on-the-same-data checkpoint from any of the three RL projects
  first. If none exists, LoRA-SFT `Qwen2.5-7B-Instruct` on the same NQ/HotpotQA mixture is
  feasible on Kaggle free tier at 7B with 4-bit + LoRA — but it is a real training job on a
  platform built for inference, and is the single largest new risk in the project.
- **A2.** Run it through the existing pipeline unchanged: one entry in `pipeline/config.py`, one
  arm in the notebook generators, 8 GPU kernel runs (planted + control × 4 datasets), ~10 h.
  Nothing in stages 03/05/05b changes.
- **A3.** Re-run the DiD with SFT as the reference arm as well as baseline. The interesting
  quantity becomes `attributable(RL) − attributable(SFT)`.
- **Honest prior:** if the SFT arm lands between baseline and the RL arms, R-Search's effect is
  probably "fine-tuning on retrieval data", not "RL". That is publishable either way, and it is the
  first thing a reviewer will ask for.
- **If A1 fails**, RQ1 must be re-scoped *in writing* to "do released RLVR search checkpoints
  exhibit more attributable citation faking than their base model" — associational, honest, still
  publishable. Draft that re-scope now, not after a failed attempt.

### Path B — write the paper

The claim set is already coherent: a pre-registered, controlled, four-dataset measurement of
attributable post-rationalization; a bounded null on RLVR; one checkpoint-level exception that
survives its pre-specified test; and a methodological contribution about controls that transfers
beyond this probe.

- **B1. Human validation of the construct — the only remaining blocker.** Stratified annotation of
  200–300 flagged and unflagged instances, reporting agreement between the automatic flag and human
  judgement. The control settled *whether the rate is attributable to the planting*; it says nothing
  about *whether the flag measures post-rationalization at all*. Annotator time, no compute.
  **Do this before submitting anywhere.**
- **B2. Pre-specified cross-dataset test of the R-Search effect.** Converts four same-signed
  estimates into a claim or retires it. No new runs — declare the test, then run it.
- **B3. Format sensitivity.** At minimum report per-arm instruction-following compliance as a
  covariate; the abstention spread (15–52%) is large enough that a reviewer will raise it.
- **B4. Quantify NQ contamination** for the RL checkpoints against their training splits. No GPU.
- **B5. Venue.** The reproducibility framing is the strongest angle: ECIR's reproducibility track
  and TMLR's reproducibility certification both explicitly reward "new insight that emerged from
  reproduction which was not reported in the original paper", which is exactly §7.4. ICTIR is where
  Wallat et al. published and is a strong direct fit. Student Research Workshops at ACL/NAACL/EMNLP
  are designed for solo thesis-stage authors. A findings- or workshop-length version of RQ1 alone
  is the de-risking fallback.

### Deferred

- **Randomize injection position** — the last unaddressed design confound, smaller than the SFT
  gap, and it costs another 16–48 kernel runs.
- **RQ2 / inference-time repair (RARR-style)** — the second half of the thesis. Gated on B1: an
  imperfect verifier flattens resampling gains (Stroebl et al.), so the flag's validity is a
  prerequisite, not an optional extra.

---

## 10. Numbers a paper or proposal can quote

| quantity | value |
|---|---|
| Datasets | 4 (KILT-NQ, ALCE-QAMPARI, HAGRID, RAGTruth-QA) |
| Questions per dataset | 500 (NQ: 499) |
| Model arms | 4, all sharing `Qwen2.5-7B-Instruct` |
| Adversarial variants | 3 (`random`, `rel_uncited`, `cited_other`) |
| Kaggle GPU kernel runs | **48** (32 main + 16 control) |
| Scored records | **36,690** (18,345 planted + 18,345 control) |
| Control records newly generated on GPU | 7,625 (`random`); the other 10,720 derived exactly |
| Attributable rate, baseline, `rel_uncited` | **+14.0 / +9.1 / +14.8 / +2.9 pp** (NQ / QAMPARI / HAGRID / RAGTruth) |
| Pooled attributable baseline rate (exploratory) | **+10.7pp** |
| No-planting control base rate, `random` | 0 in 7 of 16 cells; 0.1–0.9% in the rest |
| Share of the raw `random` rate that is attributable | 83–100%, mean 97% |
| Only Holm-significant RLVR contrast | R-Search, RAGTruth, `rel_uncited`, DiD **+3.0pp**, Holm **p = 0.002** |
| `cited_other` raw cited-anywhere rate vs its control | 79–90% vs **100%** — attributable effect negative |
| Agreement, original flag vs `cite_aligned` | 92.2–99.6% |
| Minimum detectable difference at 80% power | ≈ +7pp on `rel_uncited` |
| Corpus chunks indexed | 40,791 / 44,879 / 8,986 / 6,981 |
| Compute | free-tier Kaggle only — 2×T4 or 1×P100, 2 concurrent sessions |

---

## 11. References

- **Wallat, Zhang, Anand (2025)**, "Correctness is not Faithfulness in RAG Attributions" —
  arXiv:2412.18004, ICTIR 2025 (DOI 10.1145/3731120.3744592). *Core methodology source.*
  Repo: https://github.com/jwallat/RAG-attributions (vendored read-only at `reference/`).
- **Search-R1** — arXiv:2503.09516 · **ReSearch** — arXiv:2503.19470 ·
  **R-Search** — arXiv:2506.04185. *The three RL checkpoints under test.*
- **R1-Searcher** — arXiv:2503.05592 · **RAG-R1** — arXiv:2507.02962. *Considered and excluded.*
- **VERITAS**, "Beyond Correctness: Rewarding Faithful Reasoning in RAG" — arXiv:2510.13272.
  Shows RLVR search agents have degraded *reasoning-chain* faithfulness — a different construct
  from citation-to-evidence attribution. Reviewers will conflate the two; address it head-on.
- **VeriCite** — arXiv:2510.11394 · **CiteFix** — arXiv:2504.15629. Closest adjacent
  inference-time citation-repair baselines; neither compares against RLVR-trained agents.
- **"Do LLM Attribution Metrics Transfer?"** — arXiv:2606.23915. Off-the-shelf NLI attribution
  scorers don't transfer across datasets, which is why the scorer here is deterministic string
  matching and why human validation is a prerequisite.
- **RARR** (Gao et al. 2023) — arXiv:2210.08726, ACL 2023. *The model for RQ2's repair method.*
- **Snell et al. 2024** — arXiv:2408.03314 · **Brown et al. 2024** — arXiv:2407.21787 ·
  **Stroebl et al. 2024** — arXiv:2411.17501. *Inference-time compute: the case for RQ2, and the
  caveat that imperfect verifiers flatten resampling gains.*
- **Xiong et al. 2025** — arXiv:2504.11343. Backs "cheap methods recover most of RL's benefit".

Novelty was last adversarially checked against the literature with RQ1 and RQ2 both OPEN. That
check has a shelf life — the field moves monthly. Re-check before any submission.
