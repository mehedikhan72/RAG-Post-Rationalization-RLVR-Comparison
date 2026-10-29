# Attributable Post-Rationalization in RAG Citations — Study Summary

The design, every number and the findings of the study, in one place. Implementation, reproduction
commands and environment details are in [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md). The paper is
*Attributable Post-Rationalization in RAG Citations: A Controlled Reproduction and an RLVR
Comparison*.

---

## 1. The question

RAG systems attach citations to their answers, and readers treat those citations as evidence of
where the answer came from. Wallat et al. (2025) showed that this trust is often misplaced:
models frequently cite passages they did not use. They call this **post-rationalization**: the
answer comes first, and the citation is attached afterwards. Meanwhile, search agents are now
trained with **RLVR** (reinforcement learning from verifiable rewards). RLVR rewards a *correct
answer* and says nothing about whether the citation is honest.

**Question.** Does RLVR training change citation post-rationalization relative to the base model,
holding the base model, retrieval corpus, prompt and scorer constant? And is any change due to RL
itself, or to fine-tuning on search data at all? A non-RL fine-tune from the same base answers
the second part.

**Constraint.** Free-tier Kaggle GPUs only. Every model runs inference on released checkpoints.

---

## 2. Design

### 2.1 Models: five, one base

| arm | checkpoint | training on top of the base |
|---|---|---|
| `baseline` | `Qwen/Qwen2.5-7B-Instruct` | none |
| `search_r1` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo` | RLVR, PPO, NQ+HotpotQA |
| `research` | `agentrl/ReSearch-Qwen-7B-Instruct` | RLVR, GRPO |
| `rsearch` | `qingfei1/R-Search-7b-grpo` | RLVR, GRPO |
| `search_sft` | `Jianwen/Search-7B-SFT` | supervised fine-tuning on 1,214 search trajectories over NQ+HotpotQA, written by o3 (SkillRL's checkpoint before its RL stage) |

All five share the Qwen2.5-7B-Instruct base and have byte-identical chat templates. Any difference
from the baseline is therefore training, not model family. No agent's own search or tool calling
is ever invoked. Every model gets the same fixed top-5 passages, so "training changed citation
behaviour" is not confounded with "training changed retrieval".

Search-SFT is a **reference point for fine-tuning on search data**, not a matched control. It
shares Search-R1's training questions but not its data, format or trainer.

**Considered and excluded:**
- **RAG-R1:** no public checkpoint.
- **R1-Searcher:** trained from the *base* Qwen2.5-7B, not the instruct model.
- **SimpleDeepSearcher** (SFT): failed the go/no-go gate (§3), with 0 of 499 format-compliant
  answers on NQ. It writes long reasoning until the 300-token cap.

### 2.2 Datasets: four, chosen to break one confound

| dataset | questions | passages | gold-hit@5 | mean unique docs in top-5 |
|---|---|---|---|---|
| KILT-NQ | 499 | 40,791 | 95.4% | 2.33 |
| ALCE-QAMPARI | 500 | 44,879 | 62.7% (of 440 evaluable) | 5.00 |
| HAGRID | 500 | 8,986 | 94.4% | 4.66 |
| RAGTruth-QA | 500 | 6,981 | 91.8% | 4.82 |

Search-R1 and Search-SFT were trained on NQ questions, so on NQ alone a contrast partly measures
familiarity with the data. The other three datasets are out of domain for every fine-tuned model.
They also avoid NQ's retrieval quirk: in 186 of 499 NQ questions (37.3%) all five passages come
from one document, and the other datasets have none. QAMPARI's low gold-hit@5 is expected. Its
gold ids are recovered heuristically by answer-alias matching, so that number measures the
heuristic, not the retriever.

### 2.3 Pipeline

```
01 build         per-dataset notebook  -> sampled_questions.json + chunks.json
02 generation    GPU, one notebook per model and dataset -> clean cited answers on the top-5
03 adversarial   local, no GPU         -> plant a span, three variants per claim
04 re-inference  GPU, same model and prompt, doctored context
05 scoring       local, deterministic  -> the post-rationalization flag
03b control      local                 -> the same contexts with the span removed
04b control      GPU, `random` variant only (the other two are derived exactly, §4)
05b scoring      local                 -> attributable rates, differences, cluster bootstrap
```

**Retrieval.** BM25 (`rank_bm25`, `BM25Okapi`, default k1 = 1.5, b = 0.75) over passages of up to
100 words, top 5. Scoring uses the passage body only; the model sees "Title: text".

**Prompt.** One prompt, byte-identical across models and stages (SHA-256 `00db51102ce17927…`):
- answer using ONLY the numbered passages;
- end every sentence with exactly one citation in square brackets, never `[1][2]`;
- reply exactly "I don't know" if the passages do not contain the answer.

Decoding is greedy, at most 300 new tokens, in fp16 with no quantization.

**Unit of analysis: a claim,** meaning one sentence with exactly one citation. An answer that
breaks the format is dropped whole. Rescuing the compliant half would bring back the
claim-to-citation ambiguity the format exists to remove.

**Span rule.** The four consecutive words with the most characters in the claim. The rule is
blunt and deterministic. It is taken from the claim alone, not matched against the cited passage,
so the span is the model's own wording. `span_in_question` is recorded per record, so any overlap
with the question can be measured.

**Three variants.** Each changes exactly one of the five slots:

| variant | the doctored slot becomes | substitutes? |
|---|---|---|
| `random` | a passage from a *different question*, + span | **yes** |
| `rel_uncited` | an uncited passage from *this* question's top-5, its own text, + span | no |
| `cited_other` | a passage cited by a *different sentence* of this answer, + span | no |

The passage the model originally cited stays in the context in every variant.

### 2.4 Attrition

| dataset | arm | "I don't know" | answered | format-compliant | usable claims |
|---|---|---|---|---|---|
| NQ | baseline | 149 (30%) | 350 | 315 | 371 |
| NQ | search_r1 | 208 (42%) | 291 | 238 | 281 |
| NQ | research | 158 (32%) | 341 | 310 | 355 |
| NQ | rsearch | 129 (26%) | 370 | 341 | 387 |
| NQ | search_sft | 108 (22%) | 391 | 359 | 395 |
| QAMPARI | baseline | 186 (37%) | 314 | 241 | 329 |
| QAMPARI | search_r1 | 259 (52%) | 241 | 186 | 283 |
| QAMPARI | research | 191 (38%) | 309 | 258 | 352 |
| QAMPARI | rsearch | 167 (33%) | 333 | 285 | 362 |
| QAMPARI | search_sft | 121 (24%) | 379 | 309 | 369 |
| HAGRID | baseline | 85 (17%) | 415 | 378 | 438 |
| HAGRID | search_r1 | 121 (24%) | 379 | 327 | 389 |
| HAGRID | research | 90 (18%) | 410 | 383 | 439 |
| HAGRID | rsearch | 75 (15%) | 425 | 396 | 435 |
| HAGRID | search_sft | 75 (15%) | 425 | 388 | 408 |
| RAGTruth | baseline | 86 (17%) | 414 | 291 | 874 |
| RAGTruth | search_r1 | 127 (25%) | 373 | 234 | 705 |
| RAGTruth | research | 85 (17%) | 415 | 296 | 828 |
| RAGTruth | rsearch | 75 (15%) | 425 | 346 | 886 |
| RAGTruth | search_sft | 72 (14%) | 428 | 348 | 813 |

Search-R1 abstains most on every dataset (24–52%). Search-SFT and R-Search abstain least.
RAGTruth answers are longer, so they give more claims per question, which is why its
`cited_other` cell is the only well-powered one. The per-cell *n* in §5 is slightly smaller than
"usable claims": a variant is built only where it is eligible (`random` and `rel_uncited` need
an uncited slot; `cited_other` needs a second cited sentence).

---

## 3. Analysis plan (pre-specified)

The scoring rule, the primary outcome and the corrected comparisons were written down **before
the relevant scorer was run**. The same was done for the Search-SFT comparisons before that model
was run, including a go/no-go gate: an arm is dropped if fewer than 30% of questions get a
format-compliant answer on two or more datasets. These plans were kept in the project notes, not
filed with a public registry.

| | |
|---|---|
| Primary variant | `rel_uncited` |
| Unit | one claim; **cluster** = `question_id` |
| Denominator | every record in the cell (**unconditional**) |
| Primary outcome | `cite_aligned` (§3.1) |
| Contrast families | A: each RL agent vs baseline. B: Search-SFT vs baseline. C: each RL agent vs Search-SFT (exploratory) |
| Correction | **Holm** within each family, within dataset |

*Why unconditional:* dividing by "the span came back" conditions on something downstream of the
treatment that differs by model. The breakdown into `P(recover)`, `P(flag | recover)` and
`P(flag)` is reported alongside.

*Why clusters:* one question can yield up to 11 claims sharing a context and an
answer, so treating claims as independent would understate the uncertainty.

### 3.1 Two outcomes

- **`flag`** (stage 05): the doctored marker is cited **in the same sentence in which the planted
  span reappears**.
- **`cite_aligned`** (stage 05b, primary): the doctored marker is cited in the **aligned
  sentence**. That is the answer sentence with the highest token-set Jaccard similarity to the
  original claim, which must be at least 0.3. Below that threshold the record scores 0.

The second exists because the first cannot be evaluated when no span was planted. Agreement on the
planted condition, all five models pooled:

| dataset | `random` | `rel_uncited` | `cited_other` |
|---|---|---|---|
| NQ | 99.0% | 98.4% | 92.6% |
| QAMPARI | 99.2% | 98.5% | 92.9% |
| HAGRID | 99.2% | 98.6% | 96.0% |
| RAGTruth | 99.5% | 98.9% | 95.0% |

### 3.2 The estimand

```
attributable(arm, variant) = P(doctored slot cited | planted) − P(doctored slot cited | control)
DiD = attributable(arm) − attributable(reference arm)
```

The 95% CIs come from a **cluster bootstrap over `question_id`**: 2,000 replicates, seed 42, with
both conditions recomputed on the same resampled questions.

---

## 4. The no-planting control

Wallat et al. ran no control. Their released code builds three planted datasets and reports a raw
rate, counted only over answers in which the planted statement reappeared.

### 4.1 Two of the three control values follow from the construction

| variant | slot contents change? | span appended? | control |
|---|---|---|---|
| `random` | **yes**: a passage from another question replaces an uncited slot | yes | must be measured |
| `rel_uncited` | no | yes | 0 by construction |
| `cited_other` | no | yes | 100% under cited-anywhere, by selection |

For `rel_uncited` and `cited_other`, removing the span restores the clean context **byte for
byte**. With the same prompt and greedy decoding, the control answer *is* the clean answer. This
was checked record by record on all **13,282** records of these two variants (five models): chunk
ids, order, and every passage text. Six records score above zero, all from RAGTruth answers that
repeat an identical sentence.

Only `random` needed GPU control runs: **9,590 control records**, each paired with its planted
twin.

### 4.2 What the control found on `random`

`attributable = planted − control`, `cite_aligned`, in percentage points:

| dataset | baseline | search_r1 | research | rsearch | search_sft |
|---|---|---|---|---|---|
| NQ | 10.8 − 0.0 = **+10.8** | 8.9 − 0.0 = **+8.9** | 9.3 − 0.3 = **+9.0** | 10.6 − 0.3 = **+10.3** | 13.7 − 0.3 = **+13.4** |
| QAMPARI | 4.4 − 0.0 = **+4.4** | 5.7 − 0.0 = **+5.7** | 7.8 − 0.0 = **+7.8** | 8.5 − 0.3 = **+8.2** | 8.8 − 0.0 = **+8.8** |
| HAGRID | 8.5 − 0.9 = **+7.6** | 8.6 − 0.0 = **+8.6** | 10.6 − 0.0 = **+10.6** | 12.6 − 0.7 = **+11.9** | 12.7 − 0.2 = **+12.4** |
| RAGTruth | 2.2 − 0.1 = **+2.1** | 3.4 − 0.6 = **+2.9** | 2.4 − 0.1 = **+2.3** | 4.0 − 0.1 = **+3.9** | 3.8 − 0.0 = **+3.8** |

Across all models the control is 19 of 9,590 records (0.2%). It is **exactly 0 in 9 of the 20
cells** and 0.1–0.9% in the rest, and **83–100% of each raw rate survives as attributable (mean
97%)**.

### 4.3 Where the control destroys a number

On `cited_other`, the doctored slot is selected as one the clean answer **already cites** for a
different sentence. Under the loose "cited at all" outcome (`cite_anywhere`), the control is
therefore 100% and the attributable effect is **negative** (baseline):

| dataset | planted k/n | rate | control k/n | attributable | 95% CI |
|---|---|---|---|---|---|
| NQ | 58/69 | 84.1% | 69/69 | **−15.9pp** | [−25.4, −7.9] |
| QAMPARI | 97/123 | 78.9% | 123/123 | **−21.1pp** | [−30.3, −13.2] |
| HAGRID | 50/61 | 82.0% | 61/61 | **−18.0pp** | [−27.9, −8.8] |
| RAGTruth | 558/623 | 89.6% | 623/623 | **−10.4pp** | [−13.7, −7.4] |

A raw cited-anywhere rate of 79–90% on this variant measures the sampling rule, not the model.

### 4.4 Span recovery is not evidence that planting worked

On the `random` control, with nothing planted, the four words reappear in 60–74% of answers,
because they come from the model's own answer to the same question. This is why the denominator
is unconditional.

---

## 5. Results

### 5.1 Baseline, primary variant (`rel_uncited`, `cite_aligned`)

| dataset | planted k/n | rate | control k/n | attributable | 95% CI |
|---|---|---|---|---|---|
| NQ | 52/371 | 14.0% | 0/371 | **+14.0pp** | [+10.4, +17.8] |
| QAMPARI | 29/319 | 9.1% | 0/319 | **+9.1pp** | [+6.2, +12.0] |
| HAGRID | 64/433 | 14.8% | 0/433 | **+14.8pp** | [+11.8, +18.2] |
| RAGTruth | 25/864 | 2.9% | 0/864 | **+2.9pp** | [+1.8, +3.9] |

On the Wikipedia-based datasets this is **9–15% of probed claims**. Across all four datasets it is
**8.6%** of the baseline's probed claims (170/1,987, 95% CI [7.4, 9.8], resampling questions
within each dataset). This pooled figure is exploratory, as pooling was not pre-specified.

Attributable rate for every model (the control is 0/n in all twenty cells, so these equal the
planted rates):

| dataset | baseline | search_r1 | research | rsearch | search_sft |
|---|---|---|---|---|---|
| NQ | 52/371 = **14.0** | 36/281 = **12.8** | 50/355 = **14.1** | 57/387 = **14.7** | 70/395 = **17.7** |
| QAMPARI | 29/319 = **9.1** | 38/283 = **13.4** | 25/332 = **7.5** | 43/352 = **12.2** | 46/354 = **13.0** |
| HAGRID | 64/433 = **14.8** | 53/383 = **13.8** | 59/434 = **13.6** | 80/430 = **18.6** | 56/403 = **13.9** |
| RAGTruth | 25/864 = **2.9** | 29/699 = **4.1** | 30/823 = **3.6** | 52/879 = **5.9** | 43/813 = **5.3** |

### 5.2 Difference from the baseline (DiD, pp)

Each cell is the DiD, with a 95% CI and the p-value in parentheses. The RL agents (family A) are
Holm-adjusted within each dataset. Search-SFT (family B) is a single comparison, so its p-value
is unadjusted.

| dataset | search_r1 | research | rsearch | search_sft |
|---|---|---|---|---|
| NQ | −1.2 [−6.2, 3.9] (1.000) | +0.1 [−3.9, 4.3] (1.000) | +0.7 [−3.9, 5.1] (1.000) | +3.7 [−1.1, 8.4] (0.133) |
| QAMPARI | +4.3 [−0.5, 9.7] (0.267) | −1.6 [−5.1, 2.3] (0.415) | +3.1 [−1.4, 7.3] (0.324) | +3.9 [−0.6, 8.2] (0.081) |
| HAGRID | −0.9 [−4.9, 3.2] (1.000) | −1.2 [−4.9, 2.5] (1.000) | +3.8 [−0.3, 8.0] (0.234) | −0.9 [−4.9, 3.2] (0.666) |
| RAGTruth | +1.3 [−0.5, 3.0] (0.334) | +0.8 [−0.7, 2.2] (0.334) | **+3.0 [1.2, 4.8] (0.002)** | **+2.4 [0.5, 4.4] (0.011)** |

Each of the four per-dataset differences averaged, weighted by number of questions (exploratory):
- search_r1 +0.5
- research −0.5
- rsearch +2.7
- search_sft +2.2

### 5.3 RL agents vs Search-SFT (family C, exploratory, Holm within dataset)

| dataset | search_r1 | research | rsearch |
|---|---|---|---|
| NQ | −4.9 (0.165) | −3.6 (0.248) | −3.0 (0.248) |
| QAMPARI | +0.4 (1.000) | **−5.5 (0.039)** | −0.8 (1.000) |
| HAGRID | −0.1 (1.000) | −0.3 (1.000) | +4.7 (0.126) |
| RAGTruth | −1.1 (0.658) | −1.6 (0.270) | +0.6 (0.658) |

On `random`, all twelve of these differences are at most +0.1pp and none is significant after
correction.

### 5.4 The same comparison with the baseline, on `random`

| dataset | search_r1 | research | rsearch | search_sft |
|---|---|---|---|---|
| NQ | −1.9 (0.984) | −1.8 (0.984) | −0.4 (0.984) | +2.6 (0.201) |
| QAMPARI | +1.3 (0.379) | +3.4 (0.114) | +3.8 (0.114) | **+4.4 (0.014)** |
| HAGRID | +1.0 (0.561) | +3.0 (0.182) | +4.2 (0.081) | **+4.8 (0.018)** |
| RAGTruth | +0.8 (0.594) | +0.2 (0.764) | +1.8 (0.063) | **+1.7 (0.045)** |

### 5.5 `cited_other`, claim-aligned outcome (baseline)

| dataset | planted k/n | control k/n | attributable | 95% CI |
|---|---|---|---|---|
| NQ | 9/69 | 0/69 | **+13.0pp** | [+5.5, +22.5] |
| QAMPARI | 16/123 | 0/123 | **+13.0pp** | [+7.3, +19.7] |
| HAGRID | 12/61 | 0/61 | **+19.7pp** | [+10.8, +31.3] |
| RAGTruth | 67/623 | 1/623 | **+10.6pp** | [+7.9, +13.5] |

### 5.6 What a flag means: two-rater check

A flag says the citation moved to the doctored passage, not that the passage was wrong. Two raters
independently labelled 100 flagged `rel_uncited` records: 5 per model and dataset, shown without
model or dataset. One rater was a human annotator and the other an LLM. For each record they
judged whether the doctored passage, **before planting**, supported the claim. Data are in
`data/validation/`.

| | human | LLM |
|---|---|---|
| no support | **73%** [64, 82] | **69%** [60, 78] |
| partial support | 7% [2, 13] | 15% [8, 23] |
| full support | 20% [12, 28] | 16% [9, 24] |
| a correct original citation abandoned for an unsupporting passage | 58 / 100 | 52 / 100 |

Agreement on the main question is 89/100, Cohen's κ = 0.76. Nine of the 11 disagreements are
items the LLM called partial. A flag therefore measures **post-rationalization susceptibility**:
how easily a citation moves. In about 70% of flagged cases it moves to a passage that does not
support the claim.

---

## 6. Findings

**1. Post-rationalization is common in every model, RL-trained or not.** The pre-specified bar was
at least 8pp attributable, with a CI excluding 0, on at least three of four datasets. It is met:
NQ +14.0, QAMPARI +9.1, HAGRID +14.8 and RAGTruth +2.9, all CIs excluding 0. On Wikipedia-based
QA, 9–15% of probed citations follow four planted words. RAGTruth's much lower rate shows the
effect depends on the domain.

**2. No general RLVR effect.** Search-R1 (−1.2, +4.3, −0.9, +1.3pp) and ReSearch (+0.1, −1.6,
−1.2, +0.8pp) straddle zero, and no Holm-adjusted p is below 0.23. The minimum detectable
difference at 80% power is about 7pp. The study rules out an effect of that size; it cannot rule
out about 3pp. This is a bounded null.

**3. One RL agent drifts upward.** R-Search's DiD on `rel_uncited` is positive on all four
datasets (+0.7, +3.1, +3.8, +3.0) and Holm-significant on RAGTruth (p = 0.002), which was the
pre-specified condition. It weakens under the robustness checks:
- On `random`, R-Search is positive on three of four datasets and significant on none.
- Under `cite_anywhere`, it is −0.5pp on NQ and significant nowhere.

**4. The drift is not specific to RL.** Search-SFT was never trained with RL, yet it sits above the
baseline on three of four datasets and is significant on RAGTruth (+2.4pp, p = 0.011). Averaged
over the datasets, its difference (+2.2pp) is close to R-Search's (+2.7pp). It is *more* robust
than R-Search's: significant on three of four datasets on `random`, and on RAGTruth under
`cite_anywhere`. No RL agent is reliably above Search-SFT; the only Holm-significant difference
has an RL agent below it. The pre-specified reading applies: R-Search's drift is plausibly an
effect of fine-tuning on search data, not of RL. Only on HAGRID is R-Search well
above Search-SFT (+4.7pp, Holm p = 0.126).

### 6.1 Sensitivity: `cite_anywhere` on the primary variant

The pre-specified trigger for this check was a between-model spread in the unaligned rate above
5pp. It is crossed on NQ (8.0pp) and RAGTruth (6.6pp), so the analysis is repeated on
`cite_anywhere`:

| dataset | baseline rate | search_r1 | research | rsearch | search_sft |
|---|---|---|---|---|---|
| NQ | 17.8% | −2.5 (1.000) | −0.6 (1.000) | −0.5 (1.000) | +3.2 (0.257) |
| QAMPARI | 13.2% | +6.6 (0.102) | −1.7 (0.473) | +3.3 (0.460) | +3.5 (0.171) |
| HAGRID | 16.6% | −1.0 (0.665) | −2.1 (0.586) | +3.1 (0.420) | −1.0 (0.655) |
| RAGTruth | 7.1% | +3.4 (0.051) | +1.7 (0.224) | +2.3 (0.224) | **+4.3 (0.009)** |

Findings 1, 2 and 4 hold. Finding 3 is weaker.

---

## 7. Relation to Wallat et al. (2025)

Wallat et al. report 57% (273/476) for their relevant-but-uncited condition, using Command R+
(104B, 4-bit) on NQ. That number is not comparable to ours, for four reasons:
- A cited span could cite several passages, and success needed only that the doctored passage be
  *among* them.
- The target passage was one the model itself had marked relevant.
- The plant was the model's whole cited span.
- The rate counts only answers in which the exact span came back.

Scored their way (span reappears, doctored passage cited anywhere), our baseline's NQ rate rises
from 14.0% to 17.2%. The scoring rule explains only a few points of the gap. The paper's Appendix
I compares the two setups side by side.

---

## 8. Limitations

1. **Format mismatch.** The RL agents and Search-SFT were trained for multi-turn search in their
   own formats, and are evaluated single-turn on fixed passages with `[N]` citations. The large
   spread in abstention (14–52%) may partly reflect this. It could mask a real effect or create a
   spurious one.
2. **One SFT reference, not matched data.** Search-SFT is a single checkpoint trained on
   o3-written trajectories. It shares Search-R1's questions but not its data or trainer. A matched
   SFT/RL pair trained on identical data remains the cleanest test.
3. **What a flag means.** About 70% of flags move to a non-supporting passage (two raters, §5.6).
   In the rest, citing the doctored passage could be read as obedience to "answer from the
   passages". One of the two raters was an LLM.
4. **Training-data overlap.** Search-R1 and Search-SFT were trained on NQ questions. The other
   three datasets are out of domain, and R-Search's pattern is strongest away from NQ. The overlap
   is not quantified.
5. **Exact-substring span recovery.** This misses paraphrase. `cited & not recovered` is 1–4% of
   the `random` and `rel_uncited` cells and similar across models.
6. **Fixed injection position.** The span is always appended at the end of the passage, so any
   positional citation bias is mixed in with the treatment.
7. **`cited_other` is underpowered except on RAGTruth.** It has only 13–43 question clusters on
   the other three datasets.
8. **HAGRID distractors are not a uniform sample.** They are drawn by a streaming shuffle over
   `miracl/miracl-corpus`.
9. **Scope.** Three RL agents from one line of work and one SFT model, one base model, English
   only, one prompt, one decoding setting.

---

## 9. Numbers to quote

| quantity | value |
|---|---|
| Datasets | 4 (KILT-NQ, ALCE-QAMPARI, HAGRID, RAGTruth-QA) |
| Questions per dataset | 500 (NQ: 499) |
| Models | 5, all from `Qwen2.5-7B-Instruct` (3 RLVR, 1 SFT, base) |
| Probe variants | 3 (`random`, `rel_uncited`, `cited_other`) |
| Kaggle GPU jobs | 60 |
| Scored records | 45,744 (22,872 planted + 22,872 control) |
| Control records run on GPU / derived exactly | 9,590 (`random`) / 13,282 |
| Baseline attributable rate, `rel_uncited` | +14.0 / +9.1 / +14.8 / +2.9 pp (NQ / QAMPARI / HAGRID / RAGTruth) |
| Baseline, share of probed claims across all four datasets (exploratory) | 8.6% (170/1,987), 95% CI [7.4, 9.8] |
| `random` control base rate | 0.2% (19/9,590); 0 in 9 of 20 cells |
| Holm-significant differences from the baseline | R-Search, RAGTruth, +3.0pp (p = 0.002); Search-SFT, RAGTruth, +2.4pp (p = 0.011, single comparison) |
| Flag check, two raters | agreement 89/100, κ = 0.76; no support before planting 73% (human), 69% (LLM) |
| Minimum detectable difference at 80% power | about 7pp on `rel_uncited` |
| Compute | free-tier Kaggle GPUs; fp16, no quantization; P100 (base and RL models), T4 ×2 (Search-SFT) |

---

## 10. References

- **Wallat, Heuss, de Rijke, Anand (2025)**, "Correctness is not Faithfulness in Retrieval
  Augmented Generation Attributions", ICTIR 2025, arXiv:2412.18004. Code:
  https://github.com/jwallat/RAG-attributions
- **Search-R1** arXiv:2503.09516 · **ReSearch** arXiv:2503.19470 · **R-Search** arXiv:2506.04185.
  The three RLVR checkpoints.
- **SkillRL** arXiv:2602.08234. Source of the Search-SFT checkpoint.
- **SimpleDeepSearcher** arXiv:2505.16834. SFT candidate, dropped at the gate.
- **R1-Searcher** arXiv:2503.05592 · **RAG-R1** arXiv:2507.02962. Considered and excluded.
- **VERITAS** arXiv:2510.13272. Faithfulness within the reasoning chain of search agents, a
  different construct from citation-to-evidence attribution.
