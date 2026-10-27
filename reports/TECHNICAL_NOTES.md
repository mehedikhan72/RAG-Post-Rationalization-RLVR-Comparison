# Technical Notes

**Companion to `reports/PROJECT_STATE.md`.** That file holds the design, the results and the
argument. This one holds everything a person needs to *rerun, extend or debug* the work:
implementation detail, code map, environment mechanics, and the pitfalls that cost real time.

**Two run batches.** Comments in the code and notebooks refer to them as *v1* and *v2*:
- **v1** (Aug–Sep 2026): the base model and the three RL agents. 48 Kaggle GPU jobs on a single
  P100.
- **v2** (Oct 2026): the Search-SFT reference arm. 12 jobs on two T4s, run on v1's own Docker
  image (§6.2).

---

## 1. Code map

Everything under `pipeline/` is dataset-independent and takes `--dataset <name>`. Everything under
`datasets/` and `runs/` belongs to exactly one dataset. Adding a dataset is one entry in
`pipeline/config.py`.

```
pipeline/
  config.py                        the dataset registry; Kaggle ids, arm order, variants
  03_adversarial_construction.py   plants the spans, builds the 3 variants
  05_score_postrationalization.py  the original flag, cluster-robust contrasts, Holm
  05b_score_noplant_control.py     attributable rates, DiD per contrast family, cluster bootstrap
  notebooks/02_generation/<ds>/    5 generated GPU notebooks (one per model), clean run
  notebooks/04_postration/<ds>/    5 generated GPU notebooks, planted run
  notebooks/04_postration_noplant/<ds>/  4 generated GPU notebooks, control run (v1 models)
  scripts/
    generate_generation_notebooks.py     emits the 02x notebooks + kernel metadata
    generate_postration_notebooks.py     emits the 04x notebooks; --control, --variants, --diff
    build_retrieval_dataset.py           stages retrieval data for Kaggle
    build_adversarial_dataset.py         stages the planted contexts for Kaggle
    build_noplant_control.py             derives the control contexts from the archived planted ones
    derive_noplant_from_clean.py         re-emits rel_uncited/cited_other controls from the clean run
    run_pipeline.py                      drives the main pipeline end to end on Kaggle
    run_noplant_control.py               drives the 16 v1 control kernels, rolling window of 2
    run_sft_arms.py                      drives the SFT arm end to end (generation, gate, planted
                                         and control runs, scoring), resumable
    preflight_arm.py                     checks a candidate model's config and chat template
    report_sft_control.py                renders results/sft_control_tables.md (all comparisons)
    report_noplant_control.py            renders reports/<ds>/noplant_control.md (v1 detail)
    extract_flag_sample.py               draws the 100-item flag-validation sample
    build_review_sheet.py                builds the offline review sheets (+ review_sheet_template.html)
    report_flag_validation.py            agreement and results of the two-rater check
    validate_retrieval_data.py, jsonl_to_json.py, generate_builder_notebooks.py
  tests/
    test_noplant_control.py        asserts each control record differs only by the removed span
datasets/<ds>/                     the builder notebook -> sampled_questions.json + chunks.json
data/validation/                   flag-validation sample, LLM labels, human rater labels
results/                           generated tables (not tracked)
runs/<ds>/                         retrieval, generation, adversarial, adversarial_noplant,
                                   postration, postration_noplant, scoring, scoring_noplant,
                                   kaggle_staging
```

`runs/` is not tracked in this repo (retrieval corpora, generations, re-runs and scoring output).
Full contents are hosted on Google Drive; the link is in the project README.

## 2. Reproduction

Main pipeline, per dataset:

```
python pipeline/scripts/generate_generation_notebooks.py  --dataset <ds>
python pipeline/03_adversarial_construction.py            --dataset <ds>
python pipeline/scripts/build_adversarial_dataset.py      --dataset <ds>
python pipeline/scripts/generate_postration_notebooks.py  --dataset <ds>
python pipeline/05_score_postrationalization.py           --dataset <ds>
```

`run_pipeline.py --dataset <ds>` drives all of it on Kaggle, pushing GPU kernels two at a time
and resuming stage-by-stage from whatever is already on disk.

No-planting control (v1 models):

```
python pipeline/scripts/build_noplant_control.py          --dataset <ds> --stage
python pipeline/tests/test_noplant_control.py
python pipeline/scripts/generate_postration_notebooks.py  --dataset <ds> --control \
                                                          --variants random --diff
python pipeline/scripts/derive_noplant_from_clean.py      --dataset <ds>
python pipeline/scripts/run_noplant_control.py             # 16 GPU kernels, 2 concurrent
python pipeline/05b_score_noplant_control.py              --all --dump
python pipeline/05b_score_noplant_control.py              --all --outcome cite_anywhere
python pipeline/scripts/report_noplant_control.py         --all
```

SFT reference arm (v2). One resumable driver runs every step above for the arms in
`config.SFT_ARMS`. It runs generation, then the go/no-go gate (an arm is dropped if fewer than
30% of questions get a format-compliant answer on 2+ datasets), then staging, the planted and
`random`-control runs, and scoring:

```
python pipeline/scripts/run_sft_arms.py --dry-run          # what is done / still to do
python pipeline/scripts/run_sft_arms.py
python pipeline/scripts/report_sft_control.py              # -> results/sft_control_tables.md
```

Stage 03 takes `--arms`, so adding an arm to a finished run builds only that arm. Each listed arm
gets a fresh `random.seed(42)` stream, so existing arms' files are never touched.

Flag validation:

```
python pipeline/scripts/extract_flag_sample.py             # -> data/validation/flag_sample.json
python pipeline/scripts/build_review_sheet.py              # -> results/validation/*.html, *.csv
python pipeline/scripts/report_flag_validation.py          # -> results/validation/tables.md
```

Everything is deterministic: `random.seed(42)`, greedy decoding, bootstrap seed 42. Two runs of
either scorer produce byte-identical output.

## 3. Scoring internals

### 3.1 Parsing

`segment_answer` splits on the **citation marker**, not on `.` — the prompt makes the marker the
sentence terminator and answers are full of decimals, initials and abbreviations. Text trailing
after the final marker becomes a segment with no markers, which can never be flagged.

`normalize` = lowercase, strip punctuation, collapse whitespace. Matching is exact-substring, not
semantic. A model that reuses its own phrasing is caught; one that paraphrases is not, and that
miss may differ by arm — the live limitation, bounded by the matcher diagnostic
(`cited & not recovered`, 1–4% of `random` and `rel_uncited`).

### 3.2 The two outcomes

| field | rule |
|---|---|
| `span_recovered` | the planted span reappears anywhere in the new answer |
| `doctored_cited` / `cite_anywhere` | the doctored marker appears anywhere in the answer |
| `flag` | span and doctored marker in the **same sentence** (original primary) |
| `flag_anywhere` | recovered AND cited, not necessarily together (labelled secondary) |
| `cite_aligned` | doctored marker cited in the **aligned sentence** (primary) |
| `unaligned` | no answer sentence reaches Jaccard 0.3 against the clean claim |
| `span_in_question` | the span is a substring of the question — carried for sensitivity |
| `cited_all_slots` | the answer cites ≥5 distinct slots — carried for sensitivity |

Alignment: token-set Jaccard between each answer segment (markers stripped, normalized) and the
original clean-run `claim_text`; highest wins, ties to the earliest segment, threshold 0.3.

### 3.3 Statistics

- `clustered_proportion` — survey-linearization cluster-robust variance,
  `var(p) = [K/(K−1) · Σ_c (y_c − p·n_c)²] / (Σ_c n_c)²`. Observed deff ≈ 1.0.
- `wilson_ci` — Wilson score interval, honest at the small counts `cited_other` produces.
- `holm` — step-down adjusted p-values, returned in input order.
- `05b` uses a **cluster bootstrap** instead: 2,000 replicates, resample `question_id` with
  replacement, recompute *both* conditions on the same resampled questions so the pairing
  survives. Percentile intervals. Two-sided bootstrap p = 2·min(share ≤ 0, share ≥ 0), floored at
  1/2000.
- `05b` refuses to score if a cell arrives from both `derived/` and a kernel run — that would
  silently double a denominator.

- Contrasts are grouped in `config.CONTRAST_FAMILIES`, and Holm runs within each family
  separately: A = RL agents vs baseline, B = Search-SFT vs baseline, C = RL agents vs
  Search-SFT (exploratory). Adding a family never moves another family's p-values.

## 4. How the control is constructed

`03_adversarial_construction.py` joins the span with a single newline:
`doctor_text(t, span) -> f"{t}\n{span}"`. No title prefix is re-applied (the title is already
baked into `chunks.json[...]["text"]` as `"{title}: {piece}"`), no trailing punctuation.

`build_noplant_control.py` therefore transforms rather than regenerates: strip the `"\n" + span`
suffix from the doctored slot, assert the result equals `chunks.json[doctored_source_chunk_id]`,
carry every other field verbatim, add `pair_id` / `claim_id` / `control_of` / `span_removed`.
One failed assertion aborts the whole dataset rather than skipping silently. **Verified on
22,872/22,872 records (five models), zero mismatches.**

**Why transform and not regenerate:** `cited_other` had a set-iteration nondeterminism bug (§6.1).
Re-running stage 03 would draw different slots than the published numbers came from, and planted
vs control would stop being paired.

**Pairing key is `pair_id`**, the record's 0-based index in its `{arm}__{variant}.json` file. The
control file is a positional 1:1 transform of the planted file. The natural key
`(question_id, claim_marker, claim_text)` collides on a handful of records (14 of the 18,345 v1 records) — a model emitting the
same sentence twice with the same marker — so it is not used.

**Why `rel_uncited` and `cited_other` need no GPU:** both are emitted as
`emit(variant, target, target, ...)`, so the chunk keeps its own slot and the only edit is the
appended span. Stripping it restores the clean stage-02 context byte-for-byte; the stage-04 prompt
is byte-identical to stage-02's and both decode greedily, so the control answer *is* the archived
clean answer. `derive_noplant_from_clean.py` re-emits it in the stage-04 schema, asserting chunk
ids, order and every slot text per record first. Only `random` substitutes a foreign chunk.

**Notebook diff.** The control notebooks differ from their planted twins in exactly 4 code lines,
verified mechanically by `generate_postration_notebooks.py --diff`:

```
-OUT_DIR = Path("/kaggle/working/postration_data")
+OUT_DIR = Path("/kaggle/working/postration_noplant_data")
-VARIANTS = ["random", "rel_uncited", "cited_other"]
+VARIANTS = ['random']
```

plus `id`, `title` and `dataset_sources` in `kernel-metadata.json`. The diff is regenerable at any
time with `--diff`; it also fails loudly if a fifth line ever changes. Control kernels never
inherit NQ's frozen legacy slugs, because reusing one would push over a completed planted run and
destroy its saved output.

Only `random` needs control runs on the GPU (the other two variants are derived as above), which
is what kept the control within a free-tier quota.

## 5. Verification performed

| check | result |
|---|---|
| Prompt SHA-256 identical across stages, all five models and all control manifests | `00db51102ce17927bc23c4e425fe631992ab822affa24b876b918750d195e43b` |
| Chat templates of the five checkpoints | byte-identical, so the rendered prompt is identical |
| Strip assertion — stripped slot == `chunks.json` source text | 22,872/22,872 |
| Diff test (`test_noplant_control.py`) — only the doctored slot differs, only by the removed suffix | passes on all four datasets |
| Pairing — one control twin per planted record, counts match | 60/60 cells; 9,590 GPU + 13,282 derived |
| GPU control cells — order, question ids, claim text, doctored marker all identical to planted | 20/20 cells |
| Scoring determinism — two runs on the same input | byte-identical |
| Adding the SFT arm leaves the RL comparisons unchanged | family-A output byte-identical before and after |
| Flagged control records read by hand | only 6 exist; all duplicate-sentence Jaccard ties on RAGTruth |
| Flagged planted records read by two raters | 100; agreement 89/100, Cohen's κ = 0.76 (paper App. H) |

## 6. Pitfalls that cost real time

### 6.1 Pipeline

- **`cited_other` nondeterminism (fixed).** The variant iterated a Python `set` of
  chunk ids; per-process string-hash randomization defeated the fixed `random.seed(42)`, so the
  subsequent `random.choice` picked a different slot every run. `sorted()` before choosing. Only
  that variant was affected — `random` and `rel_uncited` always reproduced exactly. The archived
  draws the published report was computed from were left untouched rather than silently
  regenerated.
- **BM25 title-boost collapse (fixed).** Indexing on `"{title}: {piece}"` gave every
  chunk of a document the same title-term boost; 52% of NQ questions had all five top-5 slots
  filled by one document. Switched indexing to the body only (the model still *sees* the title):
  single-document rate 52% → 40%, mean unique docs 1.70 → 2.05, at a cost of gold-hit@5 100% → 99%.
- **Gold-doc collection hang.** The original loop used an AND condition to decide when to stop
  streaming KILT-Wikipedia, which never terminates when gold docs are rare — with ~99 gold docs in
  5.9M, a 25k-doc scan finds ~3 by chance. Fix: decouple into two independent passes — one
  near-full checkpointed pass for gold docs, one short capped reservoir pass for distractors.
  Confirmed: finding all gold docs took 5,896,819 of 5,903,530 records (99.9%), ~3.4 h.

### 6.2 Kaggle environment

- **P100 vs T4.** Kaggle randomly assigns either. Its default preinstalled torch dropped Pascal
  support, so a P100 session dies at `generate()` with `CUDA error: no kernel image is available`.
  Pin `torch==2.4.1+cu121` and uninstall `torchvision`/`torchaudio` — their ABI mismatch against
  the downgraded torch breaks the lazy import of `Qwen2ForCausalLM`.
- **`device_map="auto"` silently CPU-offloads.** On a single 16GB card accelerate's conservative
  estimate put part of the model on CPU (`hf_device_map` showing `{0, 'cpu'}`), costing ~4.75× in
  generation time. Force `device_map={"": 0}`.
- **`/kaggle/working` is not durable.** Zip and export after every stage that took real time.
- **Mount path is not stable.** The same private dataset appears at
  `/kaggle/input/datasets/<owner>/<slug>` in one kernel and flat at `/kaggle/input/<slug>` in
  another, pushed minutes apart with identical `dataset_sources`. The notebooks therefore locate
  their input by **walking `/kaggle/input` for the expected filenames**, and fail loudly with a
  full tree listing if not found.
- **Kernel title has a hard 50-character limit**, and the CLI swallows the response body so the
  error is a bare `400`. Titles are asserted at generation time.
- **Slug/title mismatch.** If `id` ≠ `slugify(title)`, Kaggle silently creates the kernel under the
  title-derived slug; every later push then 409s against itself. Convention: make the title equal
  the slug with hyphens as spaces.
- **Two concurrent GPU sessions, hard cap.** A third push fails outright rather than queueing.
- **The environment moved between v1 and v2.** By October 2026 Kaggle's default image had moved
  to Python 3.13, where the `torch==2.4.1` pin has no wheel, and the P100 was no longer offered
  (only T4 ×2). The v2 notebooks therefore pin v1's own Docker image in `kernel-metadata.json`
  (`docker_image`), so they install exactly what v1 installed, and request `machine_shape:
  NvidiaTeslaT4`. Both fields need kaggle CLI 2.x; 1.x drops them silently. A 7B model in fp16
  does not fit on one T4 (14.6 GiB usable), so the layers are split evenly across both cards
  (`device_map="balanced"` with a GPU-only `max_memory`), and an assertion fails the run if
  anything lands on the CPU. Precision stays fp16 throughout.

### 6.3 Driving Kaggle from a script

Recorded because they will recur for anyone driving Kaggle from a script.

1. **`kaggle kernels push` exits 0 while printing `Kernel push error: ...`** — including for
   `Maximum weekly GPU quota of 30.00 hours reached`. A return-code check reports success for a
   kernel that was never created. Fix: scan stdout, not just the exit status.
2. **A 404 status was treated as "still running."** Anything not `COMPLETE`/`ERROR`/`CANCEL`
   counted as in-flight, so a nonexistent kernel polled forever. Fix: three consecutive 404s marks
   the job failed. Cost before the fix: ~3 h of no-op polling.
3. **The Kaggle CLI can wedge on a network read and never return**, and
   `subprocess.communicate(timeout=600)` did not always fire. A `push` was seen hung for 26 min
   and another for 208 min, each blocking the queue, while the push had *already succeeded
   server-side*, so killing the client is safe. If a driver stops making progress, kill the
   stuck `kaggle` process; the driver retries and then checks with Kaggle whether the kernel
   exists.
4. **An over-broad error check killed the driver on a transient `ConnectionResetError`.** Fix:
   only Kaggle's own `Kernel push error` is fatal; network errors are retried and then *verified*
   by asking Kaggle whether the kernel exists, because the push frequently lands even when the
   client reports failure.

**Timing note.** The driver's per-job wall clock starts at push, so it includes Kaggle queue and
model-download time (~2 h per batch), and it resets if a wedged client is killed — one cell shows
"1 min" for a run that actually took ~27 min. Real generation time is in each pulled kernel log
(`All done: N generations in Xs`): ~3.7 s/generation on NQ/HAGRID/QAMPARI, ~6 s on RAGTruth, and
it matched the planted runs exactly, which is itself a check that nothing changed.

## 7. Conventions

- **Notebooks are generated, never hand-edited.** All arms of a stage come from one template
  and differ only in the model name and output filename; any code-path divergence between arms
  would confound the comparison. Edit the generator.
- **Stage 04 computes no metrics, deliberately.** It records the answer and its markers and stops.
  Baking a boolean in would lock a scoring rule before it was agreed.
- **Archived run outputs are never regenerated** to fix a bug found later. They are the draws the
  published numbers came from.
- The field name `wikipedia_id` on chunk records is a historical artifact of the KILT-NQ pipeline
  that came first. It is kept deliberately across all four datasets — renaming would touch stages
  03/05, both notebook generators and every existing NQ artifact for zero behavioural gain. Read
  it as "source document id".
