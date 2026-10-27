# Attributable Post-Rationalization in RAG Citations

Code and data for *Attributable Post-Rationalization in RAG Citations: A Controlled Reproduction
and an RLVR Comparison*.

A RAG model can cite a passage it did not use. We measure this with the planting probe of
Wallat et al. (2025). We take four words from a sentence the model wrote, append them to a
retrieved passage it did not cite, re-run the model, and check whether its citation moves to
that passage. We add a no-planting control so that only the effect of the plant is counted.

We compare five models that share the `Qwen2.5-7B-Instruct` base: the base model, three search
agents trained with reinforcement learning from verifiable rewards (Search-R1, ReSearch,
R-Search), and one search model fine-tuned without RL (Search-SFT). We evaluate them on four QA
datasets (KILT-NQ, ALCE-QAMPARI, HAGRID, RAGTruth).

**Results in short.**
- **The base model** moves its citation for 9–15% of claims on the Wikipedia-based datasets
  (2.9% on RAGTruth).
- **RL fine-tuning** does not reduce this. One RL agent drifts slightly upwards, and the non-RL
  fine-tune drifts the same way.
- **Two raters** checked 100 flagged cases. In about 70% of them, the passage the citation moved
  to did not support the claim before planting.

## Repository layout

```
pipeline/
  config.py                         dataset registry, Kaggle ids, model arms, contrast families
  03_adversarial_construction.py    plants the span; builds the three probe variants
  05_score_postrationalization.py   original same-sentence flag (cluster-robust, Holm)
  05b_score_noplant_control.py      attributable rates, differences between models, bootstrap
  notebooks/                        generated Kaggle GPU notebooks (one per model, dataset, stage)
  scripts/                          notebook generators, Kaggle drivers, reports, validation
  tests/                            checks that each control record differs only by the span
datasets/<dataset>/                 builder notebooks: question sample + retrieval corpus
data/validation/                    flag-validation sample and both raters' labels
reports/                            study summary and technical notes
```

Generated outputs go to `runs/` (model outputs and scores) and `results/` (tables). Neither is
tracked.

- **[`reports/PROJECT_STATE.md`](reports/PROJECT_STATE.md):** the study in full. Design, every
  number, findings and limitations.
- **[`reports/TECHNICAL_NOTES.md`](reports/TECHNICAL_NOTES.md):** code map, reproduction commands,
  scoring internals, verification, and Kaggle pitfalls.

## Setup

The local stages (span planting, scoring, reports) use only the Python standard library, on
Python 3.10 or later. GPU stages run as Kaggle notebooks, which install their own pinned
dependencies.

```
pip install -r requirements.txt   # only the Kaggle CLI, needed to push and pull GPU notebooks
```

## Reproducing the results

### A. Re-score from the released outputs (minutes, no GPU)

Download `runs/` (generations, re-runs and retrieval corpora) into the repository root from
https://drive.google.com/drive/folders/1BCPEJtvHYb9MIPiH1s2iaD2pcwLAiJ0A, then run:

```
for ds in nq qampari hagrid ragtruth; do
  python pipeline/05_score_postrationalization.py --dataset $ds
done
python pipeline/05b_score_noplant_control.py --all --dump
python pipeline/05b_score_noplant_control.py --all --outcome cite_anywhere
python pipeline/scripts/report_sft_control.py      # -> results/sft_control_tables.md
python pipeline/scripts/report_flag_validation.py  # -> results/validation/tables.md
```

`results/sft_control_tables.md` contains every model-comparison number in the paper. Scoring is
deterministic: the same inputs give byte-identical outputs.

### B. Re-run everything (free-tier Kaggle GPUs)

1. In `pipeline/config.py`, set `KAGGLE_USER` and the dataset ids to your own Kaggle account.
   Then regenerate the notebooks with `pipeline/scripts/generate_generation_notebooks.py` and
   `pipeline/scripts/generate_postration_notebooks.py`, once per dataset (`--dataset <name>`).
2. Build each dataset with its notebook in `datasets/<dataset>/` and stage it for Kaggle with
   `pipeline/scripts/build_retrieval_dataset.py`.
3. Run the base model and the three RL agents with `pipeline/scripts/run_pipeline.py --dataset
   <dataset>`. Run their no-planting controls with `pipeline/scripts/run_noplant_control.py`.
4. Run Search-SFT, from generation through scoring, with `pipeline/scripts/run_sft_arms.py`.

The drivers push GPU notebooks two at a time and resume from whatever is already on disk. All of
this is in `reports/TECHNICAL_NOTES.md` §2. Kaggle's GPUs have changed since the original runs;
§6 of the technical notes says how the notebooks handle that.

### Flag validation

`pipeline/scripts/extract_flag_sample.py` rebuilds the 100-item sample in `data/validation/`
(seed 42). `pipeline/scripts/build_review_sheet.py` builds the blind review sheets, and
`pipeline/scripts/report_flag_validation.py` computes agreement and the results.

## Citation

If you use this code, please cite the paper (citation to be added on publication) and the
original probe:

> Jonas Wallat, Maria Heuss, Maarten de Rijke, and Avishek Anand. 2025. Correctness is not
> Faithfulness in Retrieval Augmented Generation Attributions. In *Proceedings of ICTIR 2025*.
