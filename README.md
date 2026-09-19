# RAG Citation Faithfulness under RLVR Training

Undergraduate thesis. Does RL training (RLVR) make retrieval-augmented models *fake* citations
more than their non-RL base model does?

**Start here:**

| file | what it is |
|---|---|
| [`reports/PROJECT_STATE.md`](reports/PROJECT_STATE.md) | **The main document.** Current state, design, every result, the findings, limitations, what's next. Written to be the basis of a paper. |
| [`reports/TECHNICAL_NOTES.md`](reports/TECHNICAL_NOTES.md) | Implementation: code map, reproduction commands, scoring internals, verification, and the bugs that cost real time. |
| `reports/<dataset>/noplant_control.md` | Auto-generated per-dataset detail tables. Regenerate with `pipeline/scripts/report_noplant_control.py --all`. |

**Status:** RQ1 measurement complete on four datasets, four arms, including the no-planting
control. 48 Kaggle GPU kernel runs, 36,690 scored records. Next: an SFT control arm and/or the
write-up — see `PROJECT_STATE.md` §9.

**Data.** `runs/` — retrieval corpora, generations, scoring output (~430MB) — is not tracked in
this repo. Full contents: https://drive.google.com/drive/folders/1BCPEJtvHYb9MIPiH1s2iaD2pcwLAiJ0A

**Compute:** free-tier Kaggle only (2×T4 or 1×P100, 2 concurrent sessions, 30 h/week).
