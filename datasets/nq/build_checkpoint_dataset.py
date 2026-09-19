"""Stages the N=500 scan checkpoints as a Kaggle Dataset so the retrieval notebook can
resume instead of re-scanning KILT Wikipedia.

Why this exists: the N=500 run scanned 5,893,807 / 5,903,530 KILT docs (99.84%) over 3.4
hours, found 487/488 gold docs and filled the 5000-doc distractor reservoir, then died on
a transient `ClientResponseError` from the HF streaming endpoint ~9,700 docs from the end.
The `finally:` block checkpointed everything, so all the expensive data survived -- but the
exception propagated, so every downstream cell was skipped.

Resuming via `IterableDataset.skip(5_893_807)` would NOT help: skip is lazy and still
streams every skipped record over HTTP. Since the reservoir is full and the only gold doc
still outstanding (`4647690`) is absent from this dump version, the remaining 0.16% of the
scan cannot change the result. The notebook therefore detects a complete-enough scan from
these checkpoints and goes straight to merge -> chunk -> index.

    python datasets/nq/build_checkpoint_dataset.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs" / "nq" / "retrieval"
DEST = ROOT / "runs" / "nq" / "kaggle_staging" / "scan_checkpoint"

DATASET_ID = "mehedikhan72/rag-faithfulness-scan-checkpoint"
# Kaggle rejects dataset titles over 50 characters.
DATASET_TITLE = "RAG Faithfulness Scan Checkpoint"

FILES = [
    "gold_docs_checkpoint.json",
    "distractor_reservoir_checkpoint.json",
    "gold_ids.json",
    "sampled_questions.json",
    "scan_progress.json",
]


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    for name in FILES:
        src = SRC / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, DEST / name)
        payload = json.loads(src.read_text())
        n = len(payload) if isinstance(payload, (list, dict)) else "-"
        print(f"  {name:38} {src.stat().st_size / 1e6:7.2f} MB  ({n} records)")

    assert len(DATASET_TITLE) <= 50, f"title is {len(DATASET_TITLE)} chars"
    (DEST / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_TITLE,
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")

    rel = DEST.relative_to(ROOT)
    print(f"\nStaged in {rel}/")
    print(f"First upload:  kaggle datasets create -p {rel}")
    print(f"Later updates: kaggle datasets version -p {rel} -m '<msg>'")


if __name__ == "__main__":
    main()
