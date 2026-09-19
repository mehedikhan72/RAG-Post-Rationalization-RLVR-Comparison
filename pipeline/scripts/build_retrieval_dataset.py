"""Stages a builder's output as the Kaggle Dataset the 02x notebooks attach.

The 02x generation notebooks rebuild their BM25 index from `chunks.json` and read their
questions from `sampled_questions.json`, both mounted from this dataset. Only those two
files are staged: `corpus.json` is an intermediate the pipeline never reads, and shipping
it would double the dataset size for nothing.

    python pipeline/scripts/build_retrieval_dataset.py --dataset hagrid
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT

# Exactly what 02x reads. Keep this list minimal -- Kaggle Datasets are flat and every
# extra megabyte is re-downloaded by all four generation kernels.
FILES = ["sampled_questions.json", "chunks.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    args = ap.parse_args()

    meta = config.get(args.dataset)
    src = config.runs_dir(args.dataset, "retrieval")
    dest = config.runs_dir(args.dataset, "kaggle_staging", "retrieval")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for name in FILES:
        path = src / name
        if not path.exists():
            raise SystemExit(
                f"missing {path} - run the 06x builder notebook for "
                f"{args.dataset!r} and pull its output first"
            )
        shutil.copy2(path, dest / name)
        rows = json.loads(path.read_text())
        print(f"  {name:26} {path.stat().st_size / 1e6:7.2f} MB  ({len(rows)} records)")

    title = meta["retrieval_title"]
    assert len(title) <= 50, f"title is {len(title)} chars"
    (dest / "dataset-metadata.json").write_text(json.dumps({
        "title": title,
        "id": meta["retrieval_kaggle_id"],
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")

    rel = dest.relative_to(ROOT)
    print(f"\nStaged in {rel}/  ->  {meta['retrieval_kaggle_id']}")
    print(f"First upload:  kaggle datasets create -p {rel}")
    print(f"Later updates: kaggle datasets version -p {rel} -m '<msg>'")


if __name__ == "__main__":
    main()
