"""Stages runs/<dataset>/adversarial/ into a flat Kaggle Dataset folder and prints the push command.

Kaggle Datasets are flat -- nested per-arm folders don't survive the upload cleanly -- so
the 12 files are renamed `{arm}__{variant}.json`, which is exactly what the 04x notebooks
expect at `DATASET_DIR / f"{ARM_KEY}__{variant}.json"`.

    python pipeline/scripts/build_adversarial_dataset.py --dataset nq
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
ARMS = config.ARM_ORDER
VARIANTS = config.VARIANTS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    args = ap.parse_args()

    meta = config.get(args.dataset)
    src = config.runs_dir(args.dataset, "adversarial")
    dest = config.runs_dir(args.dataset, "kaggle_staging", "adversarial")
    dataset_id = meta["adversarial_kaggle_id"]
    dataset_title = meta["adversarial_title"]

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    total = 0
    for arm in ARMS:
        for variant in VARIANTS:
            path = src / arm / f"post_ration_{variant}.json"
            if not path.exists():
                raise SystemExit(
                    f"missing {path} - run "
                    f"03_adversarial_construction.py --dataset {args.dataset} first"
                )
            rows = json.loads(path.read_text())
            out = dest / f"{arm}__{variant}.json"
            out.write_text(json.dumps(rows, indent=2) + "\n")
            total += len(rows)
            print(f"  {out.name:32} {len(rows):4} records")

    assert len(dataset_title) <= 50, f"title is {len(dataset_title)} chars"
    (dest / "dataset-metadata.json").write_text(json.dumps({
        "title": dataset_title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")

    rel = dest.relative_to(ROOT)
    print(f"\n{total} records staged in {rel}/")
    print(f"\nFirst upload:  kaggle datasets create -p {rel}")
    print(f"Later updates: kaggle datasets version -p {rel} -m '<msg>'")


if __name__ == "__main__":
    main()
