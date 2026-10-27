"""§3 diff test - the control differs from its planted twin in exactly one way.

Run against every dataset that has both directories:

    python pipeline/tests/test_noplant_control.py
    python pipeline/tests/test_noplant_control.py --dataset nq --sample 200

Not pytest: nothing else in this repo uses it, and this needs to be runnable on a Kaggle
session with no extra install.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ARMS = config.ARM_ORDER
VARIANTS = config.VARIANTS
ADDED_KEYS = {"span_removed", "pair_id", "claim_id", "control_of"}


def load(path):
    with open(path) as f:
        return json.load(f)


def check_pair(planted, control, chunks_lookup):
    """Returns a list of failure strings; empty means the pair is clean."""
    fails = []
    slot = planted["substituted_slot_index"]

    # (d) no other key differs, and only the four pairing keys are added
    if set(control) - set(planted) != ADDED_KEYS:
        fails.append(f"unexpected added keys {set(control) - set(planted)}")
    if set(planted) - set(control):
        fails.append(f"dropped keys {set(planted) - set(control)}")
    for k in planted:
        if k == "doctored_chunks":
            continue
        if planted[k] != control[k]:
            fails.append(f"field {k!r} differs")

    p_chunks, c_chunks = planted["doctored_chunks"], control["doctored_chunks"]
    if len(p_chunks) != len(c_chunks):
        return fails + ["chunk count differs"]

    # (c) chunk order identical
    if [c["chunk_id"] for c in p_chunks] != [c["chunk_id"] for c in c_chunks]:
        fails.append("chunk order differs")

    # (a) all five slot texts identical except the doctored one
    for i, (p, c) in enumerate(zip(p_chunks, c_chunks)):
        if i == slot:
            continue
        if p["text"] != c["text"]:
            fails.append(f"non-doctored slot {i} text differs")

    # (b) the doctored one differs only by the removed suffix, and equals chunks.json
    suffix = "\n" + planted["extracted_span"]
    if p_chunks[slot]["text"] != c_chunks[slot]["text"] + suffix:
        fails.append("doctored slot is not exactly control + '\\n' + span")
    src = chunks_lookup.get(control["doctored_source_chunk_id"])
    if src is None or c_chunks[slot]["text"] != src["text"]:
        fails.append("control slot != chunks.json source text")
    if control["span_removed"] != planted["extracted_span"]:
        fails.append("span_removed != extracted_span")

    return fails


def run(dataset, sample, seed):
    rng = random.Random(seed)
    chunks_lookup = {
        c["chunk_id"]: c
        for c in load(config.runs_dir(dataset, "retrieval", "chunks.json"))
    }
    planted_root = config.runs_dir(dataset, "adversarial")
    control_root = config.runs_dir(dataset, "adversarial_noplant")
    if not control_root.exists():
        print(f"{dataset:9} SKIP - no adversarial_noplant/")
        return True

    checked = 0
    failures = []
    for arm in ARMS:
        for variant in VARIANTS:
            p = load(planted_root / arm / f"post_ration_{variant}.json")
            c = load(control_root / arm / f"post_ration_{variant}.json")
            if len(p) != len(c):
                failures.append(f"{arm}/{variant}: length {len(p)} vs {len(c)}")
                continue
            idx = range(len(p)) if len(p) <= sample else rng.sample(range(len(p)), sample)
            for i in idx:
                if c[i]["pair_id"] != i:
                    failures.append(f"{arm}/{variant}[{i}]: pair_id {c[i]['pair_id']}")
                for f in check_pair(p[i], c[i], chunks_lookup):
                    failures.append(f"{arm}/{variant}[{i}]: {f}")
                checked += 1

    if failures:
        print(f"{dataset:9} FAIL - {len(failures)} problems in {checked} pairs")
        for f in failures[:20]:
            print(f"    {f}")
        return False
    cells = len(ARMS) * len(VARIANTS)
    print(f"{dataset:9} ok - {checked} pairs checked, {cells}/{cells} cells")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, choices=sorted(config.DATASETS))
    ap.add_argument("--sample", type=int, default=200,
                    help="pairs per arm x variant cell (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else sorted(config.DATASETS)
    ok = all([run(d, args.sample, args.seed) for d in datasets])
    if not ok:
        raise SystemExit(1)
    print("\nall good")


if __name__ == "__main__":
    main()
