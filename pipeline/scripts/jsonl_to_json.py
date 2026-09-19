"""One-off migration: every .jsonl in the repo becomes a .json array.

Rationale is readability -- a JSON array opens in any viewer/editor with folding, where
JSONL is a wall of lines. All pipeline code was changed to read and write .json at the
same time; this script converts the data that already exists.

Large corpus files (chunks, corpus, the retrieval checkpoints) are written without
indentation: they are machine-only, and indenting a 36MB corpus adds ~40% for nothing.
Everything a human actually reads gets indent=2.

    python scripts/jsonl_to_json.py [--dry-run]

Safe to re-run: a .jsonl whose .json already exists and is newer is skipped.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Machine-only bulk data - no indentation, nobody reads these by eye.
BULK = {"chunks", "corpus", "gold_docs_checkpoint", "distractor_reservoir_checkpoint"}

SKIP_DIRS = {"RAG-attributions", ".git", "__pycache__"}


def iter_jsonl():
    for p in sorted(ROOT.rglob("*.jsonl")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def convert(path, dry_run=False):
    out = path.with_suffix(".json")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    indent = None if path.stem in BULK else 2
    size_before = path.stat().st_size
    if dry_run:
        return len(rows), size_before, None
    with open(out, "w") as f:
        json.dump(rows, f, indent=indent)
        f.write("\n")
    path.unlink()
    return len(rows), size_before, out.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = list(iter_jsonl())
    if not paths:
        print("No .jsonl files left to convert.")
        return

    total_before = total_after = 0
    for p in paths:
        n, before, after = convert(p, args.dry_run)
        rel = p.relative_to(ROOT)
        total_before += before
        if after is None:
            print(f"  [dry] {rel}  ({n} records, {before / 1e6:.1f} MB)")
        else:
            total_after += after
            flag = " (no indent)" if p.stem in BULK else ""
            print(f"  {rel} -> {rel.with_suffix('.json')}  "
                  f"{n} records, {before / 1e6:.1f} -> {after / 1e6:.1f} MB{flag}")

    print(f"\n{len(paths)} files, {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")
    if not args.dry_run:
        leftover = list(iter_jsonl())
        if leftover:
            print(f"WARNING: {len(leftover)} .jsonl remain", file=sys.stderr)
        else:
            print("No .jsonl remain in the repo.")


if __name__ == "__main__":
    main()
