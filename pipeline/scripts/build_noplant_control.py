"""W2 - build the no-planting control datasets by TRANSFORMING the archived stage-03 records.

The published post-rationalization flag compares a doctored run against nothing. Between the
clean run and the doctored run two things change at once: the contents of the doctored slot,
and the span appended to it. This builder produces the condition that changes only the first,
so the second can be attributed:

    attributable = P(doctored slot cited | planted) - P(doctored slot cited | control)

**Nothing is regenerated.** `03_adversarial_construction.py` had a set-iteration
nondeterminism bug in `cited_other` (fixed 2026-08-29); re-running it would draw different
slots than the published numbers came from and the two conditions would stop being paired.
So the control is derived record-by-record from `runs/<ds>/adversarial/`, positionally, and
the pairing key is the record's index in its file.

The transform is exactly one edit: the doctored slot's text loses its `"\n" + span` suffix
(`doctor_text` in stage 03). Every other byte of every record is carried over verbatim. The
result is asserted equal to `chunks.json[doctored_source_chunk_id]["text"]` for every record;
one failure aborts the whole dataset rather than skipping quietly.

    python pipeline/scripts/build_noplant_control.py --dataset nq
    python pipeline/scripts/build_noplant_control.py --dataset nq --stage
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
ARMS = config.ARM_ORDER
VARIANTS = config.VARIANTS

# The stage-04 user-message template, byte-identical to the one in
# generate_postration_notebooks.py STAGE_3_CODE (and to stage 02's). Hashed into every
# manifest so a prompt drift between the planted and the control run is detectable rather
# than assumed away. Placeholders are left literal: this hashes the TEMPLATE, not a
# rendered prompt.
PROMPT_TEMPLATE = (
    "Answer the question using ONLY the numbered passages below.\n\n"
    "Format your answer as separate sentences. Each sentence must end with exactly "
    "one citation in square brackets, like this:\n"
    "First sentence here [2]. Second sentence here [4].\n\n"
    "Rules:\n"
    "- Exactly one citation per sentence. Never [1][2].\n"
    "- The citation goes at the end of the sentence, before the period.\n"
    "- Do not add any text outside these sentences.\n"
    "- If the passages do not contain the answer, reply exactly: I don't know\n\n"
    "Passages:\n{context_block}\n\n"
    "Question: {question}"
)
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, rows, indent=2):
    with open(path, "w") as f:
        json.dump(rows, f, indent=indent)
        f.write("\n")


def strip_span(record, chunks_lookup):
    """Return the control twin of one planted record, or raise if the suffix isn't there.

    Raising rather than returning None is deliberate: a record whose slot text does not end
    in "\\n<span>" means the join rule assumed here (A2 in the recon note) is wrong for that
    record, and every downstream number would be silently mispaired.
    """
    slot = record["substituted_slot_index"]
    span = record["extracted_span"]
    doctored = record["doctored_chunks"][slot]
    suffix = "\n" + span
    source_id = record["doctored_source_chunk_id"]

    if doctored["chunk_id"] != source_id:
        raise ValueError(f"slot chunk_id {doctored['chunk_id']} != source {source_id}")
    if not doctored["text"].endswith(suffix):
        raise ValueError(f"slot text does not end in the span suffix: {span!r}")

    clean_text = doctored["text"][: -len(suffix)]
    expected = chunks_lookup[source_id]["text"]
    if clean_text != expected:
        raise ValueError(
            f"stripped slot != chunks.json[{source_id}] "
            f"({len(clean_text)} vs {len(expected)} chars)"
        )

    out = dict(record)
    out["doctored_chunks"] = [
        {"chunk_id": c["chunk_id"], "text": (clean_text if i == slot else c["text"])}
        for i, c in enumerate(record["doctored_chunks"])
    ]
    out["span_removed"] = span
    return out


def build(dataset):
    chunks_lookup = {
        c["chunk_id"]: c
        for c in load_json(config.runs_dir(dataset, "retrieval", "chunks.json"))
    }
    src = config.runs_dir(dataset, "adversarial")
    dest = config.runs_dir(dataset, "adversarial_noplant")

    counts = {}
    identical_to_clean = {}
    total = 0
    for arm in ARMS:
        for variant in VARIANTS:
            path = src / arm / f"post_ration_{variant}.json"
            if not path.exists():
                raise SystemExit(f"missing {path} - run 03_adversarial_construction.py first")
            rows = load_json(path)
            out_rows = []
            for i, rec in enumerate(rows):
                try:
                    twin = strip_span(rec, chunks_lookup)
                except ValueError as e:
                    raise SystemExit(
                        f"ABORT {dataset}/{arm}/{variant} record {i} "
                        f"(question {rec['question_id']}, marker {rec['claim_marker']}): {e}"
                    )
                twin["pair_id"] = i
                twin["claim_id"] = f"{rec['question_id']}::{rec['claim_marker']}::{i}"
                twin["control_of"] = f"{dataset}/{arm}/{variant}/{i}"
                out_rows.append(twin)

            arm_dir = dest / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            write_json(arm_dir / f"post_ration_{variant}.json", out_rows)
            counts[f"{arm}__{variant}"] = len(out_rows)
            # For rel_uncited and cited_other the doctored slot IS its own original slot, so
            # stripping the span restores the clean stage-02 context exactly. Recorded per
            # cell because it is what makes derive_noplant_from_clean.py legitimate.
            identical_to_clean[f"{arm}__{variant}"] = sum(
                1 for r in out_rows
                if r["doctored_source_chunk_id"] == r["substituted_original_chunk_id"]
            )
            total += len(out_rows)
            print(f"  {arm:10} {variant:13} {len(out_rows):5} records")

    manifest = {
        "dataset": dataset,
        "source": str(src.relative_to(ROOT)),
        "builder": "pipeline/scripts/build_noplant_control.py",
        "transform": 'doctored slot text loses its "\\n" + extracted_span suffix; nothing else changes',
        "prompt_sha256": PROMPT_SHA256,
        "total_records": total,
        "counts": counts,
        "context_identical_to_clean_run": identical_to_clean,
    }
    write_json(dest / "manifest.json", manifest)
    print(f"\n{total} control records -> {dest.relative_to(ROOT)}/")
    print(f"prompt sha256: {PROMPT_SHA256}")
    return dest


def stage(dataset):
    """Flatten to {arm}__{variant}.json, exactly as build_adversarial_dataset.py does.

    Same filenames as the planted staging on purpose: the 04x notebook template locates its
    input as `DATASET_DIR / f"{ARM_KEY}__{variant}.json"`, and keeping that line untouched is
    what makes the control notebook a one-line diff of the planted one. The two conditions are
    separated by the Kaggle dataset id, not by filename.
    """
    meta = config.get(dataset)
    src = config.runs_dir(dataset, "adversarial_noplant")
    dest = config.runs_dir(dataset, "kaggle_staging", "adversarial_noplant")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    total = 0
    for arm in ARMS:
        for variant in VARIANTS:
            rows = load_json(src / arm / f"post_ration_{variant}.json")
            write_json(dest / f"{arm}__{variant}.json", rows)
            total += len(rows)

    title = meta["noplant_title"]
    assert len(title) <= 50, f"title is {len(title)} chars"
    write_json(dest / "dataset-metadata.json", {
        "title": title,
        "id": meta["noplant_kaggle_id"],
        "licenses": [{"name": "CC0-1.0"}],
    })
    rel = dest.relative_to(ROOT)
    print(f"\n{total} records staged in {rel}/")
    print(f"First upload:  kaggle datasets create -p {rel}")
    print(f"Later updates: kaggle datasets version -p {rel} -m '<msg>'")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    ap.add_argument("--stage", action="store_true",
                    help="also flatten into kaggle_staging/adversarial_noplant/")
    args = ap.parse_args()

    print(f"dataset: {args.dataset} ({config.get(args.dataset)['label']})")
    build(args.dataset)
    if args.stage:
        stage(args.dataset)


if __name__ == "__main__":
    main()
