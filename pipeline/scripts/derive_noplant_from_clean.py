"""W2 - derive the no-planting control outputs that need no GPU at all.

For `rel_uncited` and `cited_other`, stage 03 appends the span to a chunk *in its own slot*
(`emit(variant, target, target, ...)`), so removing the span restores that slot to its
original text and the control context becomes byte-identical to the clean stage-02 context
for the same question. The stage-04 prompt is byte-identical to the stage-02 prompt, and both
decode greedily with the same `max_new_tokens`. The control answer for those two variants is
therefore already on disk: it is `generated_answer` in
`runs/<ds>/generation/02x_gen/generation_data/`.

This script re-emits it in the stage-04 output schema so `05b_score_noplant_control.py` can
read both conditions through one code path.

`random` is NOT derivable -- it substitutes a foreign chunk into an uncited slot, so its
control context is genuinely new and needs the 16 Kaggle runs. Those cells are reported as
pending and left empty.

Identity is asserted per record (chunk ids, order, and every text against `chunks.json`); a
single mismatch aborts the dataset.

    python pipeline/scripts/derive_noplant_from_clean.py --dataset nq
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
CITATION_RE = re.compile(r"\[(\d+)\]")

# Same mapping 03_adversarial_construction.py uses.
ARM_SOURCES = {
    "baseline": ("02a_gen", "generations_nonrl.json"),
    "search_r1": ("02b_gen", "generations_search_r1.json"),
    "research": ("02c_gen", "generations_research.json"),
    "rsearch": ("02d_gen", "generations_rsearch.json"),
    "search_sft": ("02e_gen", "generations_search_sft.json"),
}

# Only these two have control-context == clean-context. Kept as an explicit list rather than
# inferred at runtime so a future variant cannot quietly slip into the free path.
DERIVABLE = ["rel_uncited", "cited_other"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, rows, indent=2):
    with open(path, "w") as f:
        json.dump(rows, f, indent=indent)
        f.write("\n")


def parse_citation_markers(text, n_chunks):
    """Byte-identical to the 04x notebook's parser, so the two conditions parse the same."""
    markers = sorted({int(m) for m in CITATION_RE.findall(text)})
    return [m for m in markers if 1 <= m <= n_chunks]


def derive(dataset):
    chunks_lookup = {
        c["chunk_id"]: c
        for c in load_json(config.runs_dir(dataset, "retrieval", "chunks.json"))
    }
    control_root = config.runs_dir(dataset, "adversarial_noplant")
    out_dir = config.runs_dir(dataset, "postration_noplant", "derived")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for arm, (nb_dir, fname) in ARM_SOURCES.items():
        if arm not in config.ARM_ORDER:  # dropped at the go/no-go gate (config.DROPPED_ARMS)
            continue
        gen_path = config.runs_dir(dataset, "generation", nb_dir, "generation_data", fname)
        if not gen_path.exists():
            raise SystemExit(f"missing {gen_path}")
        clean = {r["id"]: r for r in load_json(gen_path)}

        for variant in config.VARIANTS:
            rows = load_json(control_root / arm / f"post_ration_{variant}.json")
            if variant not in DERIVABLE:
                summary[f"{arm}__{variant}"] = {"derived": 0, "pending_gpu": len(rows)}
                print(f"  {arm:10} {variant:13} {len(rows):5} PENDING (needs the Kaggle run)")
                continue

            out_rows = []
            for i, rec in enumerate(rows):
                g = clean.get(rec["question_id"])
                if g is None:
                    raise SystemExit(
                        f"ABORT {dataset}/{arm}/{variant}[{i}]: no clean generation for "
                        f"question {rec['question_id']}"
                    )
                ctx = rec["doctored_chunks"]
                if [c["chunk_id"] for c in ctx] != [c["chunk_id"] for c in g["retrieved_chunks"]]:
                    raise SystemExit(
                        f"ABORT {dataset}/{arm}/{variant}[{i}]: control chunk order differs "
                        f"from the clean run's retrieved order"
                    )
                for c in ctx:
                    if c["text"] != chunks_lookup[c["chunk_id"]]["text"]:
                        raise SystemExit(
                            f"ABORT {dataset}/{arm}/{variant}[{i}]: slot {c['chunk_id']} "
                            f"text differs from chunks.json"
                        )

                answer = g["generated_answer"]
                out_rows.append({
                    "question_id": rec["question_id"],
                    "question": rec["question"],
                    "arm": arm,
                    "variant": variant,
                    "claim_text": rec["claim_text"],
                    "claim_marker": rec["claim_marker"],
                    "extracted_span": rec["extracted_span"],
                    "cited_chunk_id": rec["cited_chunk_id"],
                    "substituted_slot_index": rec["substituted_slot_index"],
                    "doctored_marker": rec["substituted_slot_index"] + 1,
                    "doctored_source_chunk_id": rec["doctored_source_chunk_id"],
                    "new_answer": answer,
                    "new_citation_markers": parse_citation_markers(answer, len(ctx)),
                    "pair_id": rec["pair_id"],
                    "claim_id": rec["claim_id"],
                    "source": "derived_from_clean_run",
                })

            write_json(out_dir / f"postration_{arm}_{variant}.json", out_rows)
            summary[f"{arm}__{variant}"] = {"derived": len(out_rows), "pending_gpu": 0}
            print(f"  {arm:10} {variant:13} {len(out_rows):5} derived")

    write_json(out_dir / "manifest.json", {
        "dataset": dataset,
        "derived_by": "pipeline/scripts/derive_noplant_from_clean.py",
        "justification": (
            "rel_uncited and cited_other plant into the slot the chunk already occupies, so "
            "the control context is byte-identical to the clean stage-02 context; the stage-04 "
            "prompt is byte-identical to the stage-02 prompt and both decode greedily"
        ),
        "not_derivable": ["random"],
        "cells": summary,
    })
    print(f"\n-> {out_dir.relative_to(ROOT)}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    args = ap.parse_args()
    print(f"dataset: {args.dataset} ({config.get(args.dataset)['label']})")
    derive(args.dataset)


if __name__ == "__main__":
    main()
