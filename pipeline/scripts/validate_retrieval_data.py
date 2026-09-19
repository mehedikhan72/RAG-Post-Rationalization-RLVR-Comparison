"""Checks a builder's output against the contract, side by side with NQ.

The point is not "does it look plausible" but "will 02x -> 03 -> 04x -> 05 run on it
unchanged". Every assertion here mirrors something the pipeline actually does:

  02x   json.load(chunks.json); BM25 over `body_text`; shows `text` in the prompt
  03    chunks_lookup[chunk_id]["text"]; retrieved_chunks carry chunk_id + wikipedia_id
  05    never touches these files, but inherits whatever 03 emitted

NQ is loaded alongside as the reference column, so a shape that drifted is visible rather
than inferred.

    python pipeline/scripts/validate_retrieval_data.py --dataset hagrid
    python pipeline/scripts/validate_retrieval_data.py --all
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

QUESTION_FIELDS = {"id", "question", "answer", "gold_wikipedia_ids"}
CHUNK_FIELDS = {"chunk_id", "wikipedia_id", "title", "text", "body_text"}
N_EXPECTED = 500


def load(dataset):
    d = config.runs_dir(dataset, "retrieval")
    q, c = d / "sampled_questions.json", d / "chunks.json"
    if not q.exists() or not c.exists():
        return None
    return json.loads(q.read_text()), json.loads(c.read_text()), d


def check(dataset, questions, chunks, strict_n=True):
    """Returns (failures, warnings). A failure means the pipeline would break."""
    fail, warn = [], []

    def req(cond, msg):
        if not cond:
            fail.append(msg)

    # --- schema ---
    qkeys = {k for r in questions for k in r}
    ckeys = {k for r in chunks for k in r}
    req(qkeys == QUESTION_FIELDS,
        f"sampled_questions fields {sorted(qkeys)} != {sorted(QUESTION_FIELDS)}")
    req(ckeys == CHUNK_FIELDS,
        f"chunks fields {sorted(ckeys)} != {sorted(CHUNK_FIELDS)}")

    # --- types: 03 does str formatting and set ops on these ---
    req(all(isinstance(r["id"], str) for r in questions), "question id is not a str")
    req(all(isinstance(r["question"], str) for r in questions), "question is not a str")
    req(all(isinstance(r["gold_wikipedia_ids"], list) for r in questions),
        "gold_wikipedia_ids is not a list")
    req(all(isinstance(r["chunk_id"], str) for r in chunks), "chunk_id is not a str")

    # --- cardinality / uniqueness ---
    if strict_n:
        req(len(questions) == N_EXPECTED,
            f"{len(questions)} questions, expected {N_EXPECTED}")
    req(len({r["id"] for r in questions}) == len(questions), "duplicate question ids")
    req(len({r["chunk_id"] for r in chunks}) == len(chunks), "duplicate chunk_ids")

    # --- content ---
    req(all(r["question"].strip() for r in questions), "empty question text")
    req(all(r["body_text"].strip() for r in chunks), "empty chunk body_text")
    req(all(r["text"].strip() for r in chunks), "empty chunk text")

    # --- referential integrity: 03 looks chunks up by chunk_id ---
    known_docs = {r["wikipedia_id"] for r in chunks}
    dangling = {g for r in questions for g in r["gold_wikipedia_ids"]
                if g not in known_docs}
    req(not dangling, f"{len(dangling)} gold ids absent from chunks.json")

    # --- chunk_id must decompose as f"{wikipedia_id}_{n}" (03 never parses it, but the
    #     convention is what makes a chunk traceable to its document by eye) ---
    bad = [r["chunk_id"] for r in chunks
           if not r["chunk_id"].startswith(r["wikipedia_id"] + "_")][:3]
    req(not bad, f"chunk_id does not follow '<wikipedia_id>_<n>': {bad}")

    # --- warnings: legal, but worth seeing ---
    untitled = sum(1 for r in chunks if not r["title"])
    if untitled:
        warn.append(f"{untitled}/{len(chunks)} chunks have no title "
                    f"(text == body_text) -- expected for HAGRID/RAGTruth")
    no_gold = sum(1 for r in questions if not r["gold_wikipedia_ids"])
    if no_gold:
        warn.append(f"{no_gold}/{len(questions)} questions have no gold document "
                    f"-- diagnostic only, never read by 03 or 05")
    return fail, warn


def stats(questions, chunks):
    wlen = [len(r["body_text"].split()) for r in chunks]
    per_doc = Counter(r["wikipedia_id"] for r in chunks)
    return {
        "questions": len(questions),
        "chunks": len(chunks),
        "documents": len(per_doc),
        "mean chunk words": round(statistics.mean(wlen), 1),
        "mean chunks/doc": round(len(chunks) / len(per_doc), 2),
        "mean question words": round(
            statistics.mean(len(r["question"].split()) for r in questions), 1),
        "untitled chunks %": round(100 * sum(1 for r in chunks if not r["title"])
                                   / len(chunks), 1),
        "questions w/o gold %": round(100 * sum(1 for r in questions
                                                if not r["gold_wikipedia_ids"])
                                      / len(questions), 1),
    }


def report_line(name, ref, cur):
    return f"  {name:<22} {ref:>14} {cur:>14}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(config.DATASETS), default=None)
    ap.add_argument("--all", action="store_true", help="every dataset with built data")
    args = ap.parse_args()

    nq = load("nq")
    if nq is None:
        raise SystemExit("NQ retrieval data missing -- it is the reference column")
    nq_q, nq_c, _ = nq
    # NQ ran at 499 after one unanswerable question was dropped; that is the record.
    nq_fail, _ = check("nq", nq_q, nq_c, strict_n=False)
    if nq_fail:
        print("WARNING: the NQ reference itself fails the contract:")
        for f in nq_fail:
            print(f"  - {f}")
    nq_stats = stats(nq_q, nq_c)

    targets = ([d for d in sorted(config.DATASETS) if d != "nq"]
               if args.all else [args.dataset])
    targets = [t for t in targets if t]
    if not targets:
        raise SystemExit("pass --dataset <name> or --all")

    exit_code = 0
    for dataset in targets:
        loaded = load(dataset)
        print(f"\n{'=' * 62}\n{dataset.upper()}  ({config.get(dataset)['label']})\n{'=' * 62}")
        if loaded is None:
            print("  no data at " + str(config.runs_dir(dataset, "retrieval")))
            exit_code = 1
            continue
        q, c, path = loaded
        fail, warn = check(dataset, q, c)
        cur = stats(q, c)

        print(report_line("", "NQ (reference)", dataset))
        print("  " + "-" * 52)
        for k in nq_stats:
            print(report_line(k, nq_stats[k], cur[k]))

        if warn:
            print("\n  notes:")
            for w in warn:
                print(f"    - {w}")
        if fail:
            exit_code = 1
            print("\n  CONTRACT VIOLATIONS:")
            for f in fail:
                print(f"    FAIL {f}")
        else:
            print("\n  CONTRACT OK -- 02x/03/04x/05 can run on this unchanged")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
