"""Step 5 - adversarial post-rationalization dataset construction.

Takes each arm's generated answers, keeps only the ones that obey the strict
"one sentence -> exactly one citation" format, lifts a short span out of each cited
sentence, and plants that span into a passage the model did *not* cite for it. Three
flavours, mirroring Wallat et al. 2025:

  random       an uncited slot -> a chunk from a DIFFERENT question, + span
  rel_uncited  an uncited slot in THIS question's top-5 -> its own text, + span
  cited_other  a slot cited by a DIFFERENT sentence in this answer -> its own text, + span

Every doctored context stays at exactly 5 chunks (Wallat appends the random variant as a
6th document; we substitute instead, so all three variants have the same shape).

Span rule, deliberately blunt: the 4 consecutive words with the most characters in the
sentence. Wallat does no extraction at all -- Command R+ emits its own grounded spans as
`<co: 0,2>Emperor penguin</co: 0,2>` markup and he regex-parses them. Qwen has no
equivalent, so we need *some* rule. It must NOT require the span to also appear in the
cited chunk: matching against the chunk biases selection toward the question's topic
entity, so the planted span ends up being words the model would repeat anyway and the
test loses its bite. Taking the model's own words with no chunk-matching removes that
bias. `span_in_question` is recorded per record so the residual overlap is measurable
rather than assumed away.

No GPU, no model. Scoring is deliberately NOT here -- this script only builds inputs.

Output per arm: runs/<dataset>/adversarial/<arm>/post_ration_{random,rel_uncited,cited_other}.json
plus runs/<dataset>/adversarial/coverage_report.json.

    python pipeline/03_adversarial_construction.py --dataset nq
    python pipeline/03_adversarial_construction.py --dataset nq --arms search_sft
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

random.seed(42)

# One generations file per arm. The 02x notebook directory and the output filename are
# fixed by `generate_generation_notebooks.py`; only the run directory varies by dataset.
ARM_SOURCES = {
    "baseline": ("02a_gen", "generations_nonrl.json"),
    "search_r1": ("02b_gen", "generations_search_r1.json"),
    "research": ("02c_gen", "generations_research.json"),
    "rsearch": ("02d_gen", "generations_rsearch.json"),
    "search_sft": ("02e_gen", "generations_search_sft.json"),
}


def resolve_paths(dataset):
    """All three paths hang off runs/<dataset>/, so a dataset is one string end to end."""
    chunks_path = config.runs_dir(dataset, "retrieval", "chunks.json")
    out_dir = config.runs_dir(dataset, "adversarial")
    arms = {
        arm: config.runs_dir(dataset, "generation", nb_dir, "generation_data", fname)
        for arm, (nb_dir, fname) in ARM_SOURCES.items()
    }
    return chunks_path, out_dir, arms

CITATION_RE = re.compile(r"\[(\d+)\]")
# A compliant sentence is "<text> [N]" with an optional terminator. The capture group is
# the claim with the marker stripped, so a span can never contain the citation itself.
STRICT_RE = re.compile(r"^(.*?)\s*\[(\d+)\]\s*[.?!]?$")
IDK_RE = re.compile(r"\b(i\s+don'?t\s+know|i\s+do\s+not\s+know)\b", re.IGNORECASE)

SPAN_WORDS = 4


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, rows, indent=2):
    with open(path, "w") as f:
        json.dump(rows, f, indent=indent)
        f.write("\n")


def load_chunks(chunks_path):
    return {c["chunk_id"]: c for c in load_json(chunks_path)}


def is_idk(answer):
    return bool(IDK_RE.search(answer))


def parse_strict(answer, retrieved_chunks):
    """Split into sentences and require each to carry exactly one citation at its end.

    Returns a list of {text, marker, chunk_id}, or None if any sentence violates the
    format. Non-compliant answers are dropped rather than salvaged: the entire point of
    the phase-2 prompt is that claim<->citation attribution is unambiguous, and rescuing
    the compliant half of an answer would quietly reintroduce the ambiguity it removes.
    """
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    if not sents:
        return None

    claims = []
    for s in sents:
        markers = CITATION_RE.findall(s)
        if len(markers) != 1:
            return None
        m = STRICT_RE.match(s)
        if m is None:
            return None
        text = m.group(1).strip()
        if not text:
            return None
        idx = int(markers[0])
        if not 1 <= idx <= len(retrieved_chunks):
            return None
        claims.append({
            "text": text,
            "marker": idx,
            "chunk_id": retrieved_chunks[idx - 1]["chunk_id"],
        })
    return claims or None


def extract_span(claim_text, n=SPAN_WORDS):
    """The n consecutive words with the most characters. Ties go to the first window.

    Nothing is matched against the chunk. The span is the model's own wording, which is
    what Wallat's `<co:>` spans are; character count is only a proxy for informativeness,
    chosen because it is deterministic and has no free parameter to tune on one arm and
    then apply to another.
    """
    words = claim_text.split()
    if not words:
        return None
    if len(words) <= n:
        return claim_text
    best_i = max(
        range(len(words) - n + 1),
        key=lambda i: sum(len(w) for w in words[i:i + n]),
    )
    return " ".join(words[best_i:best_i + n])


def doctor_text(original_text, span):
    return f"{original_text}\n{span}"


def build_doctored_context(retrieved_chunks, chunks_lookup, slot_index, doctored_chunk_id, doctored_text):
    out = []
    for i, c in enumerate(retrieved_chunks):
        if i == slot_index:
            out.append({"chunk_id": doctored_chunk_id, "text": doctored_text})
        else:
            out.append({"chunk_id": c["chunk_id"], "text": chunks_lookup[c["chunk_id"]]["text"]})
    return out


def process_arm(arm_name, records, chunks_lookup, cov):
    out = {"random": [], "rel_uncited": [], "cited_other": []}

    # Build the compliant set first. The cross-question pool draws from it only, so a
    # foreign chunk is never sourced from an answer we already discarded.
    compliant = []
    for r in records:
        cov["total_answers"] += 1
        if is_idk(r["generated_answer"]):
            cov["idk"] += 1
            continue
        cov["answered"] += 1
        claims = parse_strict(r["generated_answer"], r["retrieved_chunks"])
        if claims is None:
            cov["non_compliant"] += 1
            continue
        cov["compliant"] += 1
        cov["total_claims"] += len(claims)
        compliant.append((r, claims))

    cross_pool = [
        (r["id"], c["chunk_id"])
        for r, _ in compliant
        for c in r["retrieved_chunks"]
    ]

    for r, claims in compliant:
        question_id = r["id"]
        retrieved_chunks = r["retrieved_chunks"]
        all_cited_ids = {c["chunk_id"] for c in claims}
        uncited_ids = [c["chunk_id"] for c in retrieved_chunks if c["chunk_id"] not in all_cited_ids]

        for claim in claims:
            span = extract_span(claim["text"])
            if span is None:
                cov["no_span"] += 1
                continue

            base = {
                "question_id": question_id,
                "question": r["question"],
                "claim_text": claim["text"],
                "claim_marker": claim["marker"],
                "cited_chunk_id": claim["chunk_id"],
                "extracted_span": span,
                "arm": arm_name,
            }

            def emit(variant, slot_id, source_chunk_id, source_text):
                slot_index = next(
                    i for i, c in enumerate(retrieved_chunks) if c["chunk_id"] == slot_id
                )
                out[variant].append({
                    **base,
                    "variant": variant,
                    "substituted_slot_index": slot_index,
                    "substituted_original_chunk_id": slot_id,
                    "doctored_source_chunk_id": source_chunk_id,
                    "doctored_chunks": build_doctored_context(
                        retrieved_chunks, chunks_lookup, slot_index,
                        source_chunk_id, doctor_text(source_text, span),
                    ),
                })
                cov[f"constructed_{variant}"] += 1

            # --- random: an uncited slot gives way to a chunk from another question ---
            # Never the claim's own cited slot. Deleting the genuine evidence would leave
            # the doctored chunk as the only place the statement appears, which measures
            # something else entirely and is not comparable to Wallat's random rate.
            foreign = [p for p in cross_pool if p[0] != question_id]
            if uncited_ids and foreign:
                cov["eligible_random"] += 1
                _, foreign_chunk_id = random.choice(foreign)
                emit(
                    "random",
                    random.choice(uncited_ids),
                    foreign_chunk_id,
                    chunks_lookup[foreign_chunk_id]["text"],
                )

            # --- rel_uncited: an uncited slot keeps its own text, gains the span ---
            if uncited_ids:
                cov["eligible_rel_uncited"] += 1
                target = random.choice(uncited_ids)
                emit("rel_uncited", target, target, chunks_lookup[target]["text"])

            # --- cited_other: a slot cited by a different sentence in this answer ---
            # `sorted`, not bare set iteration: string hashing is randomised per process
            # (PYTHONHASHSEED), so iterating `all_cited_ids` directly made the subsequent
            # random.choice pick a different slot on every run despite the fixed seed.
            # Only this variant was affected -- `uncited_ids` is built from an ordered
            # list -- but it meant the cited_other cell could not be regenerated.
            other_cited = [cid for cid in sorted(all_cited_ids) if cid != claim["chunk_id"]]
            if other_cited:
                cov["eligible_cited_other"] += 1
                target = random.choice(other_cited)
                emit("cited_other", target, target, chunks_lookup[target]["text"])

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_dataset_arg(ap)
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARM_SOURCES),
                    help="build only these arms, leaving every other arm's archived files "
                         "and coverage entry untouched (adding arms to a finished run)")
    args = ap.parse_args()

    chunks_path, out_dir, arms = resolve_paths(args.dataset)
    if args.arms:
        arms = {a: p for a, p in arms.items() if a in args.arms}
    if not chunks_path.exists():
        raise SystemExit(f"missing {chunks_path} - run the 0x builder notebook first")

    chunks_lookup = load_chunks(chunks_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {args.dataset} ({config.get(args.dataset)['label']})")

    coverage = {}
    for arm_name, path in arms.items():
        if not path.exists():
            print(f"{arm_name:10} MISSING {path} - skipped")
            continue

        # With --arms every arm gets a fresh seed-42 stream -- the stream the first arm
        # (baseline) received in the original all-arms run -- so an added arm's draws do not
        # depend on which other arms happened to be built in the same process.
        if args.arms:
            random.seed(42)
        cov = Counter()
        results = process_arm(arm_name, load_json(path), chunks_lookup, cov)
        coverage[arm_name] = dict(cov)

        arm_dir = out_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        for variant, rows in results.items():
            write_json(arm_dir / f"post_ration_{variant}.json", rows)

        pct = 100 * cov["compliant"] / max(cov["answered"], 1)
        print(
            f"{arm_name:10} answers={cov['total_answers']:4} idk={cov['idk']:3} "
            f"compliant={cov['compliant']:3}/{cov['answered']:3} ({pct:3.0f}%) "
            f"claims={cov['total_claims']:4} | random={len(results['random']):4} "
            f"rel_uncited={len(results['rel_uncited']):4} cited_other={len(results['cited_other']):4}"
        )

    report = out_dir / "coverage_report.json"
    if args.arms and report.exists():
        coverage = {**load_json(report), **coverage}
    write_json(report, coverage)
    print(f"\nCoverage -> {out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
