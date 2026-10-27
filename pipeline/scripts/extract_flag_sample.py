"""Flag validation - draw a stratified sample of flagged `rel_uncited` records for manual review.

A flag (claim-aligned, planted condition) says: after four words of the claim were appended to
a retrieved passage the model had not cited, the model cited that passage for the claim. The
open question is whether the passage *already* supported the claim before the plant -- a
legitimate alternative source -- or whether the four words are the only tie. This script only
assembles the evidence; two raters judge it (paper, Appendix H; labels in data/validation/).

Sample: 25 flagged records per dataset, 5 per arm, at most one record per question within a
(dataset, arm) cell, seed 42. Items are shuffled and given neutral ids so a reviewer does not
see the arm or dataset while judging.

Per item: the question, the model's clean answer, the claim and its original citation, the
original cited passage, the doctored passage *before* planting (the raw chunk) and the span
appended to it, the full five-passage doctored context, and the new answer with its aligned
sentence. Two lexical helpers (share of the claim's content words found in each passage) are
included as a cross-check on the human labels, not as a substitute for them.

    python pipeline/scripts/extract_flag_sample.py
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
_spec = importlib.util.spec_from_file_location(
    "score05b", ROOT / "pipeline" / "05b_score_noplant_control.py")
score05b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score05b)

# Same mapping 03_adversarial_construction.py uses.
ARM_SOURCES = {
    "baseline": ("02a_gen", "generations_nonrl.json"),
    "search_r1": ("02b_gen", "generations_search_r1.json"),
    "research": ("02c_gen", "generations_research.json"),
    "rsearch": ("02d_gen", "generations_rsearch.json"),
    "search_sft": ("02e_gen", "generations_search_sft.json"),
}
DATASET_ORDER = ["nq", "qampari", "hagrid", "ragtruth"]
VARIANT = "rel_uncited"
PER_CELL = 5
SEED = 42
OUT = ROOT / "data" / "validation" / "flag_sample.json"

STOPWORDS = set("""a an the of in on at to for from by with and or but is are was were be been
being as that this these those it its his her their they he she which who whom what when where
how than then also has have had not no do does did into over under about after before during
s""".split())


def load_json(path):
    with open(path) as f:
        return json.load(f)


def content_words(text):
    return {w for w in score05b.normalize(text).split() if w not in STOPWORDS}


def lexical_support(claim, passage):
    words = content_words(claim)
    return round(len(words & content_words(passage)) / len(words), 3) if words else 0.0


def key(r):
    return (r["question_id"], r["claim_text"], r["claim_marker"], r["doctored_source_chunk_id"])


def build_cell(dataset, arm, chunks, rng):
    scored = load_json(config.runs_dir(dataset, "scoring", f"scored_{arm}_{VARIANT}.json"))
    adversarial = {key(r): r for r in load_json(
        config.runs_dir(dataset, "adversarial", arm, f"post_ration_{VARIANT}.json"))}
    nb_dir, fname = ARM_SOURCES[arm]
    clean = {r["id"]: r for r in load_json(
        config.runs_dir(dataset, "generation", nb_dir, "generation_data", fname))}

    flagged = []
    for rec in scored:
        out = score05b.score_record_aligned(rec, "planted")
        if out["cite_aligned"]:
            flagged.append((rec, out))
    rng.shuffle(flagged)

    items, seen = [], set()
    for rec, out in flagged:
        if rec["question_id"] in seen:
            continue
        seen.add(rec["question_id"])
        adv = adversarial[key(rec)]
        span = rec["extracted_span"]
        planted_text = adv["doctored_chunks"][adv["substituted_slot_index"]]["text"]
        pre_plant = chunks[rec["doctored_source_chunk_id"]]["text"]
        if planted_text != f"{pre_plant}\n{span}":
            raise SystemExit(f"plant mismatch: {dataset} {arm} {rec['question_id']}")
        cited = chunks[rec["cited_chunk_id"]]["text"]
        segs = score05b.segment_answer(rec["new_answer"])
        items.append({
            "dataset": dataset,
            "arm": arm,
            "question_id": rec["question_id"],
            "question": rec["question"],
            "clean_answer": clean[rec["question_id"]]["generated_answer"].strip(),
            "claim": rec["claim_text"],
            "original_marker": rec["claim_marker"],
            "original_passage": cited,
            "doctored_marker": rec["doctored_marker"],
            "doctored_passage_pre_plant": pre_plant,
            "planted_span": span,
            "doctored_context": [
                {"marker": i + 1, "chunk_id": c["chunk_id"],
                 "text": chunks[c["chunk_id"]]["text"],
                 "planted": i == adv["substituted_slot_index"]}
                for i, c in enumerate(adv["doctored_chunks"])],
            "new_answer": rec["new_answer"].strip(),
            "aligned_sentence": segs[out["aligned_index"]]["text"],
            "aligned_jaccard": out["aligned_jaccard"],
            "lex_support_original": lexical_support(rec["claim_text"], cited),
            "lex_support_doctored_pre_plant": lexical_support(rec["claim_text"], pre_plant),
        })
        if len(items) == PER_CELL:
            break
    if len(items) < PER_CELL:
        raise SystemExit(f"{dataset} {arm}: only {len(items)} flagged questions")
    return items, len(flagged)


def main():
    rng = random.Random(SEED)
    items, population = [], {}
    for ds in DATASET_ORDER:
        chunks = {c["chunk_id"]: c for c in load_json(config.runs_dir(ds, "retrieval", "chunks.json"))}
        for arm in config.ARM_ORDER:
            cell, n = build_cell(ds, arm, chunks, rng)
            items += cell
            population[f"{ds}/{arm}"] = n
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item_id"] = f"F{i:03d}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"variant": VARIANT, "outcome": "cite_aligned", "condition": "planted",
                   "seed": SEED, "per_cell": PER_CELL, "flagged_population": population,
                   "items": [{"item_id": it.pop("item_id"), **it} for it in items]},
                  f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{len(items)} items from {sum(population.values())} flagged records -> "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
