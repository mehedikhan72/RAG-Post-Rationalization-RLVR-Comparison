"""Generates the 06x dataset-builder notebooks + their kernel-metadata.json.

Each builder turns one public dataset into the two files the rest of the pipeline reads:

    sampled_questions.json   {id, question, answer, gold_wikipedia_ids}      x500
    chunks.json              {chunk_id, wikipedia_id, title, text, body_text}

That contract is what `datasets/nq/01_build_nq.ipynb` already produces for KILT-NQ, so
02 -> 03 -> 04 -> 05 run on any of them with one string changed.

Only the LOAD stage differs between the three notebooks. Everything after it -- sampling,
corpus assembly, chunking, BM25, the diversity diagnostic, the distribution report -- is
one shared block emitted verbatim into all three. Same reasoning as
`generate_generation_notebooks.py`: a difference between datasets has to come from the
data, not from a code path that drifted.

    python pipeline/scripts/generate_builder_notebooks.py            # all three
    python pipeline/scripts/generate_builder_notebooks.py --dataset qampari
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT

BUILDERS = {
    "qampari": {
        "nb": "06a_build_qampari",
        "slug": "06a-build-qampari",
        "title": "06a build qampari",
        "source": "ALCE-QAMPARI (`princeton-nlp/ALCE-data`)",
    },
    "hagrid": {
        "nb": "06b_build_hagrid",
        "slug": "06b-build-hagrid",
        "title": "06b build hagrid",
        "source": "HAGRID (`miracl/hagrid`)",
    },
    "ragtruth": {
        "nb": "06c_build_ragtruth",
        "slug": "06c-build-ragtruth",
        "title": "06c build ragtruth",
        "source": "RAGTruth QA subset (`ParticleMedia/RAGTruth`)",
    },
}


def _lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text.strip())}


# ======================================================================================
# Shared cells
# ======================================================================================

STAGE_0_MD = """
## Stage 0 - Install pinned dependencies

`datasets>=4.0` dropped loading-script datasets entirely, and `miracl/hagrid` /
`miracl/miracl-corpus` are still script-based -- the same reason
`01_build_nq.ipynb` pins an older version for `facebook/kilt_tasks`. Pinning here too
keeps every builder on one `datasets` version, so a schema difference between datasets is
never a library-version artifact.

**Restart the kernel after this cell if Kaggle's preinstalled version conflicts.**
CPU is enough for the whole notebook -- there is no model here. Save the GPU quota for 02x.
"""

STAGE_0_CODE = """
!pip install -q "datasets==2.19.0" "fsspec==2023.9.2" rank_bm25
print("Installed. If this is the first install this session, restart the kernel now.")
"""

STAGE_1_MD = """
## Stage 1 - Config

`N_QUESTIONS` and `SEED` match the NQ run exactly. `CHUNK_SIZE_TOKENS = 100` is Wallat et
al.'s chunk size and must not drift between datasets -- the adversarial test plants a
4-word span into a chunk, so chunk length is part of the treatment.
"""

SHARED_CONFIG_CODE = """
DATASET_KEY = "__DATASET_KEY__"

import hashlib
import json
import random
import re
import statistics
from pathlib import Path

# ---- Identical across all builders (and to 01_build_nq.ipynb) ----
N_QUESTIONS = 500
N_DISTRACTORS = 5000
CHUNK_SIZE_TOKENS = 100
SEED = 42
# -------------------------------------------------------------------

OUT_DIR = Path("/kaggle/working/retrieval_pipeline_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def write_json(path, obj, indent=None):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def doc_id(title, text):
    \"\"\"Stable, content-addressed document id.

    NQ has real `wikipedia_id`s; QAMPARI passages carry no guaranteed id and RAGTruth
    passages carry none at all. Hashing title+text gives an id that is stable across runs
    and machines, and that deduplicates identical passages reached from different
    questions -- which is what the corpus needs.
    \"\"\"
    return hashlib.sha1((title + "\\n" + text).encode("utf-8")).hexdigest()[:12]


print(f"dataset: {DATASET_KEY}")
print(f"N_QUESTIONS={N_QUESTIONS}  N_DISTRACTORS={N_DISTRACTORS}  SEED={SEED}")
print(f"output: {OUT_DIR}")
"""

STAGE_3_MD = """
## Stage 3 - Sample 500 questions, uniformly at random

`random.sample` without replacement from the eligible pool, seeded -- the same draw
`01_build_nq.ipynb` makes. No stratification and no filtering after the draw: the sample
is meant to represent its own dataset, not to be bent into matching NQ's shape.

Stage 3b then *checks* that the draw is representative rather than assuming it.
"""

STAGE_3_CODE = """
assert len(pool) >= N_QUESTIONS, (
    f"pool has only {len(pool)} eligible questions, need {N_QUESTIONS}"
)

sample_idx = random.sample(range(len(pool)), N_QUESTIONS)
sample = [pool[i] for i in sample_idx]

print(f"Pool: {len(pool)} eligible questions -> sampled {len(sample)}")
print(f"Example: {sample[0]['question']!r}")
print(f"  passages: {len(sample[0]['passages'])}, gold idxs: {sample[0]['gold_passage_idxs']}")
"""

STAGE_3B_MD = """
## Stage 3b - Distribution check: is the sample representative of the pool?

A uniform draw is *expected* to be representative; this cell produces the evidence that it
is, so the claim is checkable rather than asserted. Three covariates -- question length,
answer length, passages per question -- summarised for pool and sample, plus a two-sample
Kolmogorov-Smirnov statistic.

**This is a report, not a filter.** Nothing is re-drawn on a large KS statistic. Re-drawing
until a test passes would bias the sample, which is exactly what the uniform draw avoids.
A large D here is something to disclose in the write-up, not to sample away.
"""

STAGE_3B_CODE = """
from scipy import stats

def covariates(rows):
    return {
        "question_len_words": [len(r["question"].split()) for r in rows],
        "answer_len_words": [len(r["answer"].split()) for r in rows],
        "n_passages": [len(r["passages"]) for r in rows],
    }


def describe(xs):
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 3),
        "sd": round(statistics.pstdev(xs), 3),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


pool_cov, sample_cov = covariates(pool), covariates(sample)
dist_report = {"dataset": DATASET_KEY, "seed": SEED,
               "n_pool": len(pool), "n_sample": len(sample), "covariates": {}}

for name in pool_cov:
    d, p = stats.ks_2samp(pool_cov[name], sample_cov[name])
    dist_report["covariates"][name] = {
        "pool": describe(pool_cov[name]),
        "sample": describe(sample_cov[name]),
        "ks_statistic": round(float(d), 4),
        "ks_pvalue": round(float(p), 4),
    }

write_json(OUT_DIR / "sample_distribution_report.json", dist_report, indent=2)

print(f"{'covariate':<20} {'pool mean':>10} {'sample mean':>12} {'KS D':>8} {'p':>8}")
for name, r in dist_report["covariates"].items():
    print(f"{name:<20} {r['pool']['mean']:>10.2f} {r['sample']['mean']:>12.2f} "
          f"{r['ks_statistic']:>8.4f} {r['ks_pvalue']:>8.4f}")
print("\\n(p > 0.05 => no detectable difference between sample and pool)")
"""

STAGE_4_MD = """
## Stage 4 - Assemble the corpus

The corpus is the sampled questions' own passages, deduplicated by content hash, plus
whatever distractors the LOAD stage produced. Distractors are load-bearing, not filler: a
corpus containing only each question's own gold passages makes retrieval trivial, and a
faithfulness test on trivial retrieval measures nothing.
"""

STAGE_4_CODE = """
corpus = {}


def add_doc(title, text):
    did = doc_id(title, text)
    if did not in corpus:
        corpus[did] = {"wikipedia_id": did, "title": title, "text": text}
    return did


questions = []
for r in sample:
    passage_ids = [add_doc(p["title"], p["text"]) for p in r["passages"]]
    gold = sorted({passage_ids[i] for i in r["gold_passage_idxs"]
                   if 0 <= i < len(passage_ids)})
    questions.append({
        "id": str(r["id"]),
        "question": r["question"],
        "answer": r["answer"],
        "gold_wikipedia_ids": gold,
    })

n_from_questions = len(corpus)
for d in distractors:
    add_doc(d["title"], d["text"])

n_gold_empty = sum(1 for q in questions if not q["gold_wikipedia_ids"])
print(f"{n_from_questions} unique docs from the sampled questions")
print(f"+ {len(corpus) - n_from_questions} distractors = {len(corpus)} documents")
print(f"questions with no gold document: {n_gold_empty}/{len(questions)}")

write_json(OUT_DIR / "sampled_questions.json", questions, indent=2)
write_json(OUT_DIR / "corpus.json", list(corpus.values()))
"""

STAGE_5_MD = """
## Stage 5 - Chunk into ~100-token passages

Two text fields per chunk, exactly as `01_build_nq.ipynb` writes them:

- `text` = `"{title}: {piece}"` -- what the model is shown at generation time
- `body_text` = `piece` alone -- what BM25 indexes

Splitting them is the retrieval-diversity fix from 2026-08-08: indexing title+body gave
every chunk of a document the same title-term boost, so top-5 filled up with chunks of one
document. HAGRID and RAGTruth passages can have no title at all, in which case the two
fields are equal and the fix is a no-op for them.
"""

STAGE_5_CODE = """
chunks = []
for doc in corpus.values():
    words = doc["text"].split()
    for i in range(0, len(words), CHUNK_SIZE_TOKENS):
        piece = " ".join(words[i:i + CHUNK_SIZE_TOKENS])
        if not piece.strip():
            continue
        title = doc["title"]
        chunks.append({
            "chunk_id": f"{doc['wikipedia_id']}_{i // CHUNK_SIZE_TOKENS}",
            "wikipedia_id": doc["wikipedia_id"],
            "title": title,
            "text": f"{title}: {piece}" if title else piece,
            "body_text": piece,
        })

write_json(OUT_DIR / "chunks.json", chunks)
n_untitled = sum(1 for c in chunks if not c["title"])
print(f"Chunked {len(corpus)} documents into {len(chunks)} passages "
      f"of ~{CHUNK_SIZE_TOKENS} tokens.")
print(f"chunks with no title (text == body_text): {n_untitled}")

# Titling must be all-or-nothing across the corpus. If gold passages were untitled and
# distractors titled (or vice versa), the "{title}: " prefix would mark which passages are
# relevant without reading them -- a cue the model could cite on instead of content.
gold_ids = {g for q in questions for g in q["gold_wikipedia_ids"]}
gold_untitled = [c for c in chunks if c["wikipedia_id"] in gold_ids and not c["title"]]
n_gold_chunks = sum(1 for c in chunks if c["wikipedia_id"] in gold_ids)
assert n_untitled in (0, len(chunks)), (
    f"mixed titling: {n_untitled}/{len(chunks)} chunks untitled, of which "
    f"{len(gold_untitled)}/{n_gold_chunks} are gold -- titles must be uniform across "
    f"the corpus or they leak which passages are relevant"
)
print("titling is uniform across the corpus")
"""

STAGE_6_MD = """
## Stage 6 - BM25 index + retrieval diversity diagnostic

Same diagnostic `01_build_nq.ipynb` Stage 6b runs, over all 500 questions. Two numbers
decide whether this dataset is worth spending GPU on:

- **gold-hit@5** -- does retrieval surface a gold document at all? NQ reached 95.4%.
- **mean unique documents in top-5** -- NQ reached 2.05 after the body-text fix. If top-5
  collapses to a single document, the `cited_other` adversarial variant has nothing to
  work with.

A dataset far outside those is a retrieval problem to fix *before* generation, not after.
"""

STAGE_6_CODE = """
from collections import Counter

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\\w+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


bm25 = BM25Okapi([tokenize(c["body_text"]) for c in chunks])
print(f"BM25 index built over {len(chunks)} chunks (indexed on body_text).")


def retrieve_top_k(query, k=5):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top]


unique_counts, gold_hits, single_doc = [], 0, []
gold_evaluable = 0
for q in questions:
    top5 = retrieve_top_k(q["question"], k=5)
    doc_ids = [c["wikipedia_id"] for c in top5]
    unique_counts.append(len(set(doc_ids)))
    if q["gold_wikipedia_ids"]:
        gold_evaluable += 1
        if set(q["gold_wikipedia_ids"]) & set(doc_ids):
            gold_hits += 1
    if len(set(doc_ids)) == 1:
        single_doc.append({"id": q["id"], "doc_id": doc_ids[0],
                           "is_gold": doc_ids[0] in q["gold_wikipedia_ids"]})

n = len(questions)
report = {
    "dataset": DATASET_KEY,
    "n_questions": n,
    "n_chunks": len(chunks),
    "n_documents": len(corpus),
    "unique_doc_count_distribution": {str(k): v for k, v in
                                      sorted(Counter(unique_counts).items())},
    "mean_unique_docs_in_top5": round(sum(unique_counts) / n, 3),
    # Denominator is questions that HAVE a gold label; QAMPARI can legitimately have none.
    "gold_evaluable_questions": gold_evaluable,
    "gold_hit_at_5": gold_hits,
    "gold_hit_at_5_pct": round(100 * gold_hits / max(gold_evaluable, 1), 2),
    "n_single_document_top5": len(single_doc),
    "n_single_document_top5_pct": round(100 * len(single_doc) / n, 2),
    "single_document_cases": single_doc,
}
write_json(OUT_DIR / "retrieval_diversity_report.json", report, indent=2)

print(f"gold-hit@5             : {gold_hits}/{gold_evaluable} "
      f"({report['gold_hit_at_5_pct']}%)      [NQ: 95.4%]")
print(f"mean unique docs top-5 : {report['mean_unique_docs_in_top5']}"
      f"                  [NQ: 2.05]")
print(f"single-document top-5  : {report['n_single_document_top5_pct']}%"
      f"                     [NQ: 40%]")
print(f"distribution           : {report['unique_doc_count_distribution']}")
"""

STAGE_7_MD = """
## Stage 7 - Validate against the contract, then persist

The assertions below are the same ones the pipeline will rely on downstream. Failing here,
in the notebook that produced the data, is much cheaper than failing in a GPU kernel.
"""

STAGE_7_CODE = """
import shutil

q = json.loads((OUT_DIR / "sampled_questions.json").read_text())
c = json.loads((OUT_DIR / "chunks.json").read_text())

assert len(q) == N_QUESTIONS, f"expected {N_QUESTIONS} questions, got {len(q)}"
assert {k for r in q for k in r} == {"id", "question", "answer", "gold_wikipedia_ids"}
assert {k for r in c for k in r} == {"chunk_id", "wikipedia_id", "title", "text", "body_text"}
assert len({r["id"] for r in q}) == len(q), "duplicate question ids"
assert len({r["chunk_id"] for r in c}) == len(c), "duplicate chunk ids"
assert all(r["question"].strip() for r in q), "empty question text"
assert all(r["body_text"].strip() for r in c), "empty chunk body"

known = {r["wikipedia_id"] for r in c}
dangling = {g for r in q for g in r["gold_wikipedia_ids"] if g not in known}
assert not dangling, f"{len(dangling)} gold ids missing from chunks.json"

shutil.make_archive("/kaggle/working/retrieval_pipeline_data", "zip", OUT_DIR)
print("CONTRACT OK")
print(f"  {len(q)} questions, {len(c)} chunks, {len(corpus)} documents")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:38} {f.stat().st_size / 1e6:8.2f} MB")
print("\\nZipped to /kaggle/working/retrieval_pipeline_data.zip")
"""


# ======================================================================================
# Per-dataset LOAD stages
#
# Each must bind exactly two names:
#   pool        list of {id, question, answer, passages:[{title,text}], gold_passage_idxs}
#   distractors list of {title, text}
# ======================================================================================

QAMPARI_MD = """
## Stage 2 - LOAD: ALCE-QAMPARI

ALCE ships **top-100 retrieved passages per question** (GTR over the 21M-passage Wikipedia
corpus), so there is no corpus to rebuild and no 5.9M-document scan like NQ needed --
the retrieval pool comes with the dataset.

**No external distractors.** Pooling 500 questions x 100 passages gives ~50k documents,
already larger than NQ's corpus, and each question's other 99 passages act as distractors
for every other question. Adding more would be noise on top of noise.

**Gold labels are derived, not given.** QAMPARI supplies answer strings, not gold document
ids, so a passage counts as gold when it contains one of the question's answer aliases as a
normalized substring. Questions where no passage matches keep an empty gold list -- that is
allowed by the contract, because `gold_wikipedia_ids` feeds only the Stage 6 diagnostic and
is never read by `03_adversarial_construction.py` or `05_score_postrationalization.py`.

The tar layout is discovered by walking the extracted tree rather than hardcoded: the
filename inside the archive has moved before.
"""

QAMPARI_CODE = """
import shutil
import tarfile

from huggingface_hub import hf_hub_download


tar_path = hf_hub_download(repo_id="princeton-nlp/ALCE-data",
                           filename="ALCE-data.tar", repo_type="dataset")
extract_to = Path("/kaggle/temp/alce")   # scratch: NOT collected as kernel output
extract_to.mkdir(parents=True, exist_ok=True)
with tarfile.open(tar_path) as tf:
    tf.extractall(extract_to)

# Locate the eval file by NAME, searched anywhere in the tree -- the archive's internal
# directory has moved before, but the filename is what the ALCE configs reference
# (configs/qampari_*_gtr_top100.yaml -> data/qampari_eval_gtr_top100.json).
#
# Matching loosely on "qampari" + "top100" is NOT safe: the tar also ships
# qampari_eval_dpr_top100.json (a different retriever) and
# qampari_eval_gtr_top100_reranked_oracle.json (oracle-reranked, which would leak gold
# passages into the top-5 and make the retrieval trivially easy). Pick the exact one.
WANTED = "qampari_eval_gtr_top100.json"
matches = [p for p in extract_to.rglob(WANTED)]
if not matches:
    raise SystemExit(f"{WANTED} not found. Tree:\\n" +
                     "\\n".join(str(p) for p in sorted(extract_to.rglob("*.json"))))
eval_path = matches[0]
print(f"eval file: {eval_path.name}")

raw = json.loads(eval_path.read_text())
print(f"{len(raw)} QAMPARI eval items")

PUNCT = re.compile(r"[^a-z0-9 ]+")


def norm(s):
    return PUNCT.sub(" ", s.lower()).strip()


def answer_aliases(item):
    \"\"\"QAMPARI answers are a list of alias-lists (one group per acceptable answer).\"\"\"
    out = []
    for group in item.get("answers", []):
        out.extend(group if isinstance(group, list) else [group])
    return [norm(str(a)) for a in out if str(a).strip()]


pool = []
for i, item in enumerate(raw):
    docs = item.get("docs") or []
    if not docs:
        continue
    passages = [{"title": d.get("title", "") or "", "text": d.get("text", "") or ""}
                for d in docs]
    aliases = answer_aliases(item)
    gold_idxs = [j for j, p in enumerate(passages)
                 if any(a and a in norm(p["text"]) for a in aliases)]
    pool.append({
        "id": item.get("id", f"qampari_{i}"),
        "question": item["question"],
        # Diagnostic only. Joined for a readable length covariate in Stage 3b.
        "answer": "; ".join(g[0] if isinstance(g, list) else str(g)
                            for g in item.get("answers", []) if g),
        "passages": passages,
        "gold_passage_idxs": gold_idxs,
    })

del raw
shutil.rmtree(extract_to, ignore_errors=True)   # ~450MB of sibling eval files

distractors = []   # the pooled top-100s already provide them
print(f"pool: {len(pool)} questions with passages")
print(f"mean passages/question: {sum(len(p['passages']) for p in pool) / len(pool):.1f}")
print(f"questions with >=1 derived gold passage: "
      f"{sum(1 for p in pool if p['gold_passage_idxs'])}/{len(pool)}")
"""

HAGRID_MD = """
## Stage 2 - LOAD: HAGRID

Train (1,922) and dev (716) are pooled to 2,638 eligible queries. Both splits carry the
same human attributability annotations and we regenerate answers ourselves, so the split
boundary carries no information here -- pooling only widens the draw.

**`quotes` has no guaranteed `title` field**, so it is probed on the first record rather
than assumed; when absent, `title` is empty and `text == body_text` downstream.

**Distractors come from `miracl/miracl-corpus`** (English, the corpus HAGRID's quotes were
drawn from). Note honestly what the sampling is: a streaming shuffle draws from a buffer
and a shuffled shard order, *not* uniformly over all ~32.9M passages. That is fine for
distractors, whose only job is to be topically unrelated -- but the write-up must not
describe it as a uniform sample of MIRACL.
"""

HAGRID_CODE = """
from datasets import load_dataset


ds = load_dataset("miracl/hagrid", trust_remote_code=True)
print({k: len(v) for k, v in ds.items()})

first_quote = next(q for split in ds.values() for r in split
                   for q in (r.get("quotes") or []))
print(f"quote fields: {sorted(first_quote.keys())}")
HAS_TITLE = "title" in first_quote
print(f"quotes carry a title field: {HAS_TITLE}")

pool = []
for split_name, split in ds.items():
    for r in split:
        quotes = r.get("quotes") or []
        if not quotes:
            continue
        answers = r.get("answers") or []
        passages = [{"title": (q.get("title") or "") if HAS_TITLE else "",
                     "text": q.get("text", "") or ""} for q in quotes]
        passages = [p for p in passages if p["text"].strip()]
        if not passages:
            continue
        pool.append({
            "id": f"{r['query_id']}",
            "question": r["query"],
            "answer": (answers[0].get("answer", "") if answers else ""),
            "passages": passages,
            # Every quote is human-labelled relevant, so all of them are gold.
            "gold_passage_idxs": list(range(len(passages))),
        })

print(f"pool: {len(pool)} queries with quotes")
print(f"mean quotes/query: {sum(len(p['passages']) for p in pool) / len(pool):.2f}")

corpus_stream = load_dataset("miracl/miracl-corpus", "en", split="train",
                             streaming=True, trust_remote_code=True)

# miracl-corpus passages DO carry a title; HAGRID's quotes do not. Keeping the title on
# distractors only would render every distractor as "Title: text" and every gold passage
# as bare text -- a visible cue in the prompt that separates gold from distractor without
# reading either. The model could then cite on formatting rather than content, which is
# precisely the behaviour this study measures. Titles are dropped so the whole corpus is
# uniformly untitled, matching the quotes.
distractors = [{"title": "", "text": d.get("text", "") or ""}
               for d in corpus_stream.shuffle(seed=SEED, buffer_size=10_000)
                                     .take(N_DISTRACTORS)]
distractors = [d for d in distractors if d["text"].strip()]
print(f"distractors sampled from miracl-corpus/en: {len(distractors)} (titles dropped)")
"""

RAGTRUTH_MD = """
## Stage 2 - LOAD: RAGTruth (QA subset)

**QA only.** RAGTruth's other two task types -- data-to-text over Yelp, summarization over
CNN/DailyMail -- have no question, and this pipeline is question-driven. That is a forced
constraint, not a preference. It leaves 989 of the 2,965 source items.

**`response.jsonl` is deliberately not used.** Its word-level hallucination annotations are
on GPT-4 / Llama-2 / Mistral outputs; our four arms produce different text, so the labels
do not transfer. Only `(question, passages)` carries over.

**`source_info["passages"]` is one concatenated string**, not a list -- segments prefixed
`passage 1:`, `passage 2:` and separated by blank lines. It is split on that marker.

**No titles anywhere in this dataset**, so `title = ""` and `text == body_text` for every
chunk. Downstream is unaffected (02x displays `text`, BM25 indexes `body_text`), but
RAGTruth prompts lack the title disambiguation the other datasets have. Worth a sentence in
the methodology; not worth inventing a title to paper over.
"""

RAGTRUTH_CODE = """
import urllib.request


SRC_URL = ("https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
           "main/dataset/source_info.jsonl")
local = Path("/kaggle/temp/source_info.jsonl")   # scratch: NOT kernel output
local.parent.mkdir(parents=True, exist_ok=True)
if not local.exists():
    urllib.request.urlretrieve(SRC_URL, local)
print(f"downloaded {local.stat().st_size / 1e6:.1f} MB")

rows = [json.loads(line) for line in local.read_text().splitlines() if line.strip()]
print("task_type counts:",
      {t: sum(1 for r in rows if r.get("task_type") == t)
       for t in sorted({r.get("task_type") for r in rows})})

PASSAGE_RE = re.compile(r"passage\\s*\\d+\\s*:", re.IGNORECASE)

pool = []
for r in rows:
    if r.get("task_type") != "QA":
        continue
    info = r.get("source_info") or {}
    question = (info.get("question") or "").strip()
    blob = info.get("passages") or ""
    if not question or not blob:
        continue
    parts = [p.strip() for p in PASSAGE_RE.split(blob) if p.strip()]
    if not parts:
        continue
    passages = [{"title": "", "text": p} for p in parts]
    pool.append({
        "id": str(r["source_id"]),
        "question": question,
        "answer": "",          # RAGTruth ships no gold answer
        "passages": passages,
        "gold_passage_idxs": list(range(len(passages))),
    })

assert pool, "no QA items parsed -- check the passage-splitting regex"
mean_p = sum(len(p["passages"]) for p in pool) / len(pool)
assert mean_p >= 2, f"passage split looks wrong: mean {mean_p:.2f} passages/item"
print(f"pool: {len(pool)} QA items, mean {mean_p:.2f} passages/item")

marco = load_dataset("microsoft/ms_marco", "v2.1", split="validation",
                     streaming=True, trust_remote_code=True)
seen, distractors = set(), []
for row in marco.shuffle(seed=SEED, buffer_size=10_000):
    for txt in (row.get("passages") or {}).get("passage_text", []):
        t = (txt or "").strip()
        if t and t not in seen:
            seen.add(t)
            distractors.append({"title": "", "text": t})
    if len(distractors) >= N_DISTRACTORS:
        break
distractors = distractors[:N_DISTRACTORS]
print(f"distractors sampled from MS MARCO v2.1: {len(distractors)}")
"""

LOAD_STAGES = {
    "qampari": (QAMPARI_MD, QAMPARI_CODE),
    "hagrid": (HAGRID_MD, HAGRID_CODE),
    "ragtruth": (RAGTRUTH_MD, RAGTRUTH_CODE),
}

# `datasets` is imported inside the HAGRID load stage; RAGTruth needs it too but imports
# urllib first, so make the import explicit there rather than relying on stage ordering.
RAGTRUTH_CODE = RAGTRUTH_CODE.replace(
    "import urllib.request\n", "import urllib.request\n\nfrom datasets import load_dataset\n")
LOAD_STAGES["ragtruth"] = (RAGTRUTH_MD, RAGTRUTH_CODE)


def stage_1_code(dataset):
    """The config cell. `DATASET_KEY` is the only line that differs between builders."""
    return SHARED_CONFIG_CODE.replace("__DATASET_KEY__", dataset)


def header_md(dataset, spec, meta):
    return f"""
# {spec['nb'].replace('_', ' ')} - build the {meta['label']} retrieval data

**Source:** {spec['source']}
**Produces:** `sampled_questions.json` + `chunks.json` -- the same two files
`datasets/nq/01_build_nq.ipynb` produces for KILT-NQ, so `02x` -> `03` -> `04x` -> `05`
run on this dataset unchanged.

**Before running:** Settings -> Internet: **ON**. Accelerator: **None/CPU** -- there is no
model in this notebook.

Only Stage 2 (LOAD) is specific to {meta['label']}. Stages 3-7 are emitted verbatim into
all three builder notebooks by `pipeline/scripts/generate_builder_notebooks.py`. Edit that
script, not this file.

Upload the output as Kaggle Dataset `{meta['retrieval_kaggle_id']}` when the run finishes.
"""


def build_notebook(dataset):
    spec, meta = BUILDERS[dataset], config.get(dataset)
    load_md, load_code = LOAD_STAGES[dataset]
    return {
        "cells": [
            md(header_md(dataset, spec, meta)),
            md(STAGE_0_MD), code(STAGE_0_CODE),
            md(STAGE_1_MD), code(stage_1_code(dataset)),
            md(load_md), code(load_code),
            md(STAGE_3_MD), code(STAGE_3_CODE),
            md(STAGE_3B_MD), code(STAGE_3B_CODE),
            md(STAGE_4_MD), code(STAGE_4_CODE),
            md(STAGE_5_MD), code(STAGE_5_CODE),
            md(STAGE_6_MD), code(STAGE_6_CODE),
            md(STAGE_7_MD), code(STAGE_7_CODE),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel_metadata(dataset):
    spec = BUILDERS[dataset]
    return {
        "id": f"{config.KAGGLE_USER}/{spec['slug']}",
        "title": spec["title"],
        "code_file": f"{spec['nb']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(BUILDERS), default=None,
                    help="build one notebook (default: all three)")
    args = ap.parse_args()

    targets = [args.dataset] if args.dataset else sorted(BUILDERS)
    for dataset in targets:
        spec = BUILDERS[dataset]
        # Kaggle caps titles at 50 chars, and silently relocates the kernel to
        # slugify(title) when the id disagrees.
        assert len(spec["title"]) <= 50, f"{spec['title']!r} is {len(spec['title'])} chars"
        assert slugify(spec["title"]) == spec["slug"], (
            f"slug {spec['slug']!r} != slugify(title) {slugify(spec['title'])!r}"
        )

        d = config.dataset_dir(dataset)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{spec['nb']}.ipynb", "w") as f:
            json.dump(build_notebook(dataset), f, indent=1)
            f.write("\n")
        with open(d / "kernel-metadata.json", "w") as f:
            json.dump(build_kernel_metadata(dataset), f, indent=2)
            f.write("\n")
        print(f"wrote {d.relative_to(ROOT)}/{spec['nb']}.ipynb")

    print(f"\n{len(targets)} builder notebook(s) generated.")


if __name__ == "__main__":
    main()
