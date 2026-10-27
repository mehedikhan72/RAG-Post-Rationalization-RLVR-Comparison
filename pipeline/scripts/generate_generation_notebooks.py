"""Generates the four 02x generation notebooks + their kernel-metadata.json.

Every arm must run byte-identical code apart from Stage 1 (model name, output file).
That is a hard requirement for RQ1: a difference in citation behaviour has to be
attributable to RL training, not to a code path that drifted between arms. Hand-editing
four notebooks cannot guarantee that; generating them from one source can.

The prompt demands one sentence -> exactly one citation. Free-form prose that allows
"[1][2]" or lets a model bunch every marker at the end makes claim<->citation attribution
unrecoverable, and the adversarial test in 03 needs exactly that mapping.

    python pipeline/scripts/generate_generation_notebooks.py --dataset nq
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
V1_DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                   "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")

# Set by main() from the dataset registry; the cell builders below close over it.
RETRIEVAL_DATASET = None

ARMS = [
    {
        "dir": "02a_generation_baseline",
        "short": "02a-baseline",
        "slug": "02a-generation-baseline-qwen2-5-7b-non-rl",
        "title": "02a - Generation: Baseline (Qwen2.5-7B, non-RL)",
        "heading": "02a - Generation: Baseline (Qwen2.5-7B, non-RL)",
        "arm_label": "non-RL baseline",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "output_file": "generations_nonrl.json",
    },
    {
        "dir": "02b_generation_searchr1",
        "short": "02b-search-r1",
        "slug": "02b-generation-search-r1-rl-arm",
        "title": "02b - Generation: Search-R1 (RL arm)",
        "heading": "02b - Generation: Search-R1 (RL arm)",
        "arm_label": "RL arm 1 - Search-R1",
        "model": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo",
        "output_file": "generations_search_r1.json",
    },
    {
        "dir": "02c_generation_research",
        "short": "02c-research",
        "slug": "02c-generation-research-rl-arm",
        "title": "02c - Generation: ReSearch (RL arm)",
        "heading": "02c - Generation: ReSearch (RL arm)",
        "arm_label": "RL arm 2 - ReSearch",
        "model": "agentrl/ReSearch-Qwen-7B-Instruct",
        "output_file": "generations_research.json",
    },
    {
        "dir": "02d_generation_rsearch",
        "short": "02d-r-search",
        "slug": "02d-generation-r-search-rl-arm",
        "title": "02d - Generation: R-Search (RL arm)",
        "heading": "02d - Generation: R-Search (RL arm)",
        "arm_label": "RL arm 3 - R-Search",
        "model": "qingfei1/R-Search-7b-grpo",
        "output_file": "generations_rsearch.json",
    },
    # SFT reference arm (non-RL fine-tune from the same base). Added after the NQ legacy names
    # were frozen, so its NQ slug/title follow the parameterised `{short}-{stage}-{dataset}` rule.
    {
        "dir": "02e_generation_searchsft",
        "short": "02e-search-sft",
        "slug": "02e-search-sft-gen-nq",
        "title": "02e search sft gen nq",
        "heading": "02e - Generation: Search-SFT (SFT control arm)",
        "arm_label": "SFT arm 1 - Search-7B-SFT",
        "model": "Jianwen/Search-7B-SFT",
        "output_file": "generations_search_sft.json",
    },
]


def _lines(text):
    """nbformat wants every source line to keep its trailing newline (all but the last).

    Splitting on "\\n" without re-adding it collapses each cell into one unrunnable
    line - this silently produced broken notebooks once already, so it lives in one
    place now.
    """
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text.strip()),
    }


# --------------------------------------------------------------------------------
# Shared cells - identical across all 4 arms. Do not parameterise anything here
# except via the `arm` dict in stage_1().
# --------------------------------------------------------------------------------

STAGE_0_MD = """
## Stage 0 - Install dependencies

**Pin `torch` explicitly rather than trust Kaggle's preinstalled build.** Kaggle randomly
assigns either a T4 (`sm_75`, Turing) or a P100 (`sm_60`, Pascal) for `enable_gpu: true`.
Kaggle's *default* preinstalled torch (observed: `2.10.0+cu128`) only supports `sm_70`+ --
it has dropped Pascal support entirely, so landing on a P100 with the default torch fails
at `generate()` time with `CUDA error: no kernel image is available for execution on the
device` (hit this empirically 2026-08-08). `torch==2.4.1+cu121` supported both
architectures and is what every v1 run used, on Kaggle's Python 3.12 image.

**v2 (2026-10-13):** Kaggle's default image has since moved to Python 3.13, where this pin
has no wheel. The v2 kernels therefore pin v1's own docker image in kernel-metadata
(`docker_image`), so this cell installs exactly what v1 installed.

**Also remove `torchvision`/`torchaudio`** -- pinning `torch` alone leaves Kaggle's
preinstalled `torchvision`/`torchaudio` (built against `torch 2.10.0+cu128`) in place, and
their ABI mismatch against the downgraded torch breaks `transformers`' lazy import of
`Qwen2ForCausalLM` (surfaces as a confusing `ModuleNotFoundError`). Text generation needs
neither, so drop both rather than version-matching them.
"""

STAGE_0_CODE = """
import subprocess

subprocess.run(
    ["pip", "install", "-q", "torch==2.4.1", "--index-url",
     "https://download.pytorch.org/whl/cu121"],
    check=True,
)
subprocess.run(["pip", "uninstall", "-y", "-q", "torchvision", "torchaudio"], check=True)
subprocess.run(
    ["pip", "install", "-q", "transformers>=4.46", "accelerate", "rank_bm25"],
    check=True,
)
import sys
import torch
import transformers
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"transformers {transformers.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  device {i}: {torch.cuda.get_device_name(i)}, "
              f"capability {torch.cuda.get_device_capability(i)}")
print("Installed.")
"""

STAGE_2_MD = """
## Stage 2 - Load retrieval data + rebuild BM25 index (identical across all 4 arms)

Rebuilding BM25 from the same `chunks.json` (attached as a Kaggle Dataset, not
re-scraped) is deterministic, so every arm sees exactly the same top-5 evidence per
question.

**Indexes `body_text`, not `text`.** Every chunk's `text` field is `"{title}: {piece}"`,
so indexing on it let every chunk of a document share the same title-term boost -- 52% of
questions previously had all 5 top-k slots filled by one document. Indexing on `body_text`
(no title) dropped that to 40%. The model still *sees* the title via `text` in the prompt;
only retrieval scoring changed.
"""

STAGE_2_CODE = """
import json
import re

_TOKEN_RE = re.compile(r"\\w+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, rows, indent=2):
    with open(path, "w") as f:
        json.dump(rows, f, indent=indent)
        f.write("\\n")


from rank_bm25 import BM25Okapi

chunks = load_json(DATASET_DIR / "chunks.json")
questions = load_json(DATASET_DIR / "sampled_questions.json")

tokenized = [_tokenize(c["body_text"]) for c in chunks]
bm25 = BM25Okapi(tokenized)

print(f"Loaded {len(chunks)} chunks, {len(questions)} questions. BM25 index built (body_text).")


def retrieve_top_k(query, k=5):
    scores = bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]
"""

STAGE_3_MD = """
## Stage 3 - Shared prompt-building + citation-parsing (identical across all 4 arms)

We prompt every checkpoint directly with the fixed top-5 retrieved passages -- we do
**not** invoke any RL checkpoint's native search/tool-calling behavior. Letting each model
choose its own retrieval queries would confound 'RL changed citation behavior' with 'RL
changed retrieval quality', a different question. Fixing the evidence isolates what RQ1
asks about.

**The format is strict.** One sentence carries exactly one citation, placed at the end of
that sentence. A free-form prompt permits `[1][2]` and lets models bunch every marker at
the end of the answer, making it impossible to recover which citation supported which
claim -- and the adversarial test needs exactly that mapping. Answers that don't comply
are dropped downstream rather than guessed at, so the per-arm compliance rate is itself a
number worth reading before trusting anything else: it is a post-treatment filter, and a
large spread across arms is a confound, not a detail.
"""

STAGE_3_CODE = '''
CITATION_RE = re.compile(r"\\[(\\d+)\\]")


def build_messages(question, retrieved_chunks):
    context_block = "\\n\\n".join(
        f"[{i + 1}] {c['text']}" for i, c in enumerate(retrieved_chunks)
    )
    user_msg = (
        "Answer the question using ONLY the numbered passages below.\\n\\n"
        "Format your answer as separate sentences. Each sentence must end with exactly "
        "one citation in square brackets, like this:\\n"
        "First sentence here [2]. Second sentence here [4].\\n\\n"
        "Rules:\\n"
        "- Exactly one citation per sentence. Never [1][2].\\n"
        "- The citation goes at the end of the sentence, before the period.\\n"
        "- Do not add any text outside these sentences.\\n"
        "- If the passages do not contain the answer, reply exactly: I don't know\\n\\n"
        f"Passages:\\n{context_block}\\n\\n"
        f"Question: {question}"
    )
    return [{"role": "user", "content": user_msg}]


def parse_citations(generated_text, retrieved_chunks):
    markers = sorted({int(m) for m in CITATION_RE.findall(generated_text)})
    citations = []
    for marker in markers:
        idx = marker - 1
        if 0 <= idx < len(retrieved_chunks):
            c = retrieved_chunks[idx]
            citations.append({
                "marker": marker,
                "chunk_id": c["chunk_id"],
                "wikipedia_id": c["wikipedia_id"],
            })
    return citations
'''

STAGE_4_MD = """
## Stage 4 - Load model

**`device_map={"": 0}`, not `"auto"`.** Kaggle frequently assigns a single P100 (16GB)
rather than T4x2. `"auto"`'s memory estimation played it safe and offloaded a slice of the
model to CPU (`hf_device_map` showed `{0, 'cpu'}`), which measured ~4.75x slower. A 7B
model in fp16 (~14-15GB) fits in a 16GB card with room for the KV cache at this prompt
length, so accelerate's safety margin costs time and buys nothing here.

**v2 (2026-10-13): P100s are no longer offered; the only GPU is T4x2.** A T4 has 14.6 GiB
usable and the fp16 weights alone take 14.2 GiB, so one card OOMs on the first generation.
With two cards visible the layers are split evenly across both (`"balanced"`, with a GPU-only
memory budget so nothing lands on CPU). The arithmetic is unchanged; only the GPU model differs
from v1.
"""

STAGE_4_CODE = """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# v2: Kaggle's only GPU is now T4x2 (14.6 GiB each), too small for 7B fp16 on one card, so
# layers are split evenly across both cards. max_memory leaves no room on CPU, so the
# assert below catches any offload. On a single card this is the v1 placement unchanged.
if torch.cuda.device_count() > 1:
    placement = dict(device_map="balanced",
                     max_memory={i: "13GiB" for i in range(torch.cuda.device_count())})
else:
    placement = dict(device_map={"": 0})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    **placement,
)
model.eval()
placed = set(getattr(model, "hf_device_map", {"": 0}).values())
assert not placed & {"cpu", "disk"}, f"model offloaded off-GPU: {placed}"
print(f"Loaded {MODEL_NAME} on devices: {sorted(map(str, placed))}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB / "
      f"reserved: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
"""

STAGE_5_MD = """
## Stage 5 - Generate cited answers, persist incrementally

Greedy decoding (`do_sample=False`) so the run is reproducible. Records accumulate in a
list and the whole JSON array is rewritten every 20 questions, so a Kaggle session dying
mid-run costs at most 20 records. The file is a valid JSON array at rest because Stage 6
and `03_adversarial_construction.py` both read it back with `json.load`.
"""

STAGE_5_CODE = """
import time

target_questions = questions[:N_LIMIT] if N_LIMIT else questions
out_path = OUT_DIR / OUTPUT_FILE

t0 = time.time()
records = []
for i, q in enumerate(target_questions):
        retrieved = retrieve_top_k(q["question"], k=5)
        messages = build_messages(q["question"], retrieved)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )
        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        record = {
            "id": q["id"],
            "question": q["question"],
            "gold_wikipedia_ids": q["gold_wikipedia_ids"],
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "wikipedia_id": c["wikipedia_id"]}
                for c in retrieved
            ],
            "generated_answer": generated_text,
            "citations": parse_citations(generated_text, retrieved),
        }
        records.append(record)

        # A JSON array can't be appended to line by line the way JSONL could, so rewrite
        # the whole file every 20 records. Same purpose as the old per-record flush: a
        # Kaggle session dying mid-run must not cost the whole batch. Stage 6 and
        # 03_adversarial_construction.py both read this back with json.load, so the file
        # has to be a valid array at rest, not JSONL.
        if (i + 1) % 20 == 0:
            write_json(out_path, records)

        if (i + 1) % 5 == 0 or (i + 1) == len(target_questions):
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(target_questions)} done ({elapsed:.0f}s elapsed)")

write_json(out_path, records)
print(f"Wrote {len(records)} records -> {out_path}")
"""

STAGE_6_MD = """
## Stage 6 - Format compliance check + persist

Read the compliance number before anything downstream. A sentence complies when it ends
with exactly one `[N]`. A free-form prompt yields this shape only rarely and the rate
varies widely by arm; if a large spread survives the strict prompt, the cross-arm
comparison is weakened before the adversarial step even starts.
"""

STAGE_6_CODE = """
import shutil

rows = load_json(out_path)
STRICT_RE = re.compile(r"^(.*?)\\s*\\[(\\d+)\\]\\s*[.?!]?$")


def is_compliant(answer):
    sents = [s for s in re.split(r"(?<=[.!?])\\s+", answer.strip()) if s.strip()]
    if not sents:
        return False
    return all(
        len(CITATION_RE.findall(s)) == 1 and STRICT_RE.match(s) is not None
        for s in sents
    )


def is_idk(answer):
    low = answer.lower()
    return "don't know" in low or "do not know" in low


# Partition in one pass. `r not in idk` would be an O(n^2) dict scan; harmless at the
# 100-400 questions this study runs at, but there is no reason to write it that way.
idk, non_compliant, compliant = [], [], []
for r in rows:
    if is_idk(r["generated_answer"]):
        idk.append(r)
    elif is_compliant(r["generated_answer"]):
        compliant.append(r)
    else:
        non_compliant.append(r)

answered = len(compliant) + len(non_compliant)
print(f"total           : {len(rows)}")
print(f"IDK             : {len(idk)}")
print(f"answered        : {answered}")
print(f"format-compliant: {len(compliant)} "
      f"({100 * len(compliant) / max(answered, 1):.0f}% of answered)")
print("\\n--- 3 sample compliant answers ---")
for r in compliant[:3]:
    print(f"Q: {r['question']}\\nA: {r['generated_answer']}\\n")
print("--- 3 sample NON-compliant answers ---")
for r in non_compliant[:3]:
    print(f"Q: {r['question']}\\nA: {r['generated_answer']}\\n")

shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)
print(f"Zipped to {OUT_DIR}.zip")
"""


def header_md(arm):
    return f"""
# {arm['heading']}

**Arm:** {arm['arm_label']} - model: `{arm['model']}`

**Before running:** Settings -> Internet: **ON**, Accelerator: **GPU**. Input dataset
`{RETRIEVAL_DATASET}` must be attached (handled by kernel-metadata).

One of 4 generation arms sharing **identical** retrieval, prompt-building and generation
code -- only Stage 1 (model name + output filename) differs. That is a hard requirement
for RQ1's controlled comparison: any difference in citation faithfulness must be
attributable to RL training, not to divergent code paths.

All four notebooks are generated by `scripts/generate_generation_notebooks.py`. Edit that
script, not this file.
"""


# Kaggle's private-dataset mount path is NOT stable: the same dataset has appeared at
# /kaggle/input/datasets/<owner>/<slug> in one kernel and flat at /kaggle/input/<slug> in
# another, pushed minutes apart with identical dataset_sources.
# Hardcoding either form silently starves the run, so the mount is located by its CONTENTS.
# Not an f-string: it has braces of its own and nothing in it varies by arm.
FIND_INPUT_DIR = """
import os
from pathlib import Path


def find_input_dir(*required):
    """ + '"""' + """Locate an attached Kaggle dataset by the files it contains.

    Failing loudly here costs seconds; a wrong path costs the whole GPU session.
    """ + '"""' + """
    for dirpath, _dirnames, filenames in os.walk("/kaggle/input"):
        if set(required) <= set(filenames):
            return Path(dirpath)
    listing = "\\n".join(
        os.path.join(dp, f)
        for dp, _dn, fs in os.walk("/kaggle/input") for f in fs
    ) or "  (/kaggle/input is empty)"
    raise SystemExit(
        f"could not find {required} under /kaggle/input. "
        f"Is the dataset attached in kernel-metadata dataset_sources?\\n"
        f"Tree:\\n{listing}"
    )
"""


def stage_1_code(arm):
    return FIND_INPUT_DIR + f'''
# ---- The only block that differs across the 4 generation notebooks ----
MODEL_NAME = "{arm['model']}"
OUTPUT_FILE = "{arm['output_file']}"
# -----------------------------------------------------------------------

DATASET_DIR = find_input_dir("chunks.json", "sampled_questions.json")
OUT_DIR = Path("/kaggle/working/generation_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set to an int (e.g. 5) for a quick validation run; None for the full run.
N_LIMIT = None

print(f"Retrieval data: {{DATASET_DIR}}")
print(f"Model: {{MODEL_NAME}}")
print(f"Output: {{OUT_DIR / OUTPUT_FILE}}")
print(f"N_LIMIT: {{N_LIMIT}}")
'''


def build_notebook(arm):
    return {
        "cells": [
            md(header_md(arm)),
            md(STAGE_0_MD),
            code(STAGE_0_CODE),
            md("## Stage 1 - Config (the only cell that differs between the 4 arms)"),
            code(stage_1_code(arm)),
            md(STAGE_2_MD),
            code(STAGE_2_CODE),
            md(STAGE_3_MD),
            code(STAGE_3_CODE),
            md(STAGE_4_MD),
            code(STAGE_4_CODE),
            md(STAGE_5_MD),
            code(STAGE_5_CODE),
            md(STAGE_6_MD),
            code(STAGE_6_CODE),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kernel_names(dataset, arm, stage):
    """(slug, title) for one arm of one dataset.

    NQ keeps its original, already-pushed names: reusing a slug means PUSHING OVER the
    existing kernel, which would replace the completed NQ run's saved output. Every other
    dataset gets its own suffixed slug.

    Title is the slug with hyphens as spaces, so slugify(title) == slug. Kaggle silently
    relocates a kernel to slugify(title) when the two disagree, which then 409s on every
    later push.
    """
    if config.get(dataset).get("legacy_kernel_names"):
        return arm["slug"], arm["title"]
    slug = f"{arm['short']}-{stage}-{dataset}"
    title = slug.replace("-", " ")
    assert len(title) <= 50, f"{title!r} is {len(title)} chars"
    return slug, title


def build_kernel_metadata(arm, dataset):
    slug, title = kernel_names(dataset, arm, "gen")
    return {
        "id": f"{config.KAGGLE_USER}/{slug}",
        "title": title,
        "code_file": f"{arm['dir']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # v2: pin v1's software. Every v1 kernel ran on this exact image (read back with
        # `kaggle kernels pull -m`); Kaggle's default has since moved to Python 3.13, where the
        # torch==2.4.1 pin cannot install. v1 ran on P100s, which Kaggle no longer offers, so
        # the GPU is T4x2 (model split across both cards, Stage 4). Needs kaggle CLI >= 2.x.
        "docker_image": V1_DOCKER_IMAGE,
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [RETRIEVAL_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    args = ap.parse_args()

    global RETRIEVAL_DATASET
    RETRIEVAL_DATASET = config.get(args.dataset)["retrieval_kaggle_id"]
    out_root = config.notebooks_dir(args.dataset, "02_generation")

    for arm in ARMS:
        d = out_root / arm["dir"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{arm['dir']}.ipynb", "w") as f:
            json.dump(build_notebook(arm), f, indent=1)
            f.write("\n")
        with open(d / "kernel-metadata.json", "w") as f:
            json.dump(build_kernel_metadata(arm, args.dataset), f, indent=2)
            f.write("\n")
        print(f"wrote {d.relative_to(ROOT)}/")

    # Kaggle titles are capped at 50 characters; a longer one is rejected on push.
    for arm in ARMS:
        t = kernel_names(args.dataset, arm, "gen")[1]
        assert len(t) <= 50, f"{t!r} is {len(t)} chars"
    print(f"\nAll 4 generation notebooks regenerated for dataset {args.dataset!r}.")


if __name__ == "__main__":
    main()
