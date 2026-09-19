"""The dataset registry. Adding a dataset means adding one entry here.

Everything downstream of a builder notebook (02 generation -> 03 adversarial ->
04 re-inference -> 05 scoring) is dataset-agnostic and resolves its paths and its
Kaggle dataset ids through this file. That is the whole point: a second dataset must
not be a second copy of the pipeline.

A NOTE ON `wikipedia_id`
    Chunk records carry a field called `wikipedia_id` even for QAMPARI, HAGRID and
    RAGTruth, whose passages are not Wikipedia articles. The name is a historical
    artifact of the KILT-NQ pipeline that came first. It is kept deliberately:
    renaming it would touch 03, 05, both notebook generators and every existing NQ
    artifact, for zero behavioural gain. Read it as "source document id".
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Kaggle dataset slugs are lowercase-hyphen and titles are capped at 50 characters
# (a longer title is rejected at push time, not at create time).
DATASETS = {
    "nq": {
        "label": "NQ",
        "description": "KILT-NaturalQuestions",
        "retrieval_kaggle_id": "mehedikhan72/rag-faithfulness-retrieval-data",
        "retrieval_title": "RAG Faithfulness - Retrieval Pipeline Data",
        "adversarial_kaggle_id": "mehedikhan72/rag-faithfulness-adversarial-data",
        "adversarial_title": "RAG Faithfulness Adversarial",
        "noplant_kaggle_id": "mehedikhan72/rag-faith-nq-adversarial-noplant",
        "noplant_title": "RAG Faith NQ Adversarial Noplant",
        # NQ's kernels were pushed before the pipeline was parameterised. Their slugs
        # and titles are frozen: changing them would orphan the completed runs whose
        # output `kaggle kernels output` still pulls.
        "legacy_kernel_names": True,
    },
    "qampari": {
        "label": "QAMPARI",
        "description": "ALCE-QAMPARI - multi-answer list QA over Wikipedia",
        "retrieval_kaggle_id": "mehedikhan72/rag-faith-qampari-retrieval",
        "retrieval_title": "RAG Faith QAMPARI Retrieval",
        "adversarial_kaggle_id": "mehedikhan72/rag-faith-qampari-adversarial",
        "adversarial_title": "RAG Faith QAMPARI Adversarial",
        "noplant_kaggle_id": "mehedikhan72/rag-faith-qampari-adv-noplant",
        "noplant_title": "RAG Faith QAMPARI Adversarial Noplant",
        "legacy_kernel_names": False,
    },
    "hagrid": {
        "label": "HAGRID",
        "description": "HAGRID - MIRACL-derived attributable generative IR",
        "retrieval_kaggle_id": "mehedikhan72/rag-faith-hagrid-retrieval",
        "retrieval_title": "RAG Faith HAGRID Retrieval",
        "adversarial_kaggle_id": "mehedikhan72/rag-faith-hagrid-adversarial",
        "adversarial_title": "RAG Faith HAGRID Adversarial",
        "noplant_kaggle_id": "mehedikhan72/rag-faith-hagrid-adv-noplant",
        "noplant_title": "RAG Faith HAGRID Adversarial Noplant",
        "legacy_kernel_names": False,
    },
    "ragtruth": {
        "label": "RAGTruth",
        "description": "RAGTruth QA subset - MS MARCO sourced",
        "retrieval_kaggle_id": "mehedikhan72/rag-faith-ragtruth-retrieval",
        "retrieval_title": "RAG Faith RAGTruth Retrieval",
        "adversarial_kaggle_id": "mehedikhan72/rag-faith-ragtruth-adversarial",
        "adversarial_title": "RAG Faith RAGTruth Adversarial",
        "noplant_kaggle_id": "mehedikhan72/rag-faith-ragtruth-adv-noplant",
        "noplant_title": "RAG Faith RAGTruth Adversarial Noplant",
        "legacy_kernel_names": False,
    },
}

KAGGLE_USER = "mehedikhan72"

ARM_ORDER = ["baseline", "search_r1", "research", "rsearch"]
VARIANTS = ["random", "rel_uncited", "cited_other"]

# The no-planting control (W2). `planted` is the archived stage-04 condition; `control` is the
# same context with the appended span removed. Both are scored by one function in 05b.
CONDITIONS = ["planted", "control"]


def get(dataset):
    if dataset not in DATASETS:
        raise SystemExit(
            f"unknown dataset {dataset!r} -- known: {', '.join(sorted(DATASETS))}"
        )
    return DATASETS[dataset]


def runs_dir(dataset, *parts):
    """runs/<dataset>/... -- every output of every stage, never mixed across datasets."""
    get(dataset)
    return ROOT / "runs" / dataset / Path(*parts) if parts else ROOT / "runs" / dataset


def notebooks_dir(dataset, stage):
    """stage is "02_generation" or "04_postration"."""
    get(dataset)
    return ROOT / "pipeline" / "notebooks" / stage / dataset


def dataset_dir(dataset):
    """datasets/<dataset>/ -- the builder notebook and anything else dataset-specific."""
    get(dataset)
    return ROOT / "datasets" / dataset


def add_dataset_arg(parser, default="nq"):
    parser.add_argument(
        "--dataset",
        default=default,
        choices=sorted(DATASETS),
        help="which dataset's run to operate on (default: %(default)s)",
    )
    return parser
