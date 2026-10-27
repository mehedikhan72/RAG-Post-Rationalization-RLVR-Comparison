"""Flag validation - build the review sheets for the flag-validation sample.

Reads data/validation/ and writes three files into results/validation/:

  review.html        self-contained, offline. One item per screen, blind by default (no arm,
                     dataset or LLM label until you choose to reveal it after labelling).
                     Labels are kept in the browser and exported as `human_labels.csv`, which
                     `report_flag_validation.py` reads to compute agreement.
  review_rater.html  the same sheet for an independent second rater: no LLM labels, arm or
                     dataset in the file at all, separate browser storage, exports
                     `rater_labels.csv`.
  review_sheet.csv   the same items flat, for a spreadsheet, with empty label columns.

    python pipeline/scripts/build_review_sheet.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
VAL = ROOT / "data" / "validation"
OUT_DIR = ROOT / "results" / "validation"
LABEL_COLS = ["prior_support", "aligned_support", "original_support", "content_change", "note"]


def load_llm():
    with open(VAL / "llm_labels.tsv", newline="") as f:
        return {r["item_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def write_csv(items):
    cols = ["item_id", "question", "clean_answer", "claim", "original_marker",
            "original_passage", "doctored_marker", "doctored_passage_pre_plant",
            "planted_span", "new_answer", "aligned_sentence"] + LABEL_COLS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "review_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow({**it, **{c: "" for c in LABEL_COLS}})


def write_html(path, payload):
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = (Path(__file__).with_name("review_sheet_template.html").read_text()
            .replace("/*__DATA__*/[]", data))
    path.write_text(html)


def main():
    sample = json.loads((VAL / "flag_sample.json").read_text())
    llm = load_llm()
    items = sample["items"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_html(OUT_DIR / "review.html", [{**it, "llm": llm[it["item_id"]]} for it in items])
    # Independent second rater: no LLM labels, arm or dataset anywhere in the file.
    hidden = {"arm", "dataset", "question_id", "lex_support_original",
              "lex_support_doctored_pre_plant"}
    write_html(OUT_DIR / "review_rater.html",
               [{k: v for k, v in it.items() if k not in hidden} for it in items])
    write_csv(items)
    print(f"{len(items)} items -> results/validation/: review.html, review_rater.html, "
          "review_sheet.csv")


if __name__ == "__main__":
    main()
