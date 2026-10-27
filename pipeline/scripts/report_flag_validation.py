"""Flag validation - tabulate the two-rater review of flagged records (paper, Appendix H).

Reads, from data/validation/: the sample (`extract_flag_sample.py`), the LLM rater's labels
(`llm_labels.tsv`), the human rater's labels (`rater_labels.csv`, exported from
`review_rater.html`) and, if present, an optional third set (`human_labels.csv`, exported from
`review.html`). Writes results/validation/tables.md. No number in the paper's validation
section is typed by hand.

Labels:
  prior_support     did the doctored passage support the claim BEFORE the span was planted?
  aligned_support   ... the new aligned sentence (differs only where the model rewrote the claim)
  original_support  did the passage the model originally cited support the claim?
  content_change    SAME / REWORDED / CHANGED, new aligned sentence vs the claim
each support label is FULL / PARTIAL / NONE.

Intervals: cluster bootstrap over questions (2,000 reps, seed 42), as in 05b; the same question
can appear under several arms. Population-weighted shares reweight each (dataset, arm) cell by
its number of flagged records, since the sample takes 5 per cell regardless of cell size.

    python pipeline/scripts/report_flag_validation.py
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
VAL = ROOT / "data" / "validation"
SAMPLE = VAL / "flag_sample.json"
LLM = VAL / "llm_labels.tsv"
HUMAN = VAL / "human_labels.csv"  # optional third rater, from review.html
RATER = VAL / "rater_labels.csv"  # independent human rater, from review_rater.html
OUT = ROOT / "results" / "validation" / "tables.md"
LEVELS = ["FULL", "PARTIAL", "NONE"]
CHANGES = ["SAME", "REWORDED", "CHANGED"]
N_BOOT, SEED = 2000, 42
DATASET_ORDER = ["nq", "qampari", "hagrid", "ragtruth"]


def load_labels(path, delimiter):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter=delimiter) if r.get("prior_support")]
    return {r["item_id"]: r for r in rows}


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "—"


def boot_share(items, labels, field, level):
    """95% cluster-bootstrap interval for the share of `level`, resampling questions."""
    clusters = defaultdict(list)
    for it in items:
        clusters[(it["dataset"], it["question_id"])].append(
            labels[it["item_id"]][field] == level)
    keys = list(clusters)
    rng = random.Random(SEED)
    shares = []
    for _ in range(N_BOOT):
        hits = [x for k in rng.choices(keys, k=len(keys)) for x in clusters[k]]
        shares.append(sum(hits) / len(hits))
    shares.sort()
    return 100 * shares[int(0.025 * N_BOOT)], 100 * shares[int(0.975 * N_BOOT) - 1]


def weighted_share(items, labels, field, level, population):
    num = den = 0.0
    for it in items:
        w = population[f"{it['dataset']}/{it['arm']}"]
        num += w * (labels[it["item_id"]][field] == level)
        den += w
    return 100 * num / den


def dist_table(title, groups, labels, field, levels):
    lines = [f"### {title}", "", "| group | n | " + " | ".join(levels) + " |",
             "|---|---:|" + "---:|" * len(levels)]
    for name, its in groups:
        c = Counter(labels[it["item_id"]][field] for it in its)
        lines.append(f"| {name} | {len(its)} | "
                     + " | ".join(pct(c[l], len(its)) for l in levels) + " |")
    return lines + [""]


def kappa(a, b, levels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in levels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(items, llm, human, field, levels, names=("LLM", "human")):
    ids = [it["item_id"] for it in items
           if it["item_id"] in human and it["item_id"] in llm and human[it["item_id"]].get(field)]
    if not ids:
        return []
    a = [llm[i][field] for i in ids]
    b = [human[i][field] for i in ids]
    lines = [f"**{field}**: {len(ids)} items, raw agreement "
             f"{pct(sum(x == y for x, y in zip(a, b)), len(ids))}, Cohen's κ = "
             f"{kappa(a, b, levels):.2f}", "",
             f"| {names[0]} ↓ / {names[1]} → | " + " | ".join(levels) + " |",
             "|---|" + "---:|" * len(levels)]
    for la in levels:
        lines.append(f"| {la} | " + " | ".join(
            str(sum(1 for x, y in zip(a, b) if x == la and y == lb)) for lb in levels) + " |")
    disagree = [i for i, x, y in zip(ids, a, b) if x != y]
    if disagree:
        lines += ["", "Disagreements: " + ", ".join(disagree)]
    return lines + [""]


def main():
    sample = json.loads(SAMPLE.read_text())
    items, population = sample["items"], sample["flagged_population"]
    llm = load_labels(LLM, "\t")
    missing = [it["item_id"] for it in items if it["item_id"] not in llm]
    if missing:
        raise SystemExit(f"LLM labels missing for {len(missing)} items: {missing[:5]}")

    lines = ["# Flag validation: tables", "",
             f"Rendered {date.today()} by `pipeline/scripts/report_flag_validation.py` from "
             "`flag_sample.json` and `llm_labels.tsv`"
             + (" and `human_labels.csv`" if HUMAN.exists() else "")
             + (" and `rater_labels.csv`" if RATER.exists() else "")
             + ". Interpretation: the paper, Appendix H.", "",
             f"Sample: {len(items)} flagged `rel_uncited` records (claim-aligned, planted), "
             f"5 per (dataset, arm) cell, drawn from {sum(population.values())} flagged records.",
             ""]

    fields = [("prior_support", LEVELS), ("aligned_support", LEVELS),
              ("original_support", LEVELS), ("content_change", CHANGES)]
    raters = [(name, load_labels(path, ","))
              for name, path in [("author", HUMAN), ("rater", RATER)] if path.exists()]
    by_ds = [(config.DATASETS[d]["label"], [it for it in items if it["dataset"] == d])
             for d in DATASET_ORDER]
    by_arm = [(a, [it for it in items if it["arm"] == a]) for a in config.ARM_ORDER]

    # Sections 1-2 for every complete set of labels: the LLM's, then each human rater's.
    for who, labels in [("LLM", llm)] + raters:
        its = [it for it in items if labels.get(it["item_id"], {}).get("prior_support")]
        if len(its) < len(items):
            continue
        lines += [f"## 1. Did the doctored passage support the claim before planting? ({who})", "",
                  "| prior_support | count | 95% CI (cluster bootstrap) | population-weighted |",
                  "|---|---:|---|---:|"]
        for lvl in LEVELS:
            k = sum(labels[it["item_id"]]["prior_support"] == lvl for it in items)
            lo, hi = boot_share(items, labels, "prior_support", lvl)
            w = weighted_share(items, labels, "prior_support", lvl, population)
            lines.append(f"| {lvl} | {pct(k, len(items))} | [{lo:.0f}, {hi:.0f}]% | {w:.0f}% |")
        lines.append("")
        lines += dist_table(f"By dataset ({who})", by_ds, labels, "prior_support", LEVELS)
        lines += dist_table(f"By arm ({who})", by_arm, labels, "prior_support", LEVELS)
        lines += [f"## 2. Switch type: prior support of the new source × support of the original "
                  f"source ({who})", "",
                  "| doctored (before plant) ↓ / original → | " + " | ".join(LEVELS) + " |",
                  "|---|" + "---:|" * len(LEVELS)]
        for p_ in LEVELS:
            lines.append(f"| {p_} | " + " | ".join(str(sum(
                1 for it in items if labels[it["item_id"]]["prior_support"] == p_
                and labels[it["item_id"]]["original_support"] == o)) for o in LEVELS) + " |")
        lines.append("")

    lines += ["## 3. Content of the new aligned sentence (LLM)", "",
              "| content_change | count | aligned_support FULL | PARTIAL | NONE |",
              "|---|---:|---:|---:|---:|"]
    for ch in CHANGES:
        its = [it for it in items if llm[it["item_id"]]["content_change"] == ch]
        c = Counter(llm[it["item_id"]]["aligned_support"] for it in its)
        lines.append(f"| {ch} | {len(its)} | " + " | ".join(str(c[l]) for l in LEVELS) + " |")
    k = sum(llm[it["item_id"]]["aligned_support"] == "NONE" for it in items)
    lines += ["", f"Judged on the new aligned sentence instead of the original claim, the "
              f"doctored passage gives no support in {pct(k, len(items))}.", ""]

    lines += ["## 4. Cross-check: lexical overlap by label (LLM)", "",
              "Share of the claim's content words found in the passage (mean). Not used to "
              "label; a sanity check that the labels track something measurable.", "",
              "| prior_support | n | doctored passage, before plant | original passage |",
              "|---|---:|---:|---:|"]
    for lvl in LEVELS:
        its = [it for it in items if llm[it["item_id"]]["prior_support"] == lvl]
        if its:
            d = sum(it["lex_support_doctored_pre_plant"] for it in its) / len(its)
            o = sum(it["lex_support_original"] for it in its) / len(its)
            lines.append(f"| {lvl} | {len(its)} | {d:.2f} | {o:.2f} |")
    lines.append("")

    lines += ["## 5. Agreement between raters", ""]
    if not raters:
        lines += ["Pending: no `human_labels.csv` or `rater_labels.csv` yet.", ""]
    for name, labels in raters:
        lines += [f"### LLM vs {name}", ""]
        for field, lv in fields:
            lines += agreement(items, llm, labels, field, lv, ("LLM", name))
        done = [it for it in items if labels.get(it["item_id"], {}).get("prior_support")]
        if done:
            lines += [f"{name.capitalize()} labels, prior_support: " + ", ".join(
                f"{lvl} {pct(sum(labels[it['item_id']]['prior_support'] == lvl for it in done), len(done))}"
                for lvl in LEVELS), ""]
    if len(raters) == 2:
        lines += ["### Author vs rater (items both labelled)", ""]
        for field, lv in fields:
            lines += agreement(items, raters[0][1], raters[1][1], field, lv, ("author", "rater"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
