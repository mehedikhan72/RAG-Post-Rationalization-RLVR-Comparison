"""Renders results/sft_control_tables.md from the 05b summaries: every table that compares the
models (RL agents, Search-SFT, baseline). No number in the paper's result tables is typed by
hand. report_noplant_control.py renders the older per-dataset control reports.

    python pipeline/05b_score_noplant_control.py --all --dump
    python pipeline/05b_score_noplant_control.py --all --outcome cite_anywhere
    python pipeline/scripts/report_sft_control.py

Families are config.CONTRAST_FAMILIES: A = RL vs baseline (primary), B = SFT vs baseline,
C = RL vs search_sft (exploratory).
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
DATASETS = ["nq", "qampari", "hagrid", "ragtruth"]
FAMILY_LABEL = {"A": "RL vs baseline (primary)",
                "B": "SFT vs baseline (confirmatory)",
                "C": "RL vs SFT (exploratory)"}


def load(dataset, outcome):
    p = config.runs_dir(dataset, "scoring_noplant", f"summary_{outcome}.json")
    if not p.exists():
        raise SystemExit(f"missing {p} - run 05b first")
    return json.loads(p.read_text())


def pp(x):
    return "n/a" if x != x else f"{100 * x:+.1f}"


def rate_table(res_by_ds, variant):
    arms = config.ARM_ORDER
    lines = ["| dataset | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for ds in DATASETS:
        res = res_by_ds[ds]
        cells = []
        for arm in arms:
            c = res["cells"].get(f"{variant}__{arm}")
            if not c:
                cells.append("—")
                continue
            p, a = c["planted"], c["attributable"]
            cells.append(f"{pp(a['point'])} <sub>{p['k']}/{p['n']}</sub>")
        lines.append(f"| {config.get(ds)['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def did_table(res_by_ds, fam, variant):
    ref = config.CONTRAST_FAMILIES[fam][0]
    lines = [f"| dataset | arm − {ref} | DiD (pp) | 95% CI | boot p | Holm p |",
             "|---|---|---:|---|---:|---:|"]
    any_row = False
    for ds in DATASETS:
        rows = res_by_ds[ds].get("did_families", {}).get(fam, {}).get(variant, [])
        for r in rows:
            any_row = True
            sig = "**" if r["p_holm"] < 0.05 else ""
            lines.append(f"| {config.get(ds)['label']} | {r['arm']} | {sig}{pp(r['point'])}{sig} | "
                         f"[{pp(r['lo'])}, {pp(r['hi'])}] | {r['p']:.3f} | {sig}{r['p_holm']:.3f}{sig} |")
    return "\n".join(lines) if any_row else "*no data yet*"


def pooled_lines(outcome):
    p = ROOT / "runs" / f"pooled_noplant_{outcome}.json"
    if not p.exists():
        return ["*no pooled file*"]
    pool = json.loads(p.read_text())
    out = [f"- attributable(baseline) = {pp(pool['attributable_baseline']['pooled'])} pp"]
    for fam, arms in pool.get("did_families", {}).items():
        ref = config.CONTRAST_FAMILIES[fam][0]
        for arm, v in arms.items():
            out.append(f"- [{fam}] DiD({arm} − {ref}) = {pp(v['pooled'])} pp")
    return out


def claim_share_lines(variant="rel_uncited", n_boot=2000, seed=42):
    """Planted claims flagged, pooled over all four datasets: the share of all probed claims.

    The pooled lines below average the per-dataset rates weighted by question count, which is
    not this share (RAGTruth has many claims per question at a low rate). CI: questions
    resampled within each dataset, as in 05b.
    """
    import importlib.util
    import random
    spec = importlib.util.spec_from_file_location(
        "score05b", ROOT / "pipeline" / "05b_score_noplant_control.py")
    score05b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score05b)
    out = []
    for arm in config.ARM_ORDER:
        by_ds = []
        for ds in DATASETS:
            p = config.runs_dir(ds, "scoring", f"scored_{arm}_{variant}.json")
            by_q = {}
            for rec in json.loads(p.read_text()):
                hit = score05b.score_record_aligned(rec, "planted")["cite_aligned"]
                by_q.setdefault(rec["question_id"], []).append(hit)
            by_ds.append(list(by_q.values()))
        k = sum(sum(q) for qs in by_ds for q in qs)
        n = sum(len(q) for qs in by_ds for q in qs)
        rng, est = random.Random(seed), []
        for _ in range(n_boot):
            kk = nn = 0
            for qs in by_ds:
                for q in rng.choices(qs, k=len(qs)):
                    kk += sum(q)
                    nn += len(q)
            est.append(kk / nn)
        est.sort()
        out.append(f"- {arm}: {k}/{n} = {100 * k / n:.1f}% "
                   f"[{100 * est[int(0.025 * n_boot)]:.1f}, {100 * est[int(0.975 * n_boot) - 1]:.1f}]")
    return out


def main():
    aligned = {ds: load(ds, "cite_aligned") for ds in DATASETS}
    anywhere = {ds: load(ds, "cite_anywhere") for ds in DATASETS}
    L = ["# SFT control: result tables", "",
         f"Rendered {date.today()} by `pipeline/scripts/report_sft_control.py` from "
         "`runs/<ds>/scoring_noplant/summary_*.json`. No number is typed by hand. Bold marks "
         "Holm p < 0.05.", "",
         "## 1. Attributable rate, `rel_uncited`, claim-aligned (pp; planted k/n below)", "",
         "The control is 0/n by construction on this variant, so these equal the raw rates.", "",
         rate_table(aligned, "rel_uncited"), ""]
    for fam in ("B", "C", "A"):
        L += [f"## 2{fam}. Family {fam}: {FAMILY_LABEL[fam]} — `rel_uncited`, claim-aligned", "",
              did_table(aligned, fam, "rel_uncited"), ""]
    L += ["## 3. Robustness: `random` variant, claim-aligned", "",
          rate_table(aligned, "random"), ""]
    for fam in ("B", "C"):
        L += [f"### Family {fam} on `random`", "", did_table(aligned, fam, "random"), ""]
    L += ["## 4. Robustness: `rel_uncited`, cite-anywhere outcome", ""]
    for fam in ("B", "C"):
        L += [f"### Family {fam}", "", did_table(anywhere, fam, "rel_uncited"), ""]
    L += ["## 5a. Share of all probed claims flagged, four datasets pooled (`rel_uncited`, "
          "claim-aligned, planted; control is 0)", "",
          "k/n over all planted records, with a 95% CI that resamples questions within each "
          "dataset (2,000 reps, seed 42).", "", *claim_share_lines(), ""]
    L += ["## 5. Across datasets (exploratory): per-dataset estimates averaged, weighted by number of questions", "",
          "Claim-aligned:", "", *pooled_lines("cite_aligned"), "",
          "Cite-anywhere:", "", *pooled_lines("cite_anywhere"), ""]
    out = ROOT / "results" / "sft_control_tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
