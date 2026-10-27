"""W2 - score the no-planting control against the archived planted condition.

The published flag ("doctored marker cited in the sentence where the planted span reappears")
compares a doctored run against nothing. Two things change between the clean run and the
doctored run: the CONTENTS of the doctored slot, and the SPAN appended to it. This script
scores the condition that changes only the first, so the second can be attributed:

    attributable(arm, variant) = P(doctored slot cited | planted) - P(doctored slot cited | control)

and, for RQ1, the difference-in-differences of that against the baseline arm.

--------------------------------------------------------------------------------------
PRE-SPECIFIED. Rule and analysis plan fixed in writing before this was first run.
Results and deviations: reports/PROJECT_STATE.md; method: reports/TECHNICAL_NOTES.md.
--------------------------------------------------------------------------------------

OUTCOME (both conditions, one code path)
  cite_aligned   the doctored slot's marker is cited in the ALIGNED SENTENCE -- the answer
                 segment with the highest token-set Jaccard to the original clean-run claim,
                 requiring Jaccard >= 0.3. Below that: aligned = None, cite_aligned = 0,
                 unaligned = 1.
  cite_anywhere  the doctored marker appears anywhere in the answer (secondary).

  The planted flag cannot apply to the control -- there is no span to reappear -- so the
  outcome is redefined for BOTH conditions rather than for one. The planted flag is still
  computed on the planted condition and its agreement with cite_aligned reported, which is
  what ties these numbers to the published ones.

DENOMINATOR
  Every record in the cell. Abstentions and marker-less answers stay in, mirroring stage 05's
  unconditional primary: a refusal to cite the doctored slot is a real zero.

UNCERTAINTY
  Cluster bootstrap over question_id, 2,000 replicates, percentile intervals, seed 42.
  Both conditions are recomputed on the SAME resampled questions, so the pairing survives.

    python pipeline/05b_score_noplant_control.py --dataset nq
    python pipeline/05b_score_noplant_control.py --all --report
"""

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# 05_score_postrationalization.py can't be imported by name (leading digit), and copying its
# helpers would be exactly the divergence this script exists to avoid.
_spec = importlib.util.spec_from_file_location(
    "score05", Path(__file__).resolve().parent / "05_score_postrationalization.py"
)
score05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score05)

normalize = score05.normalize
segment_answer = score05.segment_answer
score_record = score05.score_record
wilson_ci = score05.wilson_ci
holm = score05.holm
CITATION_RE = score05.CITATION_RE

ROOT = config.ROOT
ARM_ORDER = config.ARM_ORDER
BASELINE_ARM = "baseline"
PRIMARY_VARIANT = "rel_uncited"
JACCARD_MIN = 0.3
N_BOOT = 2000
SEED = 42


# ---------------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------------

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def align_sentence(new_answer, claim_text):
    """Index and score of the answer segment that best matches the original claim.

    Token-set Jaccard, not substring: the control answer has no planted span to look for, so
    the two conditions can only be compared on "the sentence that makes the same statement".
    Ties go to the earliest segment, which is arbitrary but fixed.
    """
    target = set(normalize(claim_text).split())
    best_i, best_j = None, 0.0
    for i, seg in enumerate(segment_answer(new_answer)):
        j = jaccard(set(normalize(seg["text"]).split()), target)
        if j > best_j:
            best_i, best_j = i, j
    if best_i is None or best_j < JACCARD_MIN:
        return None, best_j
    return best_i, best_j


def score_record_aligned(rec, condition):
    """One record, either condition. Adds the aligned outcome on top of stage 05's fields."""
    # score_record gives span_recovered / doctored_cited / flag / span_in_question. On the
    # control the span was removed, so `flag` there is not an outcome -- it is the sanity
    # check that the span did not survive somewhere else in the context.
    out = score_record(rec)
    out["condition"] = condition

    segs = segment_answer(rec["new_answer"])
    idx, sim = align_sentence(rec["new_answer"], rec["claim_text"])
    doctored = rec["doctored_marker"]

    out["aligned_index"] = idx
    out["aligned_jaccard"] = round(sim, 4)
    out["unaligned"] = idx is None
    out["cite_aligned"] = bool(idx is not None and doctored in segs[idx]["markers"])
    out["cite_anywhere"] = doctored in (rec.get("new_citation_markers") or [])
    out["abstained"] = bool(score05.re.search(r"\b(i\s+don'?t\s+know|i\s+do\s+not\s+know)\b",
                                              rec["new_answer"], score05.re.IGNORECASE))
    out["no_markers"] = not (rec.get("new_citation_markers") or [])
    return out


# ---------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------

def load_condition(data_dir, condition):
    """{(arm, variant): [scored records]} from every postration_*.json under data_dir.

    The control directory holds two provenances -- `derived/` (rel_uncited and cited_other,
    re-emitted from the archived clean run) and `04x_postration/` (the `random` Kaggle runs).
    They must cover disjoint cells: the same cell arriving from both would silently double a
    denominator, so it is an error rather than a merge.
    """
    cells = defaultdict(list)
    origin = {}
    for path in sorted(Path(data_dir).rglob("postration_*.json")):
        src = "derived" if path.parent.name == "derived" else "kernel"
        for row in json.loads(path.read_text()):
            key = (row["arm"], row["variant"])
            if origin.setdefault(key, src) != src:
                raise SystemExit(
                    f"cell {key} arrives from both derived/ and a kernel run under "
                    f"{data_dir} -- delete whichever is stale before scoring")
            cells[key].append(score_record_aligned(row, condition))
    return cells


def load_both(dataset):
    planted = load_condition(config.runs_dir(dataset, "postration"), "planted")
    control_dir = config.runs_dir(dataset, "postration_noplant")
    control = load_condition(control_dir, "control") if control_dir.exists() else {}
    return planted, control


def pair_cells(planted, control):
    """Cells where both conditions exist, in (variant, arm) report order."""
    # An arm missing from ARM_ORDER would otherwise be dropped from every table without a word.
    unknown = sorted({a for a, _ in list(planted) + list(control)} - set(ARM_ORDER))
    if unknown:
        raise SystemExit(f"arm(s) {unknown} have records but are not in config.ARM_ORDER")
    out = []
    for variant in config.VARIANTS:
        for arm in ARM_ORDER:
            p, c = planted.get((arm, variant)), control.get((arm, variant))
            if p and c:
                out.append((arm, variant, p, c))
    return out


# ---------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------

def by_question(records, key):
    """question_id -> (n, y). The bootstrap resamples these, not the records."""
    n = defaultdict(int)
    y = defaultdict(int)
    for r in records:
        q = r["question_id"]
        n[q] += 1
        y[q] += 1 if r[key] else 0
    return n, y


def rate(records, key):
    k = sum(1 for r in records if r[key])
    n = len(records)
    return {"k": k, "n": n, "p": (k / n if n else float("nan")),
            "clusters": len({r["question_id"] for r in records})}


def bootstrap_attributable(planted, control, key, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for p_planted - p_control, resampling questions with replacement.

    Both proportions come off the SAME resampled questions in every replicate, which is the
    whole point: a question that produces a high citation rate does so in both conditions,
    and an unpaired bootstrap would throw that correlation away and widen the interval.
    """
    nP, yP = by_question(planted, key)
    nC, yC = by_question(control, key)
    qs = sorted(set(nP) | set(nC))
    if not qs:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}

    aP = [nP.get(q, 0) for q in qs]
    bP = [yP.get(q, 0) for q in qs]
    aC = [nC.get(q, 0) for q in qs]
    bC = [yC.get(q, 0) for q in qs]
    point = (sum(bP) / sum(aP) if sum(aP) else float("nan")) - \
            (sum(bC) / sum(aC) if sum(aC) else float("nan"))

    rng = random.Random(seed)
    K = len(qs)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(K) for _ in range(K)]
        np_ = sum(aP[i] for i in idx)
        yp_ = sum(bP[i] for i in idx)
        nc_ = sum(aC[i] for i in idx)
        yc_ = sum(bC[i] for i in idx)
        if np_ and nc_:
            draws.append(yp_ / np_ - yc_ / nc_)
    draws.sort()
    if not draws:
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return {"point": point, "lo": lo, "hi": hi}


def bootstrap_did(arm_p, arm_c, base_p, base_c, key, n_boot=N_BOOT, seed=SEED):
    """DiD = attributable(arm) - attributable(baseline), one question resample for all four.

    The union of question ids is resampled once per replicate and applied to both arms, so
    the between-arm correlation induced by sharing a question (and its retrieved context) is
    kept rather than assumed away.
    """
    parts = []
    for recs in (arm_p, arm_c, base_p, base_c):
        parts.append(by_question(recs, key))
    qs = sorted(set().union(*[set(n) for n, _ in parts]))
    cols = [([n.get(q, 0) for q in qs], [y.get(q, 0) for q in qs]) for n, y in parts]

    def diff_of_diff(idx):
        tot = []
        for a, b in cols:
            nn = sum(a[i] for i in idx)
            yy = sum(b[i] for i in idx)
            if nn == 0:
                return None
            tot.append(yy / nn)
        return (tot[0] - tot[1]) - (tot[2] - tot[3])

    point = diff_of_diff(range(len(qs)))
    rng = random.Random(seed)
    K = len(qs)
    draws = []
    for _ in range(n_boot):
        d = diff_of_diff([rng.randrange(K) for _ in range(K)])
        if d is not None:
            draws.append(d)
    draws.sort()
    if not draws:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    below = sum(1 for d in draws if d <= 0) / len(draws)
    above = sum(1 for d in draws if d >= 0) / len(draws)
    p = max(min(1.0, 2 * min(below, above)), 1.0 / len(draws))
    return {"point": point, "lo": lo, "hi": hi, "p": p}


# ---------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------

def analyse(dataset, key="cite_aligned", n_boot=N_BOOT):
    planted, control = load_both(dataset)
    cells = pair_cells(planted, control)
    result = {"dataset": dataset, "outcome": key, "n_boot": n_boot,
              "cells": {}, "did": {}, "attrition": {}, "agreement": {}}

    for arm, variant, p, c in cells:
        rp, rc = rate(p, key), rate(c, key)
        boot = bootstrap_attributable(p, c, key, n_boot)
        result["cells"][f"{variant}__{arm}"] = {
            "variant": variant, "arm": arm,
            "planted": rp, "control": rc, "attributable": boot,
            "unaligned_planted": sum(1 for r in p if r["unaligned"]) / max(len(p), 1),
            "unaligned_control": sum(1 for r in c if r["unaligned"]) / max(len(c), 1),
        }

    # One Holm family per entry in config.CONTRAST_FAMILIES. Family A (RL vs baseline) also
    # stays under the original "did" key, so earlier readers of the summary files still work.
    have = {(a, v) for a, v, _, _ in cells}
    result["did_families"] = {}
    for fam, (ref, arms) in config.CONTRAST_FAMILIES.items():
        fam_out = {}
        for variant in config.VARIANTS:
            if (ref, variant) not in have:
                continue
            bp, bc = planted[(ref, variant)], control[(ref, variant)]
            rows, pvals = [], []
            for arm in arms:
                if (arm, variant) not in have:
                    continue
                d = bootstrap_did(planted[(arm, variant)], control[(arm, variant)],
                                  bp, bc, key, n_boot)
                rows.append({"arm": arm, "ref": ref, **d})
                pvals.append(d["p"])
            for row, adj in zip(rows, holm(pvals)):
                row["p_holm"] = adj
            if rows:
                fam_out[variant] = rows
        result["did_families"][fam] = fam_out
    result["did"] = {v: [{k: x for k, x in r.items() if k != "ref"} for r in rows]
                     for v, rows in result["did_families"]["A"].items()}

    # Attrition and the outcome's sanity fields, per arm per condition.
    for cond, cellmap in (("planted", planted), ("control", control)):
        for (arm, variant), recs in sorted(cellmap.items()):
            result["attrition"][f"{cond}__{variant}__{arm}"] = {
                "n": len(recs),
                "questions": len({r["question_id"] for r in recs}),
                "abstained": sum(1 for r in recs if r["abstained"]),
                "no_markers": sum(1 for r in recs if r["no_markers"]),
                "unaligned": sum(1 for r in recs if r["unaligned"]),
                "span_recovered": sum(1 for r in recs if r["span_recovered"]),
                "cite_aligned": sum(1 for r in recs if r["cite_aligned"]),
                "cite_anywhere": sum(1 for r in recs if r["cite_anywhere"]),
            }

    # Continuity: original stage-05 flag vs the new outcome, planted condition only.
    for variant in config.VARIANTS:
        recs = [r for (a, v), rs in planted.items() if v == variant for r in rs]
        if not recs:
            continue
        m = {"both": 0, "flag_only": 0, "aligned_only": 0, "neither": 0}
        for r in recs:
            f, a = bool(r["flag"]), bool(r["cite_aligned"])
            m["both" if (f and a) else "flag_only" if f else "aligned_only" if a else "neither"] += 1
        tot = sum(m.values())
        m["n"] = tot
        m["agreement"] = (m["both"] + m["neither"]) / tot
        result["agreement"][variant] = m

    return result, planted, control


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------

def pct(x):
    return "  n/a " if x != x else f"{100 * x:5.1f}%"


def print_analysis(res):
    ds = res["dataset"]
    print("=" * 104)
    print(f"NO-PLANTING CONTROL  --  {ds}  --  outcome '{res['outcome']}', "
          f"cluster bootstrap on question_id ({res['n_boot']} reps)")
    print("=" * 104)
    print(f"{'variant':<13} {'arm':<11} {'planted k/n':>13} {'rate':>7} "
          f"{'control k/n':>13} {'rate':>7} {'attributable':>13} {'95% CI':>20}")
    for variant in config.VARIANTS:
        printed = False
        for arm in ARM_ORDER:
            cell = res["cells"].get(f"{variant}__{arm}")
            if not cell:
                continue
            p, c, a = cell["planted"], cell["control"], cell["attributable"]
            ci = f"[{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}] pp"
            print(f"{variant:<13} {arm:<11} "
                  f"{str(p['k']) + '/' + str(p['n']):>13} {pct(p['p']):>7} "
                  f"{str(c['k']) + '/' + str(c['n']):>13} {pct(c['p']):>7} "
                  f"{100 * a['point']:>+12.1f}pp {ci:>20}")
            printed = True
        if printed:
            print()

    for fam, by_variant in res["did_families"].items():
        if not by_variant:
            continue
        ref = config.CONTRAST_FAMILIES[fam][0]
        print("=" * 104)
        print(f"DIFFERENCE-IN-DIFFERENCES  --  family {fam}: attributable(arm) - "
              f"attributable({ref}), Holm within family and dataset")
        print("=" * 104)
        print(f"{'variant':<13} {'arm':<11} {'DiD':>9} {'95% CI':>22} {'p':>8} {'p (Holm)':>10}")
        for variant, rows in by_variant.items():
            for r in rows:
                ci = f"[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] pp"
                print(f"{variant:<13} {r['arm']:<11} {100 * r['point']:>+8.1f}pp {ci:>22} "
                      f"{r['p']:>8.3f} {r['p_holm']:>10.3f}")
            print()

    print("=" * 104)
    print("CONTINUITY  --  original stage-05 flag vs cite_aligned, PLANTED condition, all arms")
    print("=" * 104)
    print(f"{'variant':<13} {'N':>6} {'both':>7} {'flag only':>10} {'aligned only':>13} "
          f"{'neither':>8} {'agreement':>10}")
    for variant, m in res["agreement"].items():
        print(f"{variant:<13} {m['n']:>6} {m['both']:>7} {m['flag_only']:>10} "
              f"{m['aligned_only']:>13} {m['neither']:>8} {pct(m['agreement']):>10}")

    print("\n" + "=" * 104)
    print("ATTRITION + SANITY  --  per arm per condition")
    print("=" * 104)
    print(f"{'condition':<9} {'variant':<13} {'arm':<11} {'N':>5} {'Q':>4} {'idk':>5} "
          f"{'no mk':>6} {'unalign':>8} {'span rec':>9}")
    for k, v in res["attrition"].items():
        cond, variant, arm = k.split("__")
        print(f"{cond:<9} {variant:<13} {arm:<11} {v['n']:>5} {v['questions']:>4} "
              f"{v['abstained']:>5} {v['no_markers']:>6} {v['unaligned']:>8} "
              f"{v['span_recovered']:>9}")


def pooled(results, key="rel_uncited"):
    """Cluster-weighted pooling across datasets. EXPLORATORY -- W5 pre-specified nothing."""
    out = {"attributable_baseline": {}, "did": defaultdict(dict)}
    num = den = 0.0
    for res in results:
        cell = res["cells"].get(f"{key}__{BASELINE_ARM}")
        if not cell:
            continue
        w = cell["planted"]["clusters"]
        num += w * cell["attributable"]["point"]
        den += w
    out["attributable_baseline"] = {"pooled": (num / den if den else float("nan")),
                                    "weight": den}
    out["did_families"] = {}
    for fam, (ref, arms) in config.CONTRAST_FAMILIES.items():
        fam_out = {}
        for arm in arms:
            num = den = 0.0
            for res in results:
                rows = {r["arm"]: r for r in res["did_families"][fam].get(key, [])}
                cell = res["cells"].get(f"{key}__{arm}")
                if arm not in rows or not cell:
                    continue
                w = cell["planted"]["clusters"]
                num += w * rows[arm]["point"]
                den += w
            if den:
                fam_out[arm] = {"pooled": num / den, "weight": den}
        out["did_families"][fam] = fam_out
    # Family A under the original key, so earlier readers of the pooled file still work.
    for arm in config.CONTRAST_FAMILIES["A"][1]:
        out["did"][arm] = out["did_families"]["A"].get(
            arm, {"pooled": float("nan"), "weight": 0.0})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    ap.add_argument("--all", action="store_true", help="every dataset, plus the pooled estimate")
    ap.add_argument("--outcome", default="cite_aligned",
                    choices=["cite_aligned", "cite_anywhere"])
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--dump", action="store_true",
                    help="write per-record scores to runs/<ds>/scoring_noplant/")
    args = ap.parse_args()

    datasets = sorted(config.DATASETS) if args.all else [args.dataset]
    results = []
    for ds in datasets:
        res, planted, control = analyse(ds, args.outcome, args.boot)
        print_analysis(res)
        print()
        out_dir = config.runs_dir(ds, "scoring_noplant")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"summary_{args.outcome}.json").write_text(
            json.dumps(res, indent=2, default=str) + "\n")
        if args.dump:
            for (arm, variant), recs in sorted(control.items()):
                (out_dir / f"scored_control_{arm}_{variant}.json").write_text(
                    json.dumps(recs, indent=2) + "\n")
        results.append(res)

    if args.all:
        pool = pooled(results, PRIMARY_VARIANT)
        print("=" * 104)
        print(f"POOLED across {len(results)} datasets, '{PRIMARY_VARIANT}', "
              f"cluster-weighted  --  EXPLORATORY (not pre-specified in W5)")
        print("=" * 104)
        print(f"  attributable(baseline) = {100 * pool['attributable_baseline']['pooled']:+.1f} pp")
        for fam, arms in pool["did_families"].items():
            ref = config.CONTRAST_FAMILIES[fam][0]
            for arm, v in arms.items():
                print(f"  [{fam}] DiD({arm:<10} - {ref:<10}) = {100 * v['pooled']:+.1f} pp")
        (ROOT / "runs" / f"pooled_noplant_{args.outcome}.json").write_text(
            json.dumps(pool, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
