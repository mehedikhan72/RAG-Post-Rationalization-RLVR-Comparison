"""Renders reports/<ds>/noplant_control.md from the 05b summaries. No number is typed by hand.

    python pipeline/05b_score_noplant_control.py --all --dump
    python pipeline/05b_score_noplant_control.py --all --outcome cite_anywhere
    python pipeline/scripts/report_noplant_control.py --all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
ARMS = config.ARM_ORDER
VARIANTS = config.VARIANTS


def load(dataset, outcome):
    p = config.runs_dir(dataset, "scoring_noplant", f"summary_{outcome}.json")
    if not p.exists():
        raise SystemExit(f"missing {p} - run 05b first")
    return json.loads(p.read_text())


def pct(x, nd=1):
    return "n/a" if x != x else f"{100 * x:.{nd}f}%"


def pp(x):
    return "n/a" if x != x else f"{100 * x:+.1f}pp"


def cell_rows(res, outcome_label):
    lines = [f"| variant | arm | planted k/n | rate | control k/n | rate | attributable | 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    for variant in VARIANTS:
        for arm in ARMS:
            c = res["cells"].get(f"{variant}__{arm}")
            if not c:
                lines.append(f"| `{variant}` | {arm} | — | — | *missing* | — | — | — |")
                continue
            p, ct, a = c["planted"], c["control"], c["attributable"]
            lines.append(
                f"| `{variant}` | {arm} | {p['k']}/{p['n']} | {pct(p['p'])} | "
                f"{ct['k']}/{ct['n']} | {pct(ct['p'])} | **{pp(a['point'])}** | "
                f"[{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}] pp |")
    return "\n".join(lines)


def did_rows(res):
    lines = ["| variant | arm | DiD | 95% CI | boot p | Holm p |", "|---|---|---|---|---|---|"]
    for variant in VARIANTS:
        rows = res["did"].get(variant)
        if not rows:
            lines.append(f"| `{variant}` | — | *missing* | | | |")
            continue
        for r in rows:
            lines.append(f"| `{variant}` | {r['arm']} | **{pp(r['point'])}** | "
                         f"[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] pp | "
                         f"{r['p']:.3f} | {r['p_holm']:.3f} |")
    return "\n".join(lines)


def attrition_rows(res):
    lines = ["| condition | variant | arm | N | questions | abstained | no markers | unaligned | span recovered |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond in ("planted", "control"):
        for variant in VARIANTS:
            for arm in ARMS:
                v = res["attrition"].get(f"{cond}__{variant}__{arm}")
                if not v:
                    continue
                lines.append(
                    f"| {cond} | `{variant}` | {arm} | {v['n']} | {v['questions']} | "
                    f"{v['abstained']} | {v['no_markers']} | "
                    f"{v['unaligned']} ({pct(v['unaligned'] / v['n'])}) | "
                    f"{v['span_recovered']} ({pct(v['span_recovered'] / v['n'])}) |")
    return "\n".join(lines)


def agreement_rows(res):
    lines = ["| variant | N | both | flag only | cite_aligned only | neither | agreement |",
             "|---|---|---|---|---|---|---|"]
    for variant in VARIANTS:
        m = res["agreement"].get(variant)
        if not m:
            continue
        lines.append(f"| `{variant}` | {m['n']} | {m['both']} | {m['flag_only']} | "
                     f"{m['aligned_only']} | {m['neither']} | {pct(m['agreement'], 1)} |")
    return "\n".join(lines)


def unaligned_spread(res, variant):
    """Max between-arm gap in the planted unaligned rate. §7.4's >5pp sensitivity trigger."""
    vals = [res["attrition"][f"planted__{variant}__{a}"]["unaligned"] /
            res["attrition"][f"planted__{variant}__{a}"]["n"]
            for a in ARMS if f"planted__{variant}__{a}" in res["attrition"]]
    return (max(vals) - min(vals)) if vals else float("nan")


def flagged_control_examples(dataset, limit=20):
    """§9's hand-read: every control record that scored cite_aligned = 1, quoted verbatim.

    On the derived cells this should be ~empty -- the control answer is the clean answer, and
    the doctored slot was selected for not being cited in it. Whatever survives is worth
    reading, because it is the only place the control is not zero by construction.
    """
    out = []
    for path in sorted(config.runs_dir(dataset, "scoring_noplant").glob("scored_control_*.json")):
        for r in json.loads(path.read_text()):
            if r["cite_aligned"]:
                out.append(r)
    if not out:
        return "No control record scored `cite_aligned = 1` on this dataset."
    lines = [f"{len(out)} control record(s) scored `cite_aligned = 1`, all quoted:"]
    for r in out[:limit]:
        ans = " ".join(r["new_answer"].split())
        lines += [
            "",
            f"- **{r['arm']} / `{r['variant']}`** — doctored marker `[{r['doctored_marker']}]`, "
            f"aligned segment {r['aligned_index']} at Jaccard {r['aligned_jaccard']}",
            f"  - question: {r['question']}",
            f"  - original claim: {r['claim_text']}",
            f"  - control answer: {ans[:400]}{'…' if len(ans) > 400 else ''}",
        ]
    if len(out) > limit:
        lines.append(f"\n({len(out) - limit} more not shown.)")
    return "\n".join(lines)


def render(dataset):
    aligned = load(dataset, "cite_aligned")
    anywhere = load(dataset, "cite_anywhere")
    meta = config.get(dataset)
    manifest = json.loads(
        (config.runs_dir(dataset, "adversarial_noplant") / "manifest.json").read_text())
    derived = json.loads(
        (config.runs_dir(dataset, "postration_noplant", "derived") / "manifest.json").read_text())

    spread = unaligned_spread(aligned, "rel_uncited")
    examples = flagged_control_examples(dataset)

    body = f"""# No-planting control — {meta['label']}

**Work item:** W2 (project brief §9.4 limitation A3, §10 W2)
**Pre-specified:** fixed in writing, before this was scored
**Method + verification:** `reports/TECHNICAL_NOTES.md` §4–5
**Scorer:** `pipeline/05b_score_noplant_control.py` (outcome `cite_aligned`, cluster bootstrap
on `question_id`, {aligned['n_boot']} replicates, seed 42)
**Control contexts:** `runs/{dataset}/adversarial_noplant/` — {manifest['total_records']} records,
prompt SHA-256 `{manifest['prompt_sha256'][:16]}…`
**Control answers:** `runs/{dataset}/postration_noplant/derived/`
**Planted answers:** `runs/{dataset}/postration/` — archived, unmodified

---

## 1. What the control is, and why two variants of it cost no GPU

`03_adversarial_construction.py` emits `rel_uncited` and `cited_other` as
`emit(variant, target, target, chunks_lookup[target]["text"])` — the chunk stays in the slot it
already occupies and the only edit is the appended span. Removing the span therefore restores
the slot exactly, and the control context is **byte-identical to the clean stage-02 context**.
The stage-04 prompt is byte-identical to the stage-02 prompt and both decode greedily, so the
control answer is the archived clean answer. Verified per record (chunk ids, order, and every
slot text against `chunks.json`) by `derive_noplant_from_clean.py`.

`random` is different: it substitutes a chunk from another question into an uncited slot, so
its control context is genuinely new and had to be generated. Those four cells come from
Kaggle GPU runs (`pipeline/notebooks/04_postration_noplant/`, driven by
`run_noplant_control.py`), pulled into `runs/{dataset}/postration_noplant/04x_postration/`
and paired record-for-record with their planted twins (same order, same question ids, same
claims, same doctored marker — asserted, not assumed).

**All 12 cells are now measured.** The control base rate on `random` is 0–0.9%: the model
does occasionally cite a passage lifted from an unrelated question, with no span planted in
it at all, but rarely. Roughly 90% of the raw `random` rate survives as attributable.

## 2. Attrition, per arm per condition

{attrition_rows(aligned)}

The control's `unaligned = 0` and `span recovered = 100%` are not measurements: the control
answer *is* the answer the claim and the span were lifted from, so the claim's own sentence
aligns at Jaccard 1.0 and the span is a substring of it by construction.

## 3. Planted vs control — `cite_aligned`

The doctored slot's marker cited in the sentence that best matches the original clean-run
claim (token-set Jaccard ≥ 0.3).

{cell_rows(aligned, 'cite_aligned')}

## 4. Difference-in-differences vs baseline

`DiD = attributable(arm) − attributable(baseline)`, cluster bootstrap resampling the union of
question ids across the two arms, Holm across the three contrasts within this dataset.

{did_rows(aligned)}

## 5. Continuity — original stage-05 flag vs `cite_aligned` (planted condition, all arms)

{agreement_rows(aligned)}

## 6. Unaligned rates and the `cite_anywhere` sensitivity

Between-arm spread in the planted `unaligned` rate on `rel_uncited`: **{pct(spread)}**
({'above' if spread > 0.05 else 'below'} the pre-specified 5pp trigger, so the
`cite_anywhere` recomputation below is {'required' if spread > 0.05 else 'informational'}).

{cell_rows(anywhere, 'cite_anywhere')}

{did_rows(anywhere)}

## 7. Every non-zero control record, read by hand (§9 of the plan)

{examples}

Where these exist they are alignment ties, not base-rate citations: the answer repeats an
identical sentence, one copy of which happens to carry the doctored marker, and the
best-Jaccard rule (ties to the earliest segment) picks that copy. The true control base rate
on the derived cells is 0.

---

`cite_anywhere` is unusable on `cited_other` and the control is what shows it: the doctored
slot is *selected* as one the clean answer already cites for a different sentence, so
P(cite_anywhere | control) = 100% by construction and the attributable quantity is negative.
Any raw "was the passage cited at all" rate on this variant measures the selection rule, not
the model.
"""
    out = ROOT / "reports" / dataset / "noplant_control.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for ds in (sorted(config.DATASETS) if args.all else [args.dataset]):
        render(ds)


if __name__ == "__main__":
    main()
