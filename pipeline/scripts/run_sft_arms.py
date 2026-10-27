"""Drives the SFT reference arm(s) end to end, unattended, resumable.

    wave 1   02x generation, every new arm x 4 datasets          (8 GPU kernels)
    gate     03 --arms, attrition, the pre-specified go/no-go    (local)
    stage    adversarial + no-plant Kaggle Datasets, derive      (local + upload)
    wave 2   04x planted + 04x random control, surviving arms    (16 GPU kernels)
    score    05, 05b (both outcomes), result tables              (local)

Kaggle's free tier runs 2 GPU sessions at once, so both waves use a rolling window of 2: the
next kernel is pushed the moment one lands. Push, poll and pull reuse `run_pipeline.py` and
`run_noplant_control.py` -- the failure modes they already learned (hung CLI reads, push exit
0 on a printed error, 404 polled forever, transient resets) are not rediscovered here.

What is new is that nobody is watching:

  * Every step judges "done" from what is on disk (or on Kaggle), so a crash, a reboot or a
    restart picks up where it stopped. A kernel that finished while the driver was
    down is pulled, never re-pushed.
  * A quota rejection does not exit. Pushing pauses for an hour and is retried until the
    weekly reset; running kernels keep being polled and pulled meanwhile.
  * A failed kernel is retried once (Kaggle assigns hardware at random and a bad draw is
    the usual cause). A second failure stops only that job; the queue continues, and the
    run ends in NEEDS ATTENTION rather than stitching around a hole.
  * runs/logs/sft_queue_status.md is rewritten on every change; on macOS a desktop
    notification fires on completion or when a human is needed.

    python pipeline/scripts/run_sft_arms.py --dry-run
    python pipeline/scripts/run_sft_arms.py
    python pipeline/scripts/run_sft_arms.py --only-gen nq:search_sft  # one kernel
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load("run_pipeline")
nc = _load("run_noplant_control")
_spec = importlib.util.spec_from_file_location(
    "adv03", Path(__file__).resolve().parent.parent / "03_adversarial_construction.py")
adv03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adv03)

ROOT = config.ROOT
PY = sys.executable
NEW_ARMS = [a for a in config.SFT_ARMS if a not in config.DROPPED_ARMS]
DATASETS = ["nq", "qampari", "hagrid", "ragtruth"]
MAX_CONCURRENT = 2
POLL_SECONDS = 60
# RAGTruth planted runs ~3.5-4h; Kaggle's own session cap is 12h.
KERNEL_TIMEOUT_SECONDS = 10 * 60 * 60
QUOTA_BACKOFF_SECONDS = 60 * 60
MAX_ATTEMPTS = 2
# Pre-specified gate: drop an arm whose format-compliant answers
# are under 30% of questions on two or more datasets.
GATE_MIN_COMPLIANT = 0.30
GATE_MAX_FAILS = 1

STATE_PATH = ROOT / "runs" / "logs" / "sft_queue_state.json"
STATUS_MD = ROOT / "runs" / "logs" / "sft_queue_status.md"
ATTRITION_MD = ROOT / "results" / "attrition.md"
SCORING_DIR = ROOT / "results" / "scoring"

# arm key -> its rows in the existing per-stage tables, matched by position like everywhere else.
_POST = {arm: (nb, out) for nb, out, arm in rp.POSTRATION_ARMS}
_GEN = {rp.POSTRATION_ARMS[i][2]: g for i, g in enumerate(rp.GEN_ARMS)}
assert len(rp.GEN_ARMS) == len(rp.POSTRATION_ARMS) == len(config.ALL_ARMS)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def notify(msg):
    """Desktop notification. Best effort: a failure here must never stop the queue."""
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg}" with title "SFT queue"'],
                       timeout=10, capture_output=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------------
# State: everything that cannot be re-derived from disk
# ---------------------------------------------------------------------------------

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"attempts": {}, "gpu_seconds": {}, "gate": None, "staged": {}, "events": [],
            "phase": "start", "failed": []}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE_PATH)


def event(state, msg):
    log(msg)
    state["events"] = (state["events"] + [f"{datetime.now():%m-%d %H:%M} {msg}"])[-25:]
    save_state(state)


# ---------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------

def gen_job(ds, arm):
    nb_dir, out_dir, fname = _GEN[arm]
    return {"kind": "gen", "dataset": ds, "arm": arm, "stage": "02_generation",
            "nb_dir": nb_dir, "kid": rp.kernel_id(ds, nb_dir, "02_generation"),
            "path": ROOT / "pipeline" / "notebooks" / "02_generation" / ds / nb_dir,
            "pull_to": config.runs_dir(ds, "generation", out_dir),
            "dest": config.runs_dir(ds, "generation", out_dir, "generation_data", fname)}


def post_job(ds, arm, control):
    nb_dir, out_dir = _POST[arm]
    stage = "04_postration_noplant" if control else "04_postration"
    sub = "postration_noplant" if control else "postration"
    variant = "random" if control else "rel_uncited"
    return {"kind": "control" if control else "planted", "dataset": ds, "arm": arm,
            "stage": stage, "nb_dir": nb_dir, "kid": rp.kernel_id(ds, nb_dir, stage),
            "path": ROOT / "pipeline" / "notebooks" / stage / ds / nb_dir,
            "pull_to": config.runs_dir(ds, sub, out_dir),
            "dest": config.runs_dir(ds, sub, out_dir, f"{sub}_data",
                                    f"postration_{arm}_{variant}.json")}


def compliance(ds, arm):
    """Share of questions with a format-compliant answer -- the quantity the gate thresholds.
    Uses stage 03's own IDK and strict-format parsers, so it equals coverage_report's figure."""
    path = gen_job(ds, arm)["dest"]
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    ok = sum(1 for r in rows if not adv03.is_idk(r["generated_answer"])
             and adv03.parse_strict(r["generated_answer"], r["retrieved_chunks"]))
    return ok / len(rows)


def gate_failures(arm):
    return [ds for ds in DATASETS
            if (c := compliance(ds, arm)) is not None and c < GATE_MIN_COMPLIANT]


def gate_pending(job, running):
    """Hold a generation kernel whose arm is one failure from the gate while another of its
    kernels -- the one that decides it -- is still running."""
    return (job["kind"] == "gen"
            and len(gate_failures(job["arm"])) == GATE_MAX_FAILS
            and any(r["kind"] == "gen" and r["arm"] == job["arm"] for r in running))


def kaggle_status(kid):
    _, out = rp.run(["kaggle", "kernels", "status", kid], check=False, timeout=120)
    for word in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if word in out:
            return word
    # CLI 1.x said 404 / not found; CLI 2.x says the `kernels.get` permission was denied.
    if "404" in out or "not found" in out.lower() or "cannot access kernel" in out.lower():
        return "MISSING"
    return "UNKNOWN"


def pull(job):
    job["pull_to"].mkdir(parents=True, exist_ok=True)
    log(f"pull    {job['kid']} -> {job['pull_to'].relative_to(ROOT)}")
    rp.run(["kaggle", "kernels", "output", job["kid"], "-p", str(job["pull_to"])],
           timeout=1800)
    if job["dest"].exists():
        log(f"        {len(json.loads(job['dest'].read_text()))} records")
        return True
    log(f"        WARNING: expected {job['dest'].relative_to(ROOT)} not found")
    return False


def drive(jobs, state, label):
    """Rolling window of MAX_CONCURRENT. Returns the jobs that failed for good.

    The go/no-go gate is applied the moment its outcome is fixed: once an arm is below the
    threshold on more than GATE_MAX_FAILS datasets, no further outcome can rescue it, so its
    remaining generation kernels are skipped rather than run only to be discarded."""
    pending = [j for j in jobs if not j["dest"].exists()]
    running, failed = [], []
    hold_until = 0.0
    write_status(state, label, jobs, running)

    # A kernel that finished or is still going from a previous driver life is adopted, not
    # re-pushed. Fresh slugs (no SFT kernel ran under them before) make COMPLETE unambiguous.
    for job in list(pending):
        st = kaggle_status(job["kid"])
        if st == "COMPLETE":
            event(state, f"adopt   {job['kid']} (finished while driver was down)")
            if pull(job):
                pending.remove(job)
        elif st in ("RUNNING", "QUEUED"):
            event(state, f"adopt   {job['kid']} (still {st.lower()})")
            job["started"] = time.time()
            pending.remove(job)
            running.append(job)

    while pending or running:
        while pending and len(running) < MAX_CONCURRENT and time.time() >= hold_until:
            job = next((j for j in pending if not gate_pending(j, running)), None)
            if job is None:
                break  # every remaining job waits on a gate verdict still running
            if job["kind"] == "gen" and len(fails := gate_failures(job["arm"])) > GATE_MAX_FAILS:
                pending.remove(job)
                state.setdefault("skipped", []).append(job["kid"])
                event(state, f"skip    {job['kid']} -- {job['arm']} already fails the gate "
                             f"on {', '.join(fails)}")
                continue
            pending.remove(job)
            pending.insert(0, job)
            try:
                nc.push(job)
            except nc.QuotaExhausted as e:
                hold_until = time.time() + QUOTA_BACKOFF_SECONDS
                event(state, f"QUOTA   push paused until {datetime.fromtimestamp(hold_until):%H:%M}"
                             f" -- {str(e).splitlines()[0][:120]}")
                notify("Kaggle GPU quota reached; queue paused, retrying hourly")
                break
            except SystemExit as e:
                # A rejected push (bad metadata, etc.) is not retryable by waiting.
                pending.pop(0)
                failed.append(job)
                event(state, f"REJECT  {job['kid']} -- {str(e)[:200]}")
                continue
            pending.pop(0)
            state["attempts"][job["kid"]] = state["attempts"].get(job["kid"], 0) + 1
            event(state, f"push    {job['kid']} (attempt {state['attempts'][job['kid']]})")
            running.append(job)
            write_status(state, label, jobs, running, hold_until)

        time.sleep(POLL_SECONDS)
        for job in list(running):
            waited = time.time() - job["started"]
            st = kaggle_status(job["kid"])
            if st == "COMPLETE":
                running.remove(job)
                state["gpu_seconds"][job["kid"]] = waited
                state["crashes"] = 0  # progress: earlier crashes were transient
                event(state, f"done    {job['kid']} ({waited / 60:.0f} min)")
                if not pull(job):
                    failed.append(job)
            elif st in ("ERROR", "CANCEL") or waited > KERNEL_TIMEOUT_SECONDS or (
                    st == "MISSING" and waited > 5 * POLL_SECONDS):
                running.remove(job)
                why = st if st != "RUNNING" else "TIMEOUT"
                # Keep the failed run's log for diagnosis before any retry overwrites it.
                keep = ROOT / "runs" / "logs" / "sft_failed" / job["kid"].split("/")[-1]
                keep.mkdir(parents=True, exist_ok=True)
                rp.run(["kaggle", "kernels", "output", job["kid"], "-p", str(keep)],
                       check=False, timeout=900)
                if state["attempts"].get(job["kid"], 0) < MAX_ATTEMPTS:
                    event(state, f"{why:7} {job['kid']} -- retrying (log in "
                                 f"{keep.relative_to(ROOT)})")
                    pending.insert(0, job)
                else:
                    event(state, f"{why:7} {job['kid']} -- giving up after "
                                 f"{MAX_ATTEMPTS} attempts")
                    failed.append(job)
            elif int(waited) % 1800 < POLL_SECONDS:
                log(f"        {job['kid']} {st.lower()} ({waited / 60:.0f} min)")
        write_status(state, label, jobs, running, hold_until)
    return failed


# ---------------------------------------------------------------------------------
# Local steps
# ---------------------------------------------------------------------------------

def sh(args, timeout=3600):
    """Run a pipeline script; its stdout goes to the log. Any failure stops the driver."""
    _, out = rp.run([PY, *[str(a) for a in args]], timeout=timeout, retries=0)
    for line in out.splitlines():
        log(f"  | {line}")
    return out


def gate(state):
    """03 --arms on every dataset, then the pre-specified go/no-go per arm."""
    table = {}
    for ds in DATASETS:
        have = [a for a in NEW_ARMS if gen_job(ds, a)["dest"].exists()]
        if have:
            sh([ROOT / "pipeline" / "03_adversarial_construction.py", "--dataset", ds,
                "--arms", *have])
        table[ds] = json.loads(
            config.runs_dir(ds, "adversarial", "coverage_report.json").read_text())
    verdict = {}
    for arm in NEW_ARMS:
        fails = gate_failures(arm)
        complete = all(arm in table[ds] for ds in DATASETS)
        verdict[arm] = {"pass": complete and len(fails) <= GATE_MAX_FAILS,
                        "below_threshold": fails}
    write_attrition(table, verdict)
    return verdict


def write_attrition(table, verdict):
    lines = ["# Attrition and go/no-go gate (SFT arms)", "",
             f"Written by `run_sft_arms.py` on {datetime.now():%Y-%m-%d %H:%M}. Same layout "
             "as the paper's attrition table; existing arms included for comparison. Gate: an arm is "
             f"dropped if compliant answers are under {GATE_MIN_COMPLIANT:.0%} of questions on "
             f"{GATE_MAX_FAILS + 1}+ datasets.", "",
             "| dataset | arm | IDK | answered | compliant | compliant / questions | claims "
             "| random | rel_uncited | cited_other |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for arm in config.ARM_ORDER:
            c = table[ds].get(arm)
            if not c:
                continue
            tot = c["total_answers"]
            lines.append(
                f"| {config.get(ds)['label']} | {arm} | {c['idk']} ({c['idk'] / tot:.0%}) | "
                f"{c['answered']} | {c['compliant']} | {c['compliant'] / tot:.0%} | "
                f"{c['total_claims']} | {c.get('constructed_random', 0)} | "
                f"{c.get('constructed_rel_uncited', 0)} | {c.get('constructed_cited_other', 0)} |")
    lines += ["", "## Gate verdict", ""]
    for arm, v in verdict.items():
        lines.append(f"- `{arm}`: **{'PASS' if v['pass'] else 'DROPPED'}**"
                     + (f" (below threshold on {', '.join(v['below_threshold'])})"
                        if v["below_threshold"] else ""))
    ATTRITION_MD.parent.mkdir(parents=True, exist_ok=True)
    ATTRITION_MD.write_text("\n".join(lines) + "\n")


def upload(staging, ds_id, msg):
    """Create or version a Kaggle Dataset, then wait until every staged file is exposed."""
    expected = {f.name for f in staging.glob("*__*.json")}
    present = rp.dataset_files(ds_id)
    # The new arms' files are absent until their first upload, so a name check is enough to
    # make a restart after a completed upload skip it rather than cut another version.
    if present and expected <= present:
        log(f"        {ds_id} already exposes all {len(expected)} files")
        return
    if present:
        rp.run(["kaggle", "datasets", "version", "-p", str(staging), "-m", msg], timeout=1800)
    else:
        rp.run(["kaggle", "datasets", "create", "-p", str(staging)], timeout=1800)
    for _ in range(120):
        time.sleep(15)
        present = rp.dataset_files(ds_id)
        if present and expected <= present:
            log(f"        {ds_id} ready ({len(present)} files)")
            return
    raise SystemExit(f"{ds_id} never exposed all {len(expected)} files")


def stage_inputs(state, ds):
    """Build + upload what wave 2 mounts for one dataset. Marked done only at the very end."""
    if state["staged"].get(ds):
        return
    meta = config.get(ds)
    sh([ROOT / "pipeline" / "scripts" / "build_adversarial_dataset.py", "--dataset", ds])
    upload(config.runs_dir(ds, "kaggle_staging", "adversarial"), meta["adversarial_kaggle_id"],
           f"SFT arms {datetime.now():%Y-%m-%d %H:%M}")
    sh([ROOT / "pipeline" / "scripts" / "build_noplant_control.py", "--dataset", ds, "--stage"])
    sh([ROOT / "pipeline" / "tests" / "test_noplant_control.py", "--dataset", ds])
    upload(config.runs_dir(ds, "kaggle_staging", "adversarial_noplant"), meta["noplant_kaggle_id"],
           f"SFT arms {datetime.now():%Y-%m-%d %H:%M}")
    sh([ROOT / "pipeline" / "scripts" / "derive_noplant_from_clean.py", "--dataset", ds])
    state["staged"][ds] = True
    event(state, f"staged  {ds}: adversarial + no-plant datasets uploaded, controls derived")


def score(state):
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        out = sh([ROOT / "pipeline" / "05_score_postrationalization.py", "--dataset", ds])
        (SCORING_DIR / f"05_{ds}.txt").write_text(out + "\n")
    for outcome in ("cite_aligned", "cite_anywhere"):
        args = [ROOT / "pipeline" / "05b_score_noplant_control.py", "--all",
                "--outcome", outcome]
        if outcome == "cite_aligned":
            args.append("--dump")
        out = sh(args, timeout=7200)
        (SCORING_DIR / f"05b_{outcome}.txt").write_text(out + "\n")
    report = ROOT / "pipeline" / "scripts" / "report_sft_control.py"
    if report.exists():
        sh([report])
    event(state, "scored  05 + 05b (both outcomes) written to results/scoring/")


# ---------------------------------------------------------------------------------
# Status page
# ---------------------------------------------------------------------------------

def write_status(state, label, jobs, running, hold_until=0.0):
    run_ids = {j["kid"] for j in running}
    lines = [f"# SFT queue status", "",
             f"Updated {datetime.now():%Y-%m-%d %H:%M:%S} · phase **{label}** · "
             f"driver `pipeline/scripts/run_sft_arms.py` · log `runs/logs/sft_queue.log`", ""]
    if hold_until > time.time():
        lines += [f"**Paused on Kaggle GPU quota** until "
                  f"{datetime.fromtimestamp(hold_until):%Y-%m-%d %H:%M}, then retried hourly.", ""]
    gpu_h = sum(state["gpu_seconds"].values()) / 3600
    lines += [f"GPU time used by finished kernels (wall, incl. queue): {gpu_h:.1f} h", "",
              "| kernel | kind | state | attempts |", "|---|---|---|---|"]
    for j in jobs:
        st = ("done" if j["dest"].exists() else "running" if j["kid"] in run_ids
              else "FAILED" if j["kid"] in state["failed"] else "pending")
        lines.append(f"| `{j['kid'].split('/')[-1]}` | {j['kind']} | {st} | "
                     f"{state['attempts'].get(j['kid'], 0)} |")
    if state.get("gate"):
        lines += ["", "Gate: " + ", ".join(
            f"`{a}` {'PASS' if v['pass'] else 'DROPPED'}" for a, v in state["gate"].items())]
    lines += ["", "## Recent events", "", "```", *state["events"][-15:], "```"]
    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MD.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--only-gen", nargs="+", default=None, metavar="DS:ARM",
                    help="run just these generation kernels and stop (smoke test)")
    args = ap.parse_args()

    state = load_state()
    # Arm-major: the primary arm's kernels go first, so a failing secondary arm meets the gate
    # before it can take GPU time from the primary.
    gen_jobs = [gen_job(ds, arm) for arm in NEW_ARMS for ds in DATASETS]

    if args.dry_run:
        for j in gen_jobs:
            print(f"wave 1  {j['kid']:55} {'DONE' if j['dest'].exists() else 'todo'}")
        for ds in ["ragtruth", "nq", "hagrid", "qampari"]:
            for arm in NEW_ARMS:
                for ctl in (False, True):
                    j = post_job(ds, arm, ctl)
                    print(f"wave 2  {j['kid']:55} {'DONE' if j['dest'].exists() else 'todo'}")
        print(f"state: {STATE_PATH.relative_to(ROOT)} phase={state['phase']} "
              f"gate={state['gate']} staged={state['staged']}")
        return

    if args.only_gen:
        picked = [gen_job(*x.split(":")) for x in args.only_gen]
        failed = drive(picked, state, "smoke test")
        raise SystemExit(1 if failed else 0)

    log(f"=== SFT queue start: arms {NEW_ARMS}, phase {state['phase']} ===")

    # ---- wave 1 ------------------------------------------------------------------
    state["phase"] = "wave 1: generation"
    failed = drive(gen_jobs, state, state["phase"])
    if failed:
        state["failed"] = sorted({j["kid"] for j in failed})
        state["phase"] = "NEEDS ATTENTION: generation failed"
        event(state, f"stopping: {len(failed)} generation kernel(s) failed for good")
        write_status(state, state["phase"], gen_jobs, [])
        notify("SFT queue needs attention: generation kernels failed")
        raise SystemExit(0)

    # ---- gate ----------------------------------------------------------------------
    if state["gate"] is None:
        state["phase"] = "gate"
        state["gate"] = gate(state)
        event(state, "gate    " + ", ".join(
            f"{a} {'PASS' if v['pass'] else 'DROPPED ' + str(v['below_threshold'])}"
            for a, v in state["gate"].items()))
    arms = [a for a in NEW_ARMS if state["gate"][a]["pass"]]
    unrecorded = [a for a in NEW_ARMS if a not in arms and a in config.ARM_ORDER]
    if unrecorded:
        # The build scripts stage every arm in config.ARM_ORDER; a dropped arm with partial
        # data must be recorded in config.DROPPED_ARMS (a reviewed, committed decision) first.
        state["phase"] = f"NEEDS ATTENTION: add {unrecorded} to config.DROPPED_ARMS"
        event(state, f"stopping: gate dropped {unrecorded}; record it in config.DROPPED_ARMS "
                     f"and restart (see results/attrition.md)")
        write_status(state, state["phase"], gen_jobs, [])
        notify("SFT queue: an arm failed the gate -- record it in config.DROPPED_ARMS")
        raise SystemExit(0)
    if not arms:
        state["phase"] = "NEEDS ATTENTION: every SFT arm failed the gate"
        event(state, "stopping: no arm passed the pre-specified gate -- see results/attrition.md")
        write_status(state, state["phase"], gen_jobs, [])
        notify("SFT queue: no arm passed the gate")
        raise SystemExit(0)

    # ---- staging -------------------------------------------------------------------
    state["phase"] = "staging inputs"
    save_state(state)
    for ds in DATASETS:
        stage_inputs(state, ds)

    # ---- wave 2 --------------------------------------------------------------------
    # Longest first (RAGTruth planted ~3.5h) so the tail of the queue is short kernels.
    post_jobs = [post_job(ds, arm, ctl)
                 for ds in ["ragtruth", "nq", "hagrid", "qampari"]
                 for ctl in (False, True) for arm in arms]
    state["phase"] = "wave 2: planted + control"
    failed = drive(post_jobs, state, state["phase"])
    if failed:
        state["failed"] = sorted({j["kid"] for j in failed})
        state["phase"] = "NEEDS ATTENTION: wave 2 kernels failed"
        event(state, f"stopping before scoring: {len(failed)} kernel(s) failed -- "
                     f"re-run them whole, do not stitch")
        write_status(state, state["phase"], post_jobs, [])
        notify("SFT queue needs attention: wave 2 kernels failed")
        raise SystemExit(0)

    # ---- score ---------------------------------------------------------------------
    state["phase"] = "scoring"
    save_state(state)
    score(state)
    state["phase"] = "COMPLETE"
    event(state, "=== SFT queue complete ===")
    write_status(state, "COMPLETE", gen_jobs + post_jobs, [])
    notify("SFT queue complete -- results in results/")


def guarded_main():
    """A supervisor (e.g. launchd or systemd) restarts the driver on a non-zero exit (a crash)
    but not on exit 0 (finished, or stopped deliberately for a human). A bug that crashes on
    every start would otherwise restart forever, so the fifth consecutive crash is turned into a
    deliberate stop."""
    try:
        main()
    except SystemExit as e:
        if e.code in (0, None):
            raise
        crashed(e)
        raise
    except BaseException as e:
        crashed(e)
        raise
    state = load_state()
    if state.get("crashes"):
        state["crashes"] = 0
        save_state(state)


def crashed(e):
    state = load_state()
    state["crashes"] = state.get("crashes", 0) + 1
    event(state, f"CRASH   {type(e).__name__}: {str(e)[:200]} "
                 f"(consecutive crash {state['crashes']})")
    if state["crashes"] >= 5:
        state["phase"] = "NEEDS ATTENTION: driver keeps crashing"
        save_state(state)
        notify("SFT queue stopped: driver crashed 5 times in a row, see the log")
        raise SystemExit(0)

if __name__ == "__main__":
    guarded_main()
