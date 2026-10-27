"""Drives the whole RQ1 pipeline for one dataset, end to end.

    02x generation (4 GPU kernels)
      -> 03 adversarial construction (local, CPU)
      -> adversarial Kaggle Dataset
      -> 04x re-inference (4 GPU kernels)
      -> 05 scoring (local, CPU)

GPU kernels are pushed **two at a time**: Kaggle's free tier caps concurrent batch GPU
sessions at 2 and a third push fails outright rather than queueing.

Every stage is idempotent and the script is resumable: a stage whose output already exists
is skipped, so a crash halfway through costs only the stage it died in. `--force` re-runs
anyway.

    python pipeline/scripts/run_pipeline.py --dataset hagrid
    python pipeline/scripts/run_pipeline.py --dataset hagrid --from-stage adversarial

Progress goes to stdout with a `[dataset]` prefix; run it under nohup and tail the log.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.ROOT
POLL_SECONDS = 60
KERNEL_TIMEOUT_SECONDS = 6 * 60 * 60

GEN_ARMS = [
    ("02a_generation_baseline", "02a_gen", "generations_nonrl.json"),
    ("02b_generation_searchr1", "02b_gen", "generations_search_r1.json"),
    ("02c_generation_research", "02c_gen", "generations_research.json"),
    ("02d_generation_rsearch", "02d_gen", "generations_rsearch.json"),
    ("02e_generation_searchsft", "02e_gen", "generations_search_sft.json"),
]
POSTRATION_ARMS = [
    ("04a_postration_baseline", "04a_postration", "baseline"),
    ("04b_postration_searchr1", "04b_postration", "search_r1"),
    ("04c_postration_research", "04c_postration", "research"),
    ("04d_postration_rsearch", "04d_postration", "rsearch"),
    ("04e_postration_searchsft", "04e_postration", "search_sft"),
]

STAGES = ["generation", "adversarial", "postration", "scoring"]


def log(dataset, msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] [{dataset}] {msg}", flush=True)


def run(cmd, check=True, timeout=600, retries=2):
    """Run a CLI command, dropping the urllib3 LibreSSL warning Kaggle prints every call.

    `timeout` is not optional here. A `kaggle kernels status` call hung on a network read
    with no timeout and silently froze the whole driver for over two hours while both
    kernels it was waiting on had already finished. subprocess.run blocks forever by
    default, so a single stuck HTTP read is indistinguishable from a long-running job.
    Transient network failures are retried rather than killing a multi-hour run.
    """
    last = ""
    for attempt in range(retries + 1):
        # start_new_session puts the child in its own process group so a timeout can
        # kill the WHOLE group. subprocess.run(timeout=...) only kills the direct child;
        # `kaggle kernels output` survived that and kept running past its 30-minute
        # timeout with 0 bytes written, stalling the driver until killed by hand.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            last = f"timed out after {timeout}s"
            print(f"    (retry {attempt + 1}/{retries}: {' '.join(cmd[:3])} {last})",
                  flush=True)
            continue

        class _Result:
            pass
        p = _Result()
        p.returncode, p.stdout, p.stderr = proc.returncode, stdout, stderr
        out = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                        if "NotOpenSSLWarning" not in l and "warnings.warn" not in l)
        if p.returncode == 0 or not check:
            return p.returncode, out
        last = out
        print(f"    (retry {attempt + 1}/{retries}: {' '.join(cmd[:3])} rc={p.returncode})",
              flush=True)
    if check:
        raise SystemExit(f"command failed after {retries + 1} tries: "
                         f"{' '.join(cmd)}\n{last}")
    return 1, last


def kernel_id(dataset, nb_dir, stage):
    """Read the slug the generator wrote, rather than re-deriving it here.

    Two places computing the same slug is how a push silently lands on the wrong kernel.
    """
    meta = (ROOT / "pipeline" / "notebooks" / stage / dataset / nb_dir /
            "kernel-metadata.json")
    return json.loads(meta.read_text())["id"]


def wait_for_kernels(dataset, kernel_ids):
    start = time.time()
    deadline = start + KERNEL_TIMEOUT_SECONDS
    pending = list(kernel_ids)
    failed = []
    while pending:
        if time.time() > deadline:
            raise SystemExit(f"timed out waiting for {pending}")
        time.sleep(POLL_SECONDS)
        waited = int(time.time() - start)
        if waited % 900 < POLL_SECONDS:
            log(dataset, f"  ... still waiting ({waited // 60} min) on {len(pending)}")
        for kid in list(pending):
            _, out = run(["kaggle", "kernels", "status", kid], check=False, timeout=120)
            if "COMPLETE" in out:
                log(dataset, f"  done    {kid}")
                pending.remove(kid)
            elif "ERROR" in out or "CANCEL" in out:
                log(dataset, f"  FAILED  {kid} -- {out.strip()}")
                failed.append(kid)
                pending.remove(kid)
    if failed:
        raise SystemExit(f"{len(failed)} kernel(s) failed: {failed}. "
                         f"Check the Kaggle logs before re-running.")


def dataset_files(ds_id):
    """Filenames a Kaggle Dataset currently exposes, or an empty set if it has none.

    `kaggle datasets files` prints a table whose first column is the filename; unlike
    `datasets status` it still prints when stdout is a pipe.
    """
    rc, out = run(["kaggle", "datasets", "files", ds_id], check=False, timeout=120)
    if rc != 0:
        return set()
    names = set()
    for line in out.splitlines():
        first = line.split()[:1]
        if first and first[0].endswith(".json"):
            names.add(first[0])
    return names


def kernel_is_busy(kid):
    """True if this kernel already has a run queued or in flight.

    Pushing over a running kernel starts a second version competing for the same GPU
    quota, and pushing over a finished one silently redoes work that is already on disk.
    Both are avoidable by asking first.
    """
    _, out = run(["kaggle", "kernels", "status", kid], check=False, timeout=120)
    return "RUNNING" in out or "QUEUED" in out


def push_in_pairs(dataset, entries, stage, skip_done=()):
    """entries: list of (notebook_dir, ...). Pushes 2, waits, pushes the next 2."""
    entries = [e for e in entries if e[0] not in skip_done]
    if not entries:
        log(dataset, "  all arms already have output, nothing to push")
        return
    for i in range(0, len(entries), 2):
        batch = entries[i:i + 2]
        ids = []
        for nb_dir, *_ in batch:
            path = ROOT / "pipeline" / "notebooks" / stage / dataset / nb_dir
            kid = kernel_id(dataset, nb_dir, stage)
            if kernel_is_busy(kid):
                log(dataset, f"  attach  {kid} (already running, not re-pushing)")
            else:
                log(dataset, f"  push    {kid}")
                run(["kaggle", "kernels", "push", "-p", str(path)])
            ids.append(kid)
        log(dataset, f"  waiting on {len(ids)} GPU kernel(s)...")
        wait_for_kernels(dataset, ids)


def pull_outputs(dataset, entries, stage, out_subdir):
    for nb_dir, out_dir, *_ in entries:
        kid = kernel_id(dataset, nb_dir, stage)
        dest = config.runs_dir(dataset, out_subdir, out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        log(dataset, f"  pull    {kid} -> {dest.relative_to(ROOT)}")
        run(["kaggle", "kernels", "output", kid, "-p", str(dest)], timeout=1800)


def stage_generation(dataset, force):
    produced = [config.runs_dir(dataset, "generation", d, "generation_data", f)
                for _, d, f in GEN_ARMS]
    if not force and all(p.exists() for p in produced):
        log(dataset, "generation: already present, skipping")
        return
    done = () if force else tuple(
        nb for nb, d, f in GEN_ARMS
        if config.runs_dir(dataset, "generation", d, "generation_data", f).exists())
    if done:
        log(dataset, f"generation: {len(done)} arm(s) already done, skipping those")
    log(dataset, "generation: pushing remaining GPU kernels, 2 at a time")
    push_in_pairs(dataset, GEN_ARMS, "02_generation", skip_done=done)
    pull_outputs(dataset, GEN_ARMS, "02_generation", "generation")
    missing = [p for p in produced if not p.exists()]
    if missing:
        raise SystemExit(f"generation output missing: {missing}")
    for _, d, f in GEN_ARMS:
        rows = json.loads(
            config.runs_dir(dataset, "generation", d, "generation_data", f).read_text())
        log(dataset, f"  {f:28} {len(rows)} records")


def stage_adversarial(dataset, force):
    out = config.runs_dir(dataset, "adversarial", "coverage_report.json")
    if not force and out.exists():
        log(dataset, "adversarial: already present, skipping")
    else:
        log(dataset, "adversarial: building doctored contexts (local)")
        _, o = run([sys.executable, str(ROOT / "pipeline" / "03_adversarial_construction.py"),
                    "--dataset", dataset])
        for line in o.splitlines():
            log(dataset, f"  {line}")

    log(dataset, "adversarial: staging + uploading Kaggle Dataset")
    _, o = run([sys.executable,
                str(ROOT / "pipeline" / "scripts" / "build_adversarial_dataset.py"),
                "--dataset", dataset])
    staging = config.runs_dir(dataset, "kaggle_staging", "adversarial")
    ds_id = config.get(dataset)["adversarial_kaggle_id"]
    expected = sorted(f.name for f in staging.glob("*__*.json"))

    if dataset_files(ds_id):
        run(["kaggle", "datasets", "version", "-p", str(staging), "-m",
             f"rebuild {datetime.now():%Y-%m-%d %H:%M}"], timeout=1800)
    else:
        run(["kaggle", "datasets", "create", "-p", str(staging)], timeout=1800)

    # 04x cannot mount a dataset that is still being processed. Readiness is judged by
    # the FILES the dataset actually exposes, not by `kaggle datasets status`: that
    # command prints nothing at all when its output is captured rather than attached to a
    # terminal, so a "ready" substring check silently never matches and the wait always
    # times out. Verified empirically -- it cost one run of this stage.
    for _ in range(80):
        time.sleep(15)
        present = dataset_files(ds_id)
        if present and set(expected) <= present:
            log(dataset, f"  dataset ready: {ds_id} ({len(present)} files)")
            return
    raise SystemExit(
        f"{ds_id} never exposed all {len(expected)} expected files; "
        f"last saw {sorted(dataset_files(ds_id))}"
    )


def stage_postration(dataset, force):
    produced = [config.runs_dir(dataset, "postration", d, "postration_data",
                                f"postration_{arm}_rel_uncited.json")
                for _, d, arm in POSTRATION_ARMS]
    if not force and all(p.exists() for p in produced):
        log(dataset, "postration: already present, skipping")
        return
    done = () if force else tuple(
        nb for nb, d, arm in POSTRATION_ARMS
        if config.runs_dir(dataset, "postration", d, "postration_data",
                           f"postration_{arm}_rel_uncited.json").exists())
    if done:
        log(dataset, f"postration: {len(done)} arm(s) already done, skipping those")
    log(dataset, "postration: pushing remaining GPU kernels, 2 at a time")
    push_in_pairs(dataset, POSTRATION_ARMS, "04_postration", skip_done=done)
    pull_outputs(dataset, POSTRATION_ARMS, "04_postration", "postration")
    missing = [p for p in produced if not p.exists()]
    if missing:
        raise SystemExit(f"postration output missing: {missing}")


def stage_scoring(dataset, force):
    log(dataset, "scoring")
    _, o = run([sys.executable,
                str(ROOT / "pipeline" / "05_score_postrationalization.py"),
                "--dataset", dataset])
    print(o, flush=True)
    summary = config.runs_dir(dataset, "scoring", "summary.json")
    log(dataset, f"scoring written to {summary.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_arg(ap)
    ap.add_argument("--from-stage", choices=STAGES, default="generation")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages even when their output already exists")
    args = ap.parse_args()

    fns = {"generation": stage_generation, "adversarial": stage_adversarial,
           "postration": stage_postration, "scoring": stage_scoring}
    todo = STAGES[STAGES.index(args.from_stage):]

    t0 = time.time()
    log(args.dataset, f"=== RQ1 pipeline start: {' -> '.join(todo)} ===")
    for name in todo:
        fns[name](args.dataset, args.force)
    log(args.dataset, f"=== PIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
