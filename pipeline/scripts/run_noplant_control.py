"""Drives the 16 W2 no-planting control GPU runs on Kaggle, 2 in flight at a time.

    upload the 4 `-noplant` Kaggle Datasets
      -> push 04x control kernels, keeping exactly 2 running
      -> pull each kernel's output the moment it finishes
      -> 05b scoring picks the results up from runs/<ds>/postration_noplant/

Only the `random` variant is generated in these notebooks (see
`generate_postration_notebooks.py --variants`): `rel_uncited` and `cited_other` plant into the
slot their chunk already occupies, so their control IS the archived clean run and is derived
locally by `derive_noplant_from_clean.py`. All three variants would be ~35 h of GPU, over
Kaggle's 30 h/week quota; `random` alone is ~10 h.

Unlike `run_pipeline.py`'s batch-of-2, this keeps a **rolling** window: as soon as one kernel
finishes the next is pushed, rather than waiting for both. Dataset order is RAGTruth first
(largest cell), then NQ, HAGRID, QAMPARI, per the W2 plan.

Resumable: a kernel whose output is already on disk is skipped, and a kernel already RUNNING
or QUEUED on Kaggle is attached to rather than re-pushed.

    python pipeline/scripts/run_noplant_control.py
    python pipeline/scripts/run_noplant_control.py --datasets ragtruth nq
"""

import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# run_pipeline.py already solved kernel push/poll/pull, including the two failure modes that
# cost real runs (hung `kernels status` reads, and `datasets status` printing nothing when
# piped). Reimplementing them here would mean rediscovering them.
_spec = importlib.util.spec_from_file_location(
    "run_pipeline", Path(__file__).resolve().parent / "run_pipeline.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

ROOT = config.ROOT
STAGE = "04_postration_noplant"
VARIANT = "random"
MAX_CONCURRENT = 2
POLL_SECONDS = 60
KERNEL_TIMEOUT_SECONDS = 6 * 60 * 60

# W2 plan §5: RAGTruth first (largest cell), then NQ, HAGRID, QAMPARI.
DATASET_ORDER = ["ragtruth", "nq", "hagrid", "qampari"]
ARMS = [
    ("04a_postration_baseline", "04a_postration", "baseline"),
    ("04b_postration_searchr1", "04b_postration", "search_r1"),
    ("04c_postration_research", "04c_postration", "research"),
    ("04d_postration_rsearch", "04d_postration", "rsearch"),
]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def out_path(dataset, out_dir, arm):
    return config.runs_dir(dataset, "postration_noplant", out_dir, "postration_noplant_data",
                           f"postration_{arm}_{VARIANT}.json")


def upload_datasets(datasets):
    """Create or version each `-noplant` Kaggle Dataset and wait until its files are exposed."""
    for ds in datasets:
        staging = config.runs_dir(ds, "kaggle_staging", "adversarial_noplant")
        if not staging.exists():
            raise SystemExit(f"missing {staging} - run build_noplant_control.py --stage first")
        ds_id = config.get(ds)["noplant_kaggle_id"]
        expected = sorted(f.name for f in staging.glob("*__*.json"))

        present = rp.dataset_files(ds_id)
        if present and set(expected) <= present:
            log(f"{ds}: dataset {ds_id} already complete ({len(present)} files)")
            continue
        if present:
            log(f"{ds}: versioning {ds_id}")
            rp.run(["kaggle", "datasets", "version", "-p", str(staging), "-m",
                    f"noplant control {datetime.now():%Y-%m-%d %H:%M}"], timeout=1800)
        else:
            log(f"{ds}: creating {ds_id}")
            rp.run(["kaggle", "datasets", "create", "-p", str(staging)], timeout=1800)

        # Readiness is judged by the files the dataset exposes; `datasets status` prints
        # nothing when piped (run_pipeline.py learned this the expensive way).
        for _ in range(80):
            time.sleep(15)
            present = rp.dataset_files(ds_id)
            if present and set(expected) <= present:
                log(f"{ds}: dataset ready ({len(present)} files)")
                break
        else:
            raise SystemExit(f"{ds_id} never exposed all {len(expected)} files")


def build_queue(datasets, force):
    jobs = []
    for ds in datasets:
        for nb_dir, out_dir, arm in ARMS:
            dest = out_path(ds, out_dir, arm)
            if dest.exists() and not force:
                log(f"{ds}/{arm}: output already on disk, skipping")
                continue
            jobs.append({
                "dataset": ds, "nb_dir": nb_dir, "out_dir": out_dir, "arm": arm,
                "kid": rp.kernel_id(ds, nb_dir, STAGE),
                "path": ROOT / "pipeline" / "notebooks" / STAGE / ds / nb_dir,
                "dest": dest,
            })
    return jobs


# Network-layer failures that say nothing about whether Kaggle accepted the push.
TRANSIENT_RE = re.compile(
    r"ConnectionResetError|Connection aborted|Connection reset|Max retries|"
    r"Read timed out|Temporary failure|502|503|504",
    re.IGNORECASE)


class QuotaExhausted(SystemExit):
    pass


def push(job):
    """Push one kernel, treating a printed error as a failure.

    `kaggle kernels push` EXITS 0 while printing "Kernel push error: ..." -- including for
    "Maximum weekly GPU quota of 30.00 hours reached". A plain returncode check therefore
    reports success for a kernel that was never created, and the driver then polls a slug
    that 404s until it times out. Cost of learning this: three hours of no-op polling.
    """
    if rp.kernel_is_busy(job["kid"]):
        log(f"attach  {job['kid']} (already running)")
        job["started"] = time.time()
        return
    log(f"push    {job['kid']}")
    _, out = rp.run(["kaggle", "kernels", "push", "-p", str(job["path"])], check=False)
    low = out.lower()
    if "quota" in low:
        raise QuotaExhausted(
            f"Kaggle GPU quota exhausted -- {out.strip()}\n"
            f"Completed runs are on disk. Re-run this script after the weekly reset; "
            f"it skips whatever already has output.")

    # A bare `"error" in out` is too broad: it matches transient network failures like
    # ConnectionResetError, which killed the driver mid-queue on 2026-09-06. Only Kaggle's
    # own rejection message is fatal; network blips are retried, then verified by asking
    # Kaggle whether the kernel actually exists -- the push often lands even when the
    # client reports a failure.
    if "kernel push error" in low:
        raise SystemExit(f"push rejected for {job['kid']}: {out.strip()}")
    if TRANSIENT_RE.search(out):
        log(f"        transient push error, verifying: {out.strip()[:120]}")
        for attempt in range(3):
            time.sleep(20)
            _, st = rp.run(["kaggle", "kernels", "status", job["kid"]],
                           check=False, timeout=120)
            if "KernelWorkerStatus" in st:
                log(f"        push landed after all ({st.strip()[-40:]})")
                break
            log(f"        not present yet (try {attempt + 1}/3), re-pushing")
            rp.run(["kaggle", "kernels", "push", "-p", str(job["path"])], check=False)
        else:
            raise SystemExit(f"push for {job['kid']} never took: {out.strip()}")
    job["started"] = time.time()


def pull(job):
    dest = config.runs_dir(job["dataset"], "postration_noplant", job["out_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    log(f"pull    {job['kid']} -> {dest.relative_to(ROOT)}")
    rp.run(["kaggle", "kernels", "output", job["kid"], "-p", str(dest)], timeout=1800)
    if job["dest"].exists():
        rows = json.loads(job["dest"].read_text())
        log(f"        {len(rows)} records")
    else:
        log(f"        WARNING: expected {job['dest'].relative_to(ROOT)} not found")


def drive(jobs, force):
    """Rolling window: keep MAX_CONCURRENT kernels in flight, start the next as one lands."""
    pending, running, failed, done = list(jobs), [], [], []
    while pending or running:
        while pending and len(running) < MAX_CONCURRENT:
            job = pending.pop(0)
            push(job)
            running.append(job)
            log(f"        in flight: {len(running)}, queued: {len(pending)}")

        time.sleep(POLL_SECONDS)
        for job in list(running):
            waited = time.time() - job["started"]
            _, out = rp.run(["kaggle", "kernels", "status", job["kid"]],
                            check=False, timeout=120)
            if "COMPLETE" in out:
                log(f"done    {job['kid']} ({waited / 60:.0f} min)")
                running.remove(job)
                pull(job)
                done.append(job)
            elif "ERROR" in out or "CANCEL" in out:
                log(f"FAILED  {job['kid']} -- {out.strip()}")
                running.remove(job)
                failed.append(job)
            elif waited > KERNEL_TIMEOUT_SECONDS:
                log(f"TIMEOUT {job['kid']} after {waited / 3600:.1f}h -- abandoning")
                running.remove(job)
                failed.append(job)
            elif "404" in out or "not found" in out.lower():
                # The kernel does not exist -- almost always a push that "succeeded" with
                # rc=0 while printing an error. Never treat this as "still running".
                job["missing"] = job.get("missing", 0) + 1
                if job["missing"] >= 3:
                    log(f"MISSING {job['kid']} -- kernel does not exist on Kaggle, "
                        f"the push did not take")
                    running.remove(job)
                    failed.append(job)
            elif int(waited) % 900 < POLL_SECONDS:
                log(f"        {job['kid']} still running ({waited / 60:.0f} min)")
    return done, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASET_ORDER,
                    choices=sorted(config.DATASETS))
    ap.add_argument("--force", action="store_true",
                    help="re-run kernels whose output is already on disk")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    datasets = [d for d in DATASET_ORDER if d in args.datasets]
    t0 = time.time()
    log(f"=== W2 no-planting control: {len(datasets)} dataset(s), variant '{VARIANT}' ===")

    if not args.skip_upload:
        upload_datasets(datasets)

    jobs = build_queue(datasets, args.force)
    if not jobs:
        log("nothing to run")
        return
    log(f"{len(jobs)} kernel run(s) queued, {MAX_CONCURRENT} concurrent")

    try:
        done, failed = drive(jobs, args.force)
    except QuotaExhausted as e:
        log(str(e))
        raise
    log(f"=== {len(done)} complete, {len(failed)} failed, "
        f"{(time.time() - t0) / 3600:.1f}h wall ===")
    if failed:
        for job in failed:
            log(f"  FAILED {job['kid']}")
        raise SystemExit(f"{len(failed)} kernel(s) failed - re-run them whole, do not stitch")
    log("next: python pipeline/05b_score_noplant_control.py --all --dump")


if __name__ == "__main__":
    main()
