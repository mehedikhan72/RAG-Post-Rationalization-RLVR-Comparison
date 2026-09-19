"""Kills `kaggle` CLI processes alive longer than --max-min.

The Kaggle CLI intermittently wedges on a network read and never returns.
`run_pipeline.py` documents this for `kernels status`; it also happens on `kernels push`,
where the push has ALREADY landed server-side and only the client is stuck -- so killing it
is safe and the driver moves on. Observed 2026-09-06: a push hung 26 minutes while
subprocess.communicate(timeout=600) failed to fire, blocking the whole 16-run queue.

Python rather than shell because macOS `ps` has no `etimes` keyword (BSD ps); the first
version of this used it and silently reaped nothing.

    python pipeline/scripts/reap_stuck_kaggle.py --max-min 12
"""

import argparse
import os
import re
import signal
import subprocess
import time
from datetime import datetime

ETIME = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$")


def etime_seconds(s):
    """Parse BSD ps ELAPSED: [[dd-]hh:]mm:ss"""
    m = ETIME.match(s.strip())
    if not m:
        return None
    days, hours, mins, secs = (int(g or 0) for g in m.groups())
    return ((days * 24 + hours) * 60 + mins) * 60 + secs


def sweep(max_seconds):
    if max_seconds < 60:
        raise ValueError("refusing to sweep with a threshold under 60s -- that kills "
                         "healthy in-flight Kaggle calls, not just wedged ones")
    out = subprocess.run(["ps", "-eo", "pid,etime,command"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        cmd = parts[2]
        argv0 = cmd.split()[0] if cmd.split() else ""
        exe_is_kaggle = argv0.endswith("/kaggle") or argv0 == "kaggle"
        # `python .../bin/kaggle kernels push ...` -- the interpreter is argv0, so also
        # accept a second token that is the kaggle entry point.
        toks = cmd.split()
        if not exe_is_kaggle:
            exe_is_kaggle = any(tok.endswith("/bin/kaggle") for tok in toks[:2])
        if not exe_is_kaggle:
            continue
        age = etime_seconds(parts[1])
        if age is None or age <= max_seconds:
            continue
        pid = int(parts[0])
        print(f"{datetime.now():%H:%M:%S} reaping stuck kaggle CLI pid={pid} "
              f"age={age // 60}min :: {parts[2][:80]}", flush=True)
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as e:
            print(f"  could not kill {pid}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-min", type=float, default=12)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    print(f"{datetime.now():%H:%M:%S} reaper up: killing kaggle CLI older than "
          f"{args.max_min} min, every {args.interval}s", flush=True)
    while True:
        sweep(args.max_min * 60)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
