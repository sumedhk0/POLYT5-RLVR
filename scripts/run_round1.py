"""Chain the remaining round-1 GRPO arms after the in-flight one finishes.

Phase 3. NOT part of the published polyT5 method -- see ``docs/rlvr_plan.md``.

One GPU means the four arms run sequentially, and at the measured ~23 s/step a
round is roughly two days. This driver removes the need for anyone to be present
at each arm boundary: it waits for the currently running arm to reach its final
step, then launches the rest in order.

It refuses to chain past a failure. An arm whose metrics stop advancing for
``--stall-minutes`` is treated as dead, and the remaining arms are NOT started --
a stalled GPU job that silently hands off to the next arm would burn the rest of
the budget producing nothing, which is exactly the failure this study cannot
afford to discover late.

Usage::

    python scripts/run_round1.py --wait-for accuracy --then validity composite constraint
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_dir_for(arm: str) -> Path:
    return REPO / "results" / f"grpo_{arm}"


def last_step(arm: str) -> int | None:
    """Highest step recorded in an arm's metrics.csv, or None if unreadable."""
    path = run_dir_for(arm) / "metrics.csv"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    steps = [int(r["step"]) for r in rows if r.get("step", "").isdigit()]
    return max(steps) if steps else None


def wait_for(arm: str, *, target_steps: int, stall_seconds: float, poll: float) -> bool:
    """Block until `arm` reaches target_steps. False if it stalls or dies."""
    last_seen, last_change = last_step(arm), time.monotonic()
    print(f"[driver] waiting for arm={arm} to reach step {target_steps} "
          f"(currently {last_seen})", flush=True)
    while True:
        time.sleep(poll)
        current = last_step(arm)
        if current is not None and current >= target_steps:
            print(f"[driver] arm={arm} reached step {current}", flush=True)
            return True
        if current != last_seen:
            last_seen, last_change = current, time.monotonic()
            continue
        stalled = time.monotonic() - last_change
        if stalled > stall_seconds:
            print(f"[driver] ABORT: arm={arm} stuck at step {last_seen} for "
                  f"{stalled / 60:.0f} min. Not starting the remaining arms.", flush=True)
            return False


def train(arm: str) -> bool:
    """Run one arm to completion. False on a non-zero exit."""
    log_path = REPO / "results" / f"rl_{arm}_round1.log"
    print(f"[driver] starting arm={arm} -> {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [PYTHON, "-u", str(REPO / "scripts" / "train_grpo.py"), "--arm", arm],
            stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False,
        )
    if completed.returncode != 0:
        print(f"[driver] ABORT: arm={arm} exited {completed.returncode}; "
              f"see {log_path}. Remaining arms not started.", flush=True)
        return False
    print(f"[driver] arm={arm} finished", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", default=None,
                        help="Arm already running; chain starts once it completes.")
    parser.add_argument("--then", nargs="+", required=True, help="Arms to run, in order.")
    parser.add_argument("--target-steps", type=int, default=2000,
                        help="Step count that marks the waited-for arm complete.")
    parser.add_argument("--stall-minutes", type=float, default=45.0,
                        help="Treat the waited-for arm as dead after this long with no new step.")
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    args = parser.parse_args()

    if args.wait_for and not wait_for(
        args.wait_for,
        target_steps=args.target_steps,
        stall_seconds=args.stall_minutes * 60.0,
        poll=args.poll_seconds,
    ):
        return 1

    for arm in args.then:
        if not train(arm):
            return 1

    print(f"[driver] round 1 complete: {', '.join(args.then)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
