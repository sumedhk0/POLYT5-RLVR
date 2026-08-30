#!/usr/bin/env python3
"""Chain the round-2 ``kl_coef`` sweep.

Round 1 scored every Tg-free reward as destroying conditioning: ``composite`` ended at
TP 0.112 against the baseline's 0.693, with the per-target slope down to 0.274 from
0.904. ``kl_coef`` is the only force preserving conditioning during RL, and at round 1's
0.02 the KL still drifted to 0.073 over 2000 steps. This sweep asks whether a stronger
anchor to the reference policy keeps conditioning while retaining the structural gain.
It changes no reward term, so unlike re-adding a Tg term it costs nothing in
verifiability.

``run_round1.py`` cannot drive this: its ``--then`` values are passed to
``train_grpo.py --arm``, which allow-lists REWARD arm names. Every run here is the same
``composite`` reward under a different config, so each needs ``--arm composite --config
configs/rl/composite_kl<N>.yaml``.

    python scripts/run_kl_sweep.py
    python scripts/run_kl_sweep.py --configs configs/rl/composite_kl1.yaml
    python scripts/run_kl_sweep.py --max-steps 50        # smoke

Like ``run_round1.py`` it refuses to continue past a failure, and it resumes a partially
trained run rather than discarding it.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

#: Steps a run must reach to count as finished; mirrors ``train.max_steps``.
TARGET_STEPS = 2000

DEFAULT_CONFIGS = [
    REPO / "configs" / "rl" / "composite_kl05.yaml",
    REPO / "configs" / "rl" / "composite_kl1.yaml",
    REPO / "configs" / "rl" / "composite_kl2.yaml",
]


def experiment_name(config: Path) -> str:
    """The run directory ``train_grpo.py`` will write to for ``config``."""
    import yaml

    loaded = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return str(loaded.get("experiment_name") or f"grpo_{config.stem}")


def last_step(run_dir: Path) -> int | None:
    """Highest step recorded, or None if the run has written no metrics."""
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            steps = [int(r["step"]) for r in csv.DictReader(handle) if r.get("step")]
    except (OSError, csv.Error, ValueError):
        return None
    return max(steps) if steps else None


def resume_target(run_dir: Path, target_steps: int) -> Path | None:
    """Checkpoint directory to resume from, or None to start fresh.

    None when there are no checkpoints, and None when the run already reached
    ``target_steps`` -- a finished run must never be reopened and overwritten.
    """
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir() or not any(checkpoints.glob("step_*.pt")):
        return None
    done = last_step(run_dir)
    if done is not None and done >= target_steps:
        return None
    return checkpoints


def train(config: Path, *, resume: bool = True, max_steps: int | None = None) -> bool:
    """Run one sweep point to completion. False on a non-zero exit."""
    name = experiment_name(config)
    run_dir = REPO / "results" / name
    log_path = REPO / "results" / f"rl_{name}.log"
    cmd = [
        PYTHON, "-u", str(REPO / "scripts" / "train_grpo.py"),
        "--arm", "composite", "--config", str(config),
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    target = TARGET_STEPS if max_steps is None else max_steps
    if resume:
        resume_from = resume_target(run_dir, target)
        if resume_from is not None:
            cmd += ["--resume", str(resume_from)]
            print(f"[sweep] {name} has partial progress at step {last_step(run_dir)}; "
                  f"resuming from {resume_from}", flush=True)
        elif last_step(run_dir) is not None and last_step(run_dir) >= target:
            print(f"[sweep] {name} already complete at step {last_step(run_dir)}; skipping",
                  flush=True)
            return True
    print(f"[sweep] starting {name} ({config.name}) -> {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False,
        )
    if completed.returncode != 0:
        print(f"[sweep] ABORT: {name} exited {completed.returncode}; see {log_path}. "
              f"Remaining configs not started.", flush=True)
        return False
    print(f"[sweep] {name} finished", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke test)")
    parser.add_argument("--no-resume", action="store_true",
                        help="restart a partial run from step 0 instead of resuming")
    args = parser.parse_args(argv)

    for config in args.configs:
        if not config.is_file():
            print(f"[sweep] ABORT: no such config {config}", flush=True)
            return 1
        if not train(config, resume=not args.no_resume, max_steps=args.max_steps):
            return 1
    print(f"[sweep] complete: {len(args.configs)} configs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
