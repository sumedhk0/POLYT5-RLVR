"""Chain the remaining round-1 GRPO arms after the in-flight one finishes.

Phase 3. NOT part of the published polyT5 method -- see ``docs/rlvr_plan.md``.

Accepts any arm name with a ``configs/rl/<arm>.yaml`` -- this driver never
allow-lists arm names itself (see :func:`_experiment_name_for`), so it needs
no change to pick up new arms. 2026-08-23: Tg was dropped from every RLVR
reward -- see ``polyt5.rewards.composite``'s module docstring -- so the
current arm set to chain is ``validity``/``novelty``/``synthesisability``/
``composite``/``control`` (``constraint`` kept as a redefined multi-criterion
conjunction; ``accuracy`` retired, not chained for new runs).

One GPU means the arms run sequentially, and at the measured ~23 s/step a
round is roughly two days per arm. This driver removes the need for anyone to
be present at each arm boundary: it waits for the currently running arm to
reach its final step, then launches the rest in order.

It refuses to chain past a failure. An arm whose metrics stop advancing for
``--stall-minutes`` is treated as dead, and the remaining arms are NOT started --
a stalled GPU job that silently hands off to the next arm would burn the rest of
the budget producing nothing, which is exactly the failure this study cannot
afford to discover late.

Multi-seed chaining (Addition 2)
---------------------------------
``--seeds`` chains every arm in ``--then`` across more than one training seed,
seed-OUTER / arm-INNER: every arm in ``--then`` finishes seed 0 before ANY arm
starts seed 1, so an early ``scripts/compare_arms.py`` run after just the first
pass already has a comparable row for every requested arm, rather than a full
multi-seed picture for only the first arm. Defaults to ``[0]`` -- today's
single-seed behaviour, unchanged (same command line, same log file name)
unless ``--seeds`` is actually passed. Seed 0 is never given ``--set seed=0``
on the ``train_grpo.py`` command line (matching ``configs/rl/*.yaml``'s own
``seed: 0``); every other seed gets ``--set seed=<N>``, which
``train_grpo.run_experiment_name`` (mirrored here by :func:`run_dir_for`)
turns into a ``results/grpo_<arm>_seed<N>/`` run directory distinct from every
other seed's.

Usage::

    python scripts/run_round1.py --wait-for validity \\
        --then novelty synthesisability composite control
    python scripts/run_round1.py --then novelty synthesisability composite constraint \\
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

#: Steps an arm must reach to count as finished. Mirrors every
#: ``configs/rl/*.yaml``'s ``train.max_steps``; ``--target-steps`` overrides it.
TARGET_STEPS = 2000


def _experiment_name_for(arm: str) -> str:
    """The base run-directory name ``train_grpo.py`` uses for ``arm`` at seed 0.

    Review finding 3: an earlier version of this function hardcoded
    ``f"grpo_{arm}"`` instead of reading ``configs/rl/<arm>.yaml``'s own
    ``experiment_name`` key, so a YAML that overrode it would silently
    desynchronise this driver from the directory ``train_grpo.py`` actually
    writes to -- ``last_step`` would poll a directory that never gets a
    ``metrics.csv``, and ``wait_for`` would either hang forever (while
    ``trainer_is_alive()`` stays ``True``, which it will) or abort the whole
    chain after the stall budget with a misleading "stuck at step None".

    Reads the plain YAML key directly (``yaml.safe_load``, already a
    dependency) rather than importing ``train_grpo.py`` or
    ``polyt5.utils.load_config`` -- this driver's whole job is polling small
    files and spawning subprocesses, and stays free of ``torch``/``rdkit``
    at import time on purpose (``train_grpo.py`` pulls both in at module
    level).

    Mirrors ``train_grpo.main()``'s own fallback exactly:
    ``cfg.get("experiment_name") or f"grpo_{arm}"``. A missing or unreadable
    config file falls back to the same ``f"grpo_{arm}"`` default rather than
    raising -- this function is called before every poll, and a driver that
    cannot start because a YAML briefly failed to read is worse than one
    that falls back to the common-case name.
    """
    config_path = REPO / "configs" / "rl" / f"{arm}.yaml"
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError:
        raw = {}
    return raw.get("experiment_name") or f"grpo_{arm}"


def _run_experiment_name(arm: str, seed: int) -> str:
    """Mirrors ``train_grpo.run_experiment_name`` without importing it.

    ``train_grpo.py`` imports ``torch``/``rdkit`` at module level; this
    driver's whole job is polling small files and spawning subprocesses, so
    it deliberately never imports that module -- see
    ``test_run_dir_for_follows_a_configs_own_experiment_name_override`` for
    the test that pins base-name derivation (not just the seed-suffix rule)
    against the real YAML files.
    """
    base = _experiment_name_for(arm)
    return base if seed == 0 else f"{base}_seed{seed}"


def run_dir_for(arm: str, seed: int = 0) -> Path:
    return REPO / "results" / _run_experiment_name(arm, seed)


def last_step(arm: str, seed: int = 0) -> int | None:
    """Highest step recorded in an arm's metrics.csv, or None if unreadable."""
    path = run_dir_for(arm, seed) / "metrics.csv"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    steps = [int(r["step"]) for r in rows if r.get("step", "").isdigit()]
    return max(steps) if steps else None


def trainer_is_alive() -> bool | None:
    """Whether a train_grpo.py process exists. None if that cannot be determined.

    Liveness, not progress, is what distinguishes a dead run from a slow one --
    and it is the only signal that survives the machine suspending. Wall clock
    does not: ``time.monotonic()`` on Windows counts suspended time, so a laptop
    that slept overnight looks identical to a hung trainer.
    """
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" |"
             " Where-Object { $_.CommandLine -like '*train_grpo.py*' } |"
             " Measure-Object).Count"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip()) > 0
    except ValueError:
        return None


def wait_for(arm: str, *, target_steps: int, stall_seconds: float, poll: float) -> bool:
    """Block until `arm` reaches target_steps. False only if it is truly dead.

    Aborts when the step count has not advanced for `stall_seconds` AND no
    trainer process exists. Both conditions are required: a machine that
    suspends, or a GPU throttled to a third of its clock on battery power,
    stops advancing steps while remaining perfectly healthy -- an earlier
    version of this function aborted exactly that case after charging ~17 h
    of laptop sleep against its stall budget.
    """
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
        if stalled <= stall_seconds:
            continue
        alive = trainer_is_alive()
        if alive is None:
            print(f"[driver] arm={arm} at step {last_seen}, no progress for "
                  f"{stalled / 60:.0f} min; liveness undetermined, still waiting.", flush=True)
            last_change = time.monotonic()
            continue
        if alive:
            print(f"[driver] arm={arm} at step {last_seen}, no progress for "
                  f"{stalled / 60:.0f} min, but the trainer is alive (suspended or "
                  f"throttled?). Still waiting.", flush=True)
            last_change = time.monotonic()
            continue
        print(f"[driver] ABORT: arm={arm} stuck at step {last_seen} for "
              f"{stalled / 60:.0f} min and no trainer process is running. "
              f"Not starting the remaining arms.", flush=True)
        return False


def resume_target(arm: str, seed: int = 0, *, target_steps: int = TARGET_STEPS) -> Path | None:
    """The checkpoint directory to resume ``arm`` from, or None to start fresh.

    An interrupted arm is the common case on a laptop, not the exception: this
    study has lost a partial arm to a shutdown three times. Without this the
    driver restarts from step 0 and discards every completed step, which at
    ~47 s/step is half a day for an arm interrupted near the end.

    Resuming is not a different experiment. ``train_grpo.py --resume`` restores
    policy weights, optimizer state AND the step count, and a step's RNG is
    seeded from ``(config.seed, step_index)`` alone -- so a resumed step samples
    exactly what an uninterrupted run would have sampled at that index.

    Returns None when the arm has no checkpoints, or when it already reached
    ``target_steps`` (a finished arm must not be reopened).
    """
    checkpoints = run_dir_for(arm, seed) / "checkpoints"
    if not checkpoints.is_dir():
        return None
    if not any(checkpoints.glob("step_*.pt")):
        return None
    done = last_step(arm, seed)
    if done is not None and done >= target_steps:
        return None
    return checkpoints


def train(arm: str, seed: int = 0, *, resume: bool = True) -> bool:
    """Run one arm, at one seed, to completion. False on a non-zero exit.

    ``seed == 0`` (the default) issues EXACTLY the command line and log file
    name this function always used before multi-seed support existed --
    ``train_grpo.py`` is never given ``--set seed=0`` explicitly, matching
    every ``configs/rl/*.yaml``'s own ``seed: 0``. Every other seed appends
    ``--set seed=<N>``, which lands in ``results/grpo_<arm>_seed<N>/`` (see
    ``run_dir_for`` / ``train_grpo.run_experiment_name``).
    """
    label = arm if seed == 0 else f"{arm} (seed {seed})"
    log_name = f"rl_{arm}_round1.log" if seed == 0 else f"rl_{arm}_seed{seed}_round1.log"
    log_path = REPO / "results" / log_name
    cmd = [PYTHON, "-u", str(REPO / "scripts" / "train_grpo.py"), "--arm", arm]
    if seed != 0:
        cmd += ["--set", f"seed={seed}"]
    resume_from = resume_target(arm, seed) if resume else None
    if resume_from is not None:
        cmd += ["--resume", str(resume_from)]
        print(f"[driver] arm={label} has partial progress at step {last_step(arm, seed)}; "
              f"resuming from {resume_from}", flush=True)
    print(f"[driver] starting arm={label} -> {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False,
        )
    if completed.returncode != 0:
        print(f"[driver] ABORT: arm={label} exited {completed.returncode}; "
              f"see {log_path}. Remaining arms not started.", flush=True)
        return False
    print(f"[driver] arm={label} finished", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", default=None,
                        help="Arm already running; chain starts once it completes.")
    parser.add_argument("--then", nargs="+", required=True, help="Arms to run, in order.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Restart a partially-trained arm from step 0 instead of "
                             "resuming from its newest checkpoint.")
    parser.add_argument("--target-steps", type=int, default=TARGET_STEPS,
                        help="Step count that marks the waited-for arm complete.")
    parser.add_argument("--stall-minutes", type=float, default=45.0,
                        help="Treat the waited-for arm as dead after this long with no new step.")
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Seeds to chain every arm in --then across, seed-outer / "
                             "arm-inner (every arm finishes seed 0 before any arm starts "
                             "seed 1). Defaults to [0] -- today's single-seed behaviour, "
                             "with an identical train_grpo.py command line and log file "
                             "name to before this flag existed.")
    args = parser.parse_args()

    if args.wait_for and not wait_for(
        args.wait_for,
        target_steps=args.target_steps,
        stall_seconds=args.stall_minutes * 60.0,
        poll=args.poll_seconds,
    ):
        return 1

    for seed in args.seeds:
        for arm in args.then:
            if not train(arm, seed, resume=not args.no_resume):
                return 1

    print(f"[driver] round 1 complete: {', '.join(args.then)} across seeds {args.seeds}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
