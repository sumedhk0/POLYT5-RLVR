# RLVR run runbook

How to stop, resume, and chain the Phase 3 GRPO runs. Phase 3 is **not** part of the
published polyT5 method — see [`rlvr_plan.md`](rlvr_plan.md).

This file exists because the runs take days on one GPU and the machine gets turned off.
Everything here is recoverable from disk; nothing depends on a live session.

## The study

Two rounds, four arms each, run sequentially on one GPU.

| Round | Tg reward source | Scored by |
|---|---|---|
| 1 | the 4-model learned ensemble (splits 0–3) | split-4 auditor, untouched |
| 2 | the same 4 models **plus** the group-contribution oracle | the same split-4 auditor |

Arms: `accuracy`, `validity`, `composite`, `constraint`. Run directory per arm is
`results/grpo_<arm>/`.

**Measured cost:** ~21–23 s/step, so ~12–13 h per arm at the configured 2000 steps.
Checkpoints are written every `train.save_every` steps (see the arm's `config.yaml`).

## Turning the machine off

Safe at any time. An interrupted run loses only the steps since its last checkpoint —
at `save_every: 250` that is up to ~90 minutes, which is why later arms use a shorter
cadence (see "Reducing what an interruption costs").

To stop deliberately rather than by power-off, kill the `train_grpo.py` process. The
checkpoint on disk is complete: checkpoints are written atomically, so a half-written
file is not a failure mode you need to check for.

## Checking where a run got to

```bash
python -c "import csv;rows=list(csv.DictReader(open('results/grpo_accuracy/metrics.csv')));print(max(int(r['step']) for r in rows))"
ls results/grpo_accuracy/checkpoints/
```

`metrics.csv` may be ahead of the newest checkpoint — that is normal, and the gap is
exactly what a resume replays.

## Resuming an interrupted arm

`--resume` restores policy weights, optimizer state **and** step count, so the run
continues rather than restarting:

```bash
# resume from the newest checkpoint in the run directory
python scripts/train_grpo.py --arm accuracy --resume results/grpo_accuracy/checkpoints

# or from one specific checkpoint
python scripts/train_grpo.py --arm accuracy --resume results/grpo_accuracy/checkpoints/step_001000.pt
```

Determinism note: a step's RNG is seeded from `(config.seed, step_index)` alone, so a
resumed step samples exactly what it would have sampled in an uninterrupted run. A
resume is not a different experiment.

## Chaining the remaining arms unattended

`scripts/run_round1.py` waits for the in-flight arm to reach its final step, then runs
the rest in order. It **refuses to chain past a failure** — an arm whose metrics stop
advancing for `--stall-minutes` aborts the chain instead of handing a dead GPU job to
the next arm.

```bash
# after `accuracy` finishes, run the other three back to back
python scripts/run_round1.py --wait-for accuracy --then validity composite constraint

# nothing in flight — just run them in order
python scripts/run_round1.py --then validity composite constraint
```

Launch it in the background so it survives the terminal:

```bash
python -u scripts/run_round1.py --wait-for accuracy --then validity composite constraint \
    > results/round1_driver.log 2>&1 &
```

## Reducing what an interruption costs

`save_every: 250` at ~22 s/step means up to ~90 minutes lost to an unplanned shutdown.
Shortening it is safe — checkpoint cadence has no effect on the objective, the reward,
or the pre-registered success criterion, and `--set` only refuses **reward** parameters
(`sigma0`, `sigma_unknown`, `min_coverage`), which this is not.

```bash
python scripts/train_grpo.py --arm validity --set train.save_every=100
```

Cost is disk: 89.7 MB per checkpoint, so `save_every=100` is ~1.8 GB per arm.

## What to watch in `metrics.csv`

Stop a run and investigate if any of these appear:

| Column | Healthy | Bad |
|---|---|---|
| `clip_fraction` | ~0.0 early | climbing early — ratio drifting for non-policy reasons |
| `zero_variance_group_fraction` | ~0.0 | → 1.0: groups all score alike, steps are no-ops |
| `ensemble_partial_fraction` | low | rising: drifting toward chemistry the ensemble cannot parse |
| `ensemble_backed_rate` (C4) | stable | → 0: the constraint arm is starving under `min_coverage` |
| `max_tanimoto_mean` | stable | falling: leaving the predictor's support |
| `reward_mean` | rising | rising *while* `abs_error_mean` also rises — reward hacking |

That last row is the one that matters. Reward rising while the thing it proxies gets
worse is the failure this whole apparatus exists to detect.

## After round 1

```bash
python scripts/compare_arms.py
```

Writes `results/arm_comparison/{matrix.csv,matrix.md,summary.json}`. It aborts if the
novelty index is missing (`--allow-missing-novelty-index` to override, which disables
duplicate detection for every row) and refuses to score an arm whose sampling settings
do not match Arm B's protocol, since that row would not be comparable.

**No arm has been trained to completion yet. This repository reports no RLVR result.**
