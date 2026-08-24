# RLVR run runbook

How to stop, resume, and chain the Phase 3 GRPO runs. Phase 3 is **not** part of the
published polyT5 method — see [`rlvr_plan.md`](rlvr_plan.md).

This file exists because the runs take days on one GPU and the machine gets turned off.
Everything here is recoverable from disk; nothing depends on a live session.

## The study

Round 1 started with four arms (`accuracy`, `validity`, `composite`, `constraint`), run
sequentially on one GPU. **2026-08-23: Tg was dropped from every reward still being
trained** — see [`rlvr_plan.md`](rlvr_plan.md)'s superseded banner and
[`src/polyt5/rewards/composite.py`](../src/polyt5/rewards/composite.py)'s module
docstring. `accuracy` had already completed (below) and is now **retired**, not
re-run; `composite` and `constraint` were **redefined** to fully verifiable, Tg-free
rewards; two new arms, `novelty` and `synthesisability`, were added. The Tg-reward-source
table below describes ONLY `accuracy`'s already-completed run — no arm still to be
trained reads a Tg prediction at all.

| Round | Tg reward source | Scored by |
|---|---|---|
| 1 (`accuracy` only, complete) | the 4-model learned ensemble (splits 0–3) | split-4 auditor, untouched |
| 2 (not started; would need a Tg-reading arm to exist again) | the same 4 models **plus** the group-contribution oracle | the same split-4 auditor |

Current arms: `validity`, `novelty`, `synthesisability`, `composite`, `constraint`,
`control` (plus the retired `accuracy`, kept only for reproducibility — see
`configs/rl/accuracy.yaml`). Run directory per arm is `results/grpo_<arm>/`.

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
python -c "import csv;rows=list(csv.DictReader(open('results/grpo_validity/metrics.csv')));print(max(int(r['step']) for r in rows))"
ls results/grpo_validity/checkpoints/
```

`metrics.csv` may be ahead of the newest checkpoint — that is normal, and the gap is
exactly what a resume replays.

## Resuming an interrupted arm

`--resume` restores policy weights, optimizer state **and** step count, so the run
continues rather than restarting:

```bash
# resume from the newest checkpoint in the run directory
python scripts/train_grpo.py --arm validity --resume results/grpo_validity/checkpoints

# or from one specific checkpoint
python scripts/train_grpo.py --arm validity --resume results/grpo_validity/checkpoints/step_001000.pt
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
# after `validity` finishes, run the rest of the current arm set back to back
python scripts/run_round1.py --wait-for validity --then novelty synthesisability composite constraint control

# nothing in flight — just run them in order
python scripts/run_round1.py --then novelty synthesisability composite constraint control
```

Launch it in the background so it survives the terminal:

```bash
python -u scripts/run_round1.py --wait-for validity --then novelty synthesisability composite constraint control \
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
| `ensemble_partial_fraction` | low | rising: drifting toward chemistry the ensemble cannot parse (only meaningful for `accuracy`'s completed run; no current arm reads the ensemble) |
| `max_tanimoto_mean` | stable | falling: leaving the training distribution's support |
| `reward_mean` | rising | rising *while* `duplicate_rate`/`unique_fraction` moves the wrong way (e.g. mode collapse, as `accuracy` showed) or `max_tanimoto_mean` rises toward near-copies — reward hacking |

That last row is the one that matters. Reward rising while a diagnostic OUTSIDE the
reward gets worse is the failure this whole apparatus exists to detect. Every current
arm's reward is fully verifiable (see `rlvr_plan.md`'s superseded banner), so this is
no longer specifically about a Tg proxy diverging from the truth — it is the general
GRPO risk that optimising the reward exactly as specified still damages an axis the
reward does not measure at all (e.g. `composite`'s diversity term exists precisely to
watch for the collapse `accuracy` showed on an axis its own reward never priced in).

## After round 1

```bash
python scripts/compare_arms.py
```

Writes `results/arm_comparison/{matrix.csv,matrix.md,summary.json}`. It aborts if the
novelty index is missing (`--allow-missing-novelty-index` to override, which disables
duplicate detection for every row) and refuses to score an arm whose sampling settings
do not match Arm B's protocol, since that row would not be comparable.

**`accuracy` has been trained to completion; it is retired and reported as a motivating
negative result (reward-scored error 52.5 -> 31.4 K while `unique_fraction` fell 0.951 ->
0.535). No arm has been through `compare_arms.py`, so this repository reports no final RLVR
result.**

## Commit provenance note

`51cd00f` carries a destroyed commit message — the literal text `$(cat <<'EOF'` —
because an amend's heredoc failed. Its content is intact and reviewed: it is
**Phase 4 Group A, Task 12**, adding `src/polyt5/evaluation/ablation.py`, its package
export, and `tests/test_group_a_ablation.py` (475 insertions). That commit implements
the **pre-registered verdict**: `helps` below 27.9142 K, `no effect` inside the
baseline's own spread, `hurts` above it, with the threshold derived from
caller-supplied baseline statistics and provably not recomputable from the arms' own
results.

Its tree hash is identical to the pre-amend commit `50d9484`, so the review packaged
against that SHA remains valid.

The message was not repaired by rewriting history: doing so would have meant
rebasing twelve reviewed commits immediately before a long GPU run, risking real work
to fix a cosmetic defect in a commit whose content is sound.
