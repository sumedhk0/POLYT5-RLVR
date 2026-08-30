# Phase 3 results — round 1, seed 0

**Our RLVR extension obtains** the numbers below. They are not the paper's, and they are not
our Phase 1–2 reproduction's. Three claims never merge — see
[`reproduction.md`](reproduction.md) §9. polyT5 (Sahu et al., npj AI 2026) contains no
reinforcement learning; everything here is ours.

Produced by `scripts/compare_arms.py` on 2026-08-24 against
`artifacts/baseline/frozen_baseline.json`. Protocol: targets 300/400/500 K, n=500 each
(1,500 samples per arm), T=0.70 top_p=0.95 matching Arm B, tolerance 50 K.
**One seed (0).**

## The matrix

| | Arm A | Arm B | control | **validity** | **accuracy** |
|---|---|---|---|---|---|
| | baseline | baseline | random reward | PV cascade | Tg closeness |
| **PV rate** | 0.521 | 0.526 | 0.515 | **0.901** | 0.284 |
| SV rate | 0.999 | 1.000 | 0.999 | 1.000 | 0.999 |
| duplicate rate | 0.092 | 0.121 | 0.129 | 0.069 | **0.635** |
| SR rate | 0.376 | 0.451 | 0.452 | 0.779 | 0.763 |
| novelty rate | 0.569 | 0.597 | 0.594 | **0.963** | 0.599 |
| near-copy fraction | 0.223 | 0.229 | 0.235 | **0.585** | 0.456 |
| mean length | 63.0 | 63.2 | 62.9 | **130.7** | 59.2 |
| auditor Tg MAE (K) | 34.7 | 36.6 | 36.3 | **150.4** | **21.8** |
| auditor TP rate | 0.716 | 0.693 | 0.693 | **0.047** | **0.869** |

## Pre-registered verdicts

| arm | metric | Δ vs Arm B | 95% CI | success |
|---|---|---|---|---|
| validity | `pv_rate` | **+0.375** | [0.347, 0.401] | **yes** |
| accuracy | `accuracy_score` | **+0.342** | [0.324, 0.362] | **yes** |
| control | — | — | — | `None` |

Both cleared their pre-registered margin. `control` correctly reports `success=None` — it
optimised nothing, so it cannot win or lose.

Validity's audit column reads **"N/A — structural metric, no predictor involved"**: PV is
RDKit chemistry, and there is no model opinion to audit. Accuracy's gain *was* audit-confirmed
— Δ_auditor +0.363, CI [0.337, 0.388] — by split 4, which never entered any reward path.

## What the control establishes

`control` — a uniform random, candidate-independent reward — landed at PV 0.515 against
Arm B's 0.526, duplicate rate 0.129 vs 0.121, length 62.9 vs 63.2, auditor MAE 36.3 vs 36.6.
**Flat on every axis.**

That was not guaranteed. GRPO could plausibly have moved these metrics on its own through KL
drift, entropy change, or a sampling shift unrelated to any reward. It did not. So every
movement in the other two arms is attributable to their reward rather than to the algorithm.

Without this arm, nothing below would be a finding.

## The result: both arms won, and both broke something else

**Validity** bought a verified +37.5-point PV gain and paid for it on every axis its reward
did not price:

- auditor Tg MAE **36.6 → 150.4 K**, four times worse
- auditor TP rate **0.693 → 0.047**
- mean length **63 → 131**
- near-copy fraction **0.229 → 0.585**

It learned to write long, near-duplicate, chemically-valid strings. Every one of those PV
passes is real and checkable by anyone with RDKit. The polymers are far worse.

**Accuracy** is the mirror image. Auditor MAE **36.6 → 21.8 K** — a 40% improvement confirmed
by a model that never saw its reward — while PV **collapsed 0.526 → 0.284** and the duplicate
rate hit **0.635**.

Neither arm is a failure of its objective. Both are failures of everything else, and the
off-diagonal is what makes that visible.

## Two things the metrics themselves revealed

**Novelty-as-exact-match is not novelty.** Validity's `novelty_rate` rose to **0.963** while its
`near_copy_fraction` rose to **0.585**. The hash index says 96% of its output is novel; ECFP6
Tanimoto says 58% are near-copies of training polymers. Both numbers are correct — novelty is
defined as exact canonical non-membership, so a one-atom edit counts as new chemistry. This
limitation was documented before the run and is now measured.

**SR improved in both arms without being rewarded.** SELFIES round-trip fidelity went
0.451 → 0.779 (validity) and 0.763 (accuracy), against a flat control at 0.452. Neither reward
mentions SR. This is a genuine off-diagonal *gain*, and unexplained.

## Caveats that bound these numbers

**One seed.** `n_seeds=1`, so `across_seed_unanimous=yes` is vacuous — a single run cannot
disagree with itself. The matrix prints `spread=[1 seed]` rather than a fabricated 0, but the
across-seed criterion is not meaningfully exercised. Nothing here is protected against
seed-to-seed variance.

**Tg numbers are model-scored, and the auditor is a sibling.** Split 4 never entered a reward
path, but it shares roughly 80% of its training data with each reward model. It detects
ensemble-specific error well and corpus-wide error barely. See
[`instrument_audit.md`](instrument_audit.md).

**Validity's length inflation is a confound, not a footnote.** Longer sequences have more ways
to satisfy the cascade. The PV gain is verified; whether it reflects better chemistry or more
of it is not settled by these numbers.

## Reproducing

```bash
python scripts/compare_arms.py --arms accuracy validity control
```

Outputs `results/arm_comparison/{matrix.csv,matrix.md,summary.json,manifest.json}`. The
manifest records checkpoint SHA-256s, the novelty index hash, and every reward config hash, so
a row can be traced to the exact artifacts that produced it.

## Round 2, arm `novelty` — trained to completion 2026-08-27

Reward: mean ECFP6 Tanimoto DISTANCE from the training corpus (5,275 reference
fingerprints). Fully verifiable, no learned model anywhere in the reward path.

The reward went up and the population collapsed:

| step | reward | unique_fraction | zero_variance_groups | KL |
|---|---|---|---|---|
| 10 | 0.578 | 0.949 | 0.000 | 7.2e-06 |
| 250 | 0.721 | 0.949 | 0.000 | 0.0053 |
| 500 | 0.744 | 0.727 | 0.094 | 0.0220 |
| 1000 | 0.879 | 0.609 | 0.313 | 0.0639 |
| 1500 | 0.908 | 0.564 | 0.469 | 0.0776 |
| **2000** | **0.949** | **0.516** | **0.594** | 0.0968 |

`reward_mean` 0.578 -> 0.949 while `unique_fraction` 0.949 -> 0.516. That final
diversity figure is within noise of the retired `accuracy` arm's 0.535, which was
disqualified for exactly this behaviour.

**The mechanism is in the reward's definition, not in GRPO.** Novelty is measured as
distance from the TRAINING CORPUS, never as diversity within the sampled batch. A
thousand copies of one novel polymer therefore score perfectly. Nothing in the objective
prefers a varied population over a single lucky motif repeated, so the optimiser is doing
precisely what it was asked, and what it was asked is not what was wanted.

`zero_variance_group_fraction` reaching 0.594 makes the cost concrete: in 59% of groups
all 16 samples scored identically at the end. GRPO's advantage is
``(r - mean(r)) / std(r)`` within a group, so those groups contribute NO gradient. By
step 2000 the majority of the compute was buying nothing.

`clip_fraction` stayed at 0.0 throughout and KL rose smoothly to 0.097. The optimiser was
healthy the whole way; there is no training bug to find here. This is the reward working.

**Third arm, same shape.** `accuracy` (retired) drove reward-scored error 52.5 -> 31.4 K
while `unique_fraction` fell 0.951 -> 0.535. `validity` reached PV 0.901 while its
near-copy fraction rose to 0.585 and its auditor Tg MAE went 36.6 -> 150.4 K. Now
`novelty`. The pattern across all three: **a verifiable reward optimises exactly what it
measures and silently damages whatever it does not.** The verifiability is real; it buys
correctness of the measurement, not completeness of the objective.

This is why `composite` carries an explicit diversity term. If `composite` holds
`unique_fraction` where `novelty` did not, that is a controlled comparison rather than an
anecdote, and it is the most useful thing round 2 can produce.

Full per-step trajectory: `results/grpo_novelty/trajectory_summary.json`.

## Round 2, arm `synthesisability` — trained to completion 2026-08-28

Reward: SA score (synthetic accessibility). Fully verifiable, no learned model.

| step | reward | unique_fraction | zero_variance_groups | KL |
|---|---|---|---|---|
| 10 | 0.633 | 0.951 | 0.094 | 5.2e-06 |
| 500 | 0.828 | 0.854 | 0.375 | 0.0114 |
| 1000 | 0.941 | 0.467 | 0.563 | 0.0709 |
| 1500 | 0.988 | 0.289 | 0.844 | 0.1382 |
| **2000** | **0.982** | **0.221** | **0.781** | 0.1548 |

**The worst collapse of the four arms.** `unique_fraction` 0.951 -> 0.221, against
`novelty`'s 0.516 and the retired `accuracy` arm's 0.535. Reward saturated near 0.99 by
step 1500 and stopped improving; diversity kept falling after that.

`zero_variance_group_fraction` peaked at 0.844: in 84% of groups all 16 samples scored
identically, and GRPO's advantage is ``(r - mean(r)) / std(r)`` within a group, so those
steps contributed NO gradient. Most of the second half of this run bought nothing.
`clip_fraction` held at 0.0 throughout and KL rose smoothly to 0.155 -- no training bug,
just the reward doing exactly what it was told.

**An intermediate reading was wrong and is worth recording.** At step 420 this arm sat at
`unique_fraction` 0.875 where `novelty` had already fallen to ~0.61, and that was read as
"holding, not collapsing." It was not holding; it collapsed later and further. Diversity
trajectories are not comparable at a fixed step, and a mid-run snapshot is not evidence
of a trend.

## Four arms, one pattern

| arm | its own metric | diversity cost |
|---|---|---|
| `accuracy` (retired) | reward-scored error 52.5 -> 31.4 K | `unique_fraction` 0.951 -> 0.535 |
| `validity` | PV 0.526 -> 0.901 | near-copy fraction 0.229 -> 0.585 |
| `novelty` | reward 0.578 -> 0.949 | `unique_fraction` 0.949 -> 0.516 |
| `synthesisability` | reward 0.633 -> 0.982 | `unique_fraction` 0.951 -> **0.221** |

Every verifiable reward tested optimised precisely what it measured and destroyed an axis
it did not measure. Verifiability buys a correct measurement, not a complete objective.
Four arms make this a finding rather than an anecdote.

**`composite` is now the study's most informative arm.** It is the only one carrying an
explicit diversity term. If it holds `unique_fraction` where four others did not, the
collapse is a reward-design failure and is fixable; if it collapses too, the problem is
deeper than any single reward's definition.

Trajectory: `results/grpo_synthesisability/trajectory_summary.json`.

## Round 2, arm `composite` — trained to completion 2026-08-29

Reward: equal-weighted sum of four verifiable structural terms — PV, novelty, SA-pass,
and **within-batch diversity**. No term reads a model prediction. This is the only arm
that prices diversity, and it is the controlled test of whether the collapse the other
four showed is a reward-design failure or something inherent to GRPO.

**It is a reward-design failure. Pricing diversity fixes it.**

| arm | reward | `unique_fraction` |
|---|---|---|
| `accuracy` | 0.214 -> 0.459 | 0.951 -> 0.535 |
| `novelty` | 0.578 -> 0.949 | 0.949 -> 0.516 |
| `synthesisability` | 0.633 -> 0.982 | 0.951 -> **0.221** |
| **`composite`** | 0.625 -> **0.948** | 0.949 -> **0.980** |

`composite` reached the same reward level as `novelty` (0.948 vs 0.949) while ending
with HIGHER diversity than it started. Same algorithm, same frozen generator, same 2000
steps, same group size; only the reward differs.

### Where the reward gain came from

| column | start -> end |
|---|---|
| `gated_fraction` | 0.328 -> **0.035** |
| `novelty_mean` | 0.858 -> 0.986 |
| `synthesisable_rate` | 0.939 -> 0.964 |
| `unique_fraction` | 0.949 -> 0.980 |
| `mean_length` | 72.6 -> 84.5 |

Almost all of the +0.323 came from the **gate**: the share of candidates failing the
SV/PV chemistry screen fell from 33% to 3.5%. Gated candidates receive a floor reward,
so rescuing 29% of the batch dominates the total. Novelty and SA improved modestly and
had little room -- they started at 0.86 and 0.94.

Length inflated 16% (72.6 -> 84.5), far less than `validity`'s 63 -> 131.

### The limitation: reward saturation, not collapse

`zero_variance_group_fraction` still reached **0.344**, and its mirror
`nonzero_advantage_fraction` fell 1.000 -> 0.656. By the end a third of groups produce
NO gradient, because GRPO's advantage is ``(r - mean(r)) / std(r)`` within a group and
those members score identically.

This is NOT diversity collapse -- `unique_fraction` is 0.980. It is the reward running
out of headroom: once nearly every candidate clears the gate, is novel and is
synthesisable, there is nothing left to discriminate on. `reward_std` falling
0.449 -> 0.192 says the same thing. The diversity term slows concentration; it does not
give the reward more to say.

### What is NOT yet known

Everything above is training-time diagnostics on rollout batches. Whether `composite` is
a BETTER GENERATOR -- PV rate, target-property rate and auditor Tg MAE under the frozen
1,500-sample protocol -- needs `compare_arms.py`, which has not run on it.

That distinction matters: `validity` also looked excellent on its own metric, then turned
out to have destroyed conditioning (TP 0.693 -> 0.047). `composite`'s reward never reads
a Tg prediction, so the honest expectation is that it HOLDS TP rather than improving it.

Trajectory: `results/grpo_composite/trajectory_summary.json`.

## Round 2, arm `constraint` — trained to completion 2026-08-30

Reward: a Tg-free multi-criterion CONJUNCTION -- a candidate must satisfy several
verifiable requirements at once. No diversity term.

It collapsed, and that is the point.

| | start -> end |
|---|---|
| `reward_mean` | 0.537 -> 0.955 |
| **`unique_fraction`** | 0.949 -> **0.539** |
| `gated_fraction` | 0.328 -> 0.027 |
| `novel_rate` | 0.858 -> 0.998 |
| `zero_variance_group_fraction` | 0.000 -> **0.656** |
| `mean_length` | 72.6 -> **62.6** |

Two thirds of groups end with no gradient at all, and diversity landed beside
`accuracy` (0.535) and `novelty` (0.516).

## Round 1 complete: five arms, one controlled variable

| arm | diversity term? | reward | `unique_fraction` |
|---|---|---|---|
| `accuracy` | no | 0.214 -> 0.459 | 0.951 -> 0.535 |
| `novelty` | no | 0.578 -> 0.949 | 0.949 -> 0.516 |
| `synthesisability` | no | 0.633 -> 0.982 | 0.951 -> 0.221 |
| `constraint` | no | 0.537 -> 0.955 | 0.949 -> 0.539 |
| **`composite`** | **yes** | 0.625 -> 0.948 | 0.949 -> **0.980** |

**Four arms without a diversity term collapsed; the one with it did not.**

`constraint` is the strongest single piece of evidence, because it is a multi-criterion
conjunction rather than a single objective. If breadth of requirements were what
protected `composite`, this arm should have held too. It did not, which isolates the
diversity term specifically rather than "more terms" generally.

Length is likewise reward-specific and not a universal artifact: `constraint` got
SHORTER (72.6 -> 62.6) where `composite` inflated to 84.5 and `validity` to 131.

The collapse is therefore not inherent to GRPO, and not inherent to verifiable rewards.
It is what a reward does when it does not price diversity, and one term fixes it.

Trajectory: `results/grpo_constraint/trajectory_summary.json`.

**All seven arms are trained. None has yet been scored by `compare_arms.py`
except `validity`, `control` and the retired `accuracy`.**
