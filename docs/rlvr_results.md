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

## Round 1 scored: every Tg-free reward destroys conditioning

`compare_arms.py` over all seven arms, frozen protocol (targets 300/400/500 K, 500
samples each, tolerance 50 K, auditor `split4` held out of every reward path).

**All six scored arms cleared their own pre-registered criterion.** Every one also
wrecked the metric it did not price.

| arm | its metric | TP rate | auditor MAE | mean Tg produced |
|---|---|---|---|---|
| baseline (Arm B) | — | **0.693** | **36.6 K** | 376 K |
| `accuracy` (retired) | Tg closeness | **0.869** | **21.8 K** | 437 K |
| `synthesisability` | SA pass 0.998 | 0.489 | 67.8 K | 282 K |
| `composite` | composite 0.916 | 0.112 | 149.8 K | 251 K |
| `novelty` | novelty 0.969 | 0.072 | 152.6 K | 273 K |
| `constraint` | constraint 0.985 | 0.060 | 153.1 K | 256 K |
| `validity` | PV 0.901 | 0.047 | 150.4 K | 248 K |
| `control` | none | 0.693 | 36.3 K | 375 K |

Targets averaged 400 K. Every Tg-free arm produced 248-282 K regardless of what was
asked. `control` stayed flat on everything, so this is the rewards, not GRPO.

The only arm that IMPROVED conditioning optimised a Tg term -- the arm retired as
unverifiable.

### It is attenuation, not indifference

An aggregate TP cannot distinguish "distribution shifted" from "target ignored". Sampling
each target separately (`scripts/conditioning_frontier.py`) separates them:

| model | Tg @300 | @400 | @500 | slope |
|---|---|---|---|---|
| baseline | 309.6 | 375.8 | 490.4 | **0.904** |
| `composite` | 231.5 | 252.1 | 285.2 | **0.268** |
| `validity` | 214.8 | 246.8 | 282.5 | 0.339 |

Slope 1.0 is perfect tracking, 0.0 total indifference. The arms still respond; they
respond weakly and from far too low a baseline.

### Two failure modes, on different schedules

Scoring `composite`'s intermediate checkpoints:

| checkpoint | PV | slope | Tg@400 | TP |
|---|---|---|---|---|
| baseline | 0.658 | 0.896 | 375.2 | 0.740 |
| step 200 | 0.736 | 0.891 | 348.9 | 0.671 |
| step 400 | **0.802** | **0.914** | 324.4 | 0.551 |
| step 800 | 0.860 | 0.842 | 297.0 | 0.296 |
| step 1200 | 0.862 | 0.535 | 283.1 | 0.191 |
| step 2000 | 0.933 | 0.274 | 250.2 | 0.144 |

**Offset drift starts immediately; slope collapse only begins around step 800.** At step
400 the slope is 0.914 -- better than baseline -- while PV has already gained 14 points.
The model still knows what the target means; it is simply 51 K low.

### Calibration recovers some of it, and shows why the rest cannot be recovered

An affine distortion with an intact slope should be invertible by prompting, so
`scripts/conditioning_calibration.py` fits the map on held-out targets (280/350/420/470)
and tests on the protocol's 300/400/500. It helps and does not rescue:

    pooled TP  raw 0.555   calibrated 0.608   baseline 0.740
    pooled PV  calibrated 0.693              baseline 0.658

Three reasons it falls short, and the third is the finding:

1. The response is not affine. Fitting on 280-470 gives slope 0.687; measuring across
   300-500 gives 0.914. A linear inverse overshoots the middle -- prompting 499.8 for
   400 K returned 461 K.
2. The fit used a separate target set from the evaluation, so this is an honest test
   rather than a fit to its own scoreboard.
3. **Asking for higher Tg costs validity.** At target 500, prompting the raw 500 gave
   PV 0.685; prompting the calibrated 645 gave PV **0.510**, below baseline.

Point 3 is direct evidence that the structural rewards are chemically ANTI-CORRELATED
with high Tg: high-Tg polymers are rigid aromatics, harder to emit validly and worse on
synthetic accessibility. The rewards do not merely neglect Tg, they pull against it. The
frontier is chemical, not an artifact of training length.

### What this leaves

- **Early stopping is a genuine, free Pareto gain.** Step 200 trades TP 0.740 -> 0.671
  for PV 0.658 -> 0.736, and the checkpoint already exists.
- **No operating point keeps baseline TP and the full +27-point PV gain.**
- **`kl_coef` is untested.** At 0.02 it is the only force preserving conditioning, and KL
  still drifted to 0.073. It is the one knob that buys retention at no cost in
  verifiability -- swept next, see `configs/rl/composite_kl*.yaml`.
- **A verifiable Tg term remains the principled fix.** Bicerano group contribution is a
  published empirical correlation rather than a learned model, so it stays inside the
  study's verifiability tier -- see
  `docs/superpowers/specs/2026-08-21-bicerano-oracle-design.md`, still unbuilt.

## Round 2 verdict: `kl_coef` is not the lever

`composite_kl05` trained to 2000 steps at `kl_coef: 0.05`, against round 1's 0.02.
Everything else identical, verified by a test that the configs differ in exactly
`{experiment_name, train.replay_coef}` — one variable.

Measured with `scripts/conditioning_frontier.py`, not inferred from the KL column:

| model | PV | slope | Tg@400 | TP |
|---|---|---|---|---|
| baseline | 0.665 | **0.904** | 375.8 | **0.738** |
| `composite` (kl 0.02) | 0.915 | 0.268 | 252.1 | 0.138 |
| `composite_kl05` (kl 0.05) | 0.918 | **0.295** | 257.6 | **0.192** |

**A 2.5x stronger KL penalty recovered under 5% of the damage** — +0.027 slope against a
0.636 gap, +0.054 TP against a 0.600 gap. The pre-registered bar for a usable arm is
slope >= 0.70; this reaches 0.295.

The KL column said the same thing more quietly: 0.0627 at step 2000 against `composite`'s
0.0731, only 14% tighter for a 2.5x penalty. The policy pays the KL cost and drifts anyway.

**`kl1` (0.10) and `kl2` (0.20) were not run.** Extrapolating this curve, neither is
likely to approach slope 0.70, and establishing that would cost ~40 GPU-hours.
`composite_kl1` stopped at step 140 and is resumable if the question is ever worth
reopening.

### Why it fails, and what that implies

KL anchors the WHOLE distribution. It resists the structural gains the reward is buying
just as much as it resists the conditioning loss, so raising it either does nothing (as
here) or would cost the PV gain too. It cannot separate the two.

The conditioning skill is not diffuse: it lives in 6,619 supervised `(target, polymer)`
pairs, and nothing in the GRPO objective mentions it. That is a forgetting problem with a
targeted fix — keep training on those pairs — which is round 3
(`docs/superpowers/specs/2026-08-31-supervised-replay-design.md`).

## Round 3 verdict: supervised replay preserves conditioning AND the structural gain

`composite_replay05` trained to 2000 steps: the same `composite` reward, `kl_coef` back
at round 1's 0.02, plus `loss += 0.5 * cross_entropy(batch of the 6,619 supervised
pairs)`. No Tg predictor is read, so verifiability is untouched.

| model | PV | slope | Tg@300 | Tg@400 | Tg@500 | TP |
|---|---|---|---|---|---|---|
| baseline | 0.665 | 0.904 | 309.6 | 375.8 | 490.4 | **0.738** |
| `composite` (kl 0.02) | **0.915** | 0.268 | 231.5 | 252.1 | 285.2 | 0.138 |
| `composite_kl05` (kl 0.05) | 0.918 | 0.295 | 241.2 | 257.6 | 300.2 | 0.192 |
| **`composite_replay05`** | **0.790** | **0.973** | 286.8 | 322.8 | 481.4 | **0.582** |

**Both pre-registered conditions cleared: slope 0.973 >= 0.70, PV 0.790 >= 0.75.**

The slope is ABOVE the baseline's 0.904 — the conditioning response is not merely
preserved but slightly sharper than the model started with. Against `composite`'s 0.268,
replay recovered **99% of the lost slope**, where a 2.5x KL penalty recovered under 5%.

And it kept a real structural gain: PV 0.790 against the baseline's 0.665, +12.5 points.

### What the three rounds establish together

1. Every Tg-free verifiable reward destroys conditioning (round 1, five arms, flat control).
2. Anchoring harder with KL does not fix it — under 5% recovered for a 2.5x penalty (round 2).
3. It is catastrophic forgetting, and replaying the supervised data fixes it (round 3).

The mechanism is the one the round-3 spec predicted. KL restrains the WHOLE distribution,
so the reward pays the penalty and drifts anyway; replay restrains the ONE skill being
lost and leaves the rest free to improve.

### What is NOT fixed

**TP is 0.582 against the baseline's 0.738.** Far ahead of every other arm, and not full
recovery. The residual damage is OFFSET, not slope: at target 400 the model produces
322.8 K, about 53 K low, while tracking the target almost perfectly. Round 1 showed an
offset with an intact slope is partly correctable by prompt calibration, and with slope
0.973 that should work far better here than it did on `composite` (slope 0.268).

**PV 0.790 is below `composite`'s 0.915.** Replay costs some structural gain. A real
trade, and a far better one than any alternative measured.

**One seed, one coefficient.** `replay01` (0.1) and `replay20` (2.0) are untested, so 0.5
is shown to be SUFFICIENT, not optimal.

Trajectory: `results/grpo_composite_replay05/`. Reproduce the table with
`scripts/conditioning_frontier_rounds.py`.

**Rounds 1-3 complete.**
