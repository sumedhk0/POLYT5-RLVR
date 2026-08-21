# GRPO/RLVR extension — design

**Status:** approved in discussion 2026-08-20, pending spec review
**Phase:** 3. Not part of the published polyT5 method — see `docs/rlvr_plan.md`.

## 1. Purpose

The supervised baseline is complete. It optimizes token likelihood, which is not the thing we care
about: validation cross-entropy is 0.398 while conditioning error is 47.3 K, and those numbers are
unconnected. This phase trains the generator against a **computed** score of what it actually produces.

The research question is *not* "can polyT5 generate polymers" — the paper settles that. It is:

> Can verifiable reinforcement learning improve property-conditioned polymer generation beyond a
> **properly tuned** supervised baseline?

"Properly tuned" is load-bearing. Arm B (sampling-tuned) already improved conditioning MAE by 7.1% over
the default sampling settings. Any RL gain must be measured against that, not against an untuned baseline.

## 2. Study design: the objective is the independent variable

Rather than choose one reward, we train four policies that differ **only** in their reward, then evaluate
all of them on the **same full metric set**. The output is a matrix, not a number.

| Arm | Reward optimizes |
|---|---|
| **C1 accuracy** | Closeness of predicted Tg to the requested Tg |
| **C2 validity** | Passing the full SV → TSD → DD → PV cascade |
| **C3 composite** | Weighted sum of accuracy, PV, and novelty |
| **C4 constraint** | Conjunction: Tg in window **and** SA below threshold **and** novel |

Compared against:

| Baseline | Definition | Measured |
|---|---|---|
| **Arm A** | supervised + default sampling (T=1.1, top_p=0.75) | MAE 50.9 K, PV 56.2%, TP 58.0% |
| **Arm B** | supervised + tuned sampling (T=0.7, top_p=0.95) | **MAE 47.3 K, PV 55.8%, TP 58.8%** |

The diagonal of the matrix answers "can RL optimize what we asked for?". **The off-diagonal is the
science**: if the validity-optimized policy destroys conditioning accuracy, we have measured reward
mis-specification directly instead of speculating about it.

C4 is deliberately reduced from the paper's full dielectric cascade, which needs Tg, Tm, Td, Eg, ε and
solubility models. We have **Tg only**; the other datasets are withheld and public substitutes are thin.
Tg + SA + novelty keeps the multi-criteria character using only quantities we can compute today.

## 3. Success criterion — fixed before any run

An arm succeeds if it **beats Arm B on the metric it optimized, and the improvement survives scoring by
the held-out auditor model.**

Auditor confirmation is what separates a real gain from predictor exploitation. Without it, an
"improvement" may exist only inside the reward model. This criterion is fixed now, in writing, so that no
metric can be selected after the fact.

## 4. Reward design

### 4.1 Validity is a gate, not a term

```
not RDKit-parseable                    → R = 0
not exactly two [At], each valency 1   → R = 0
```

A structure that is not a polymer earns nothing on any axis. This prevents accumulating partial credit
for chemistry that cannot exist.

### 4.2 The Tg term is confidence-weighted

$\hat{T_g}$ and $\sigma$ come from a **4-model ensemble**; a 5th model is the auditor and never touches
the reward.

```
closeness  = max(0, 1 − |T̂g − T_target| / 100)
confidence = 1 / (1 + σ / 17)          σ₀ = 17 K, the observed mean disagreement
r_tg       = closeness × confidence
```

**Rationale.** The Tg reward is a model prediction, and predictions on unfamiliar chemistry are guesses
wearing numbers. Our five models disagree by 16.7 K on average and up to 45.2 K, so exploitable regions
demonstrably exist. Two candidates both "on target" at |err| = 10 K earn 0.695 (σ=5 K) versus 0.247
(σ=45 K) — the policy is steered toward hits it can back.

**What this deliberately does not do:** it does not penalise novelty. Validity, novelty and diversity
rewards carry **no** similarity constraint, and a novel valid polymer receives full credit on those axes.
An earlier draft proposed a Tanimoto-to-training-set floor; that was wrong — it conflated "unfamiliar to
the predictor" with "bad", and it would have fought C3, which *rewards* novelty. Only the accuracy claim
is discounted, because only that claim depends on the predictor.

A soft weight is used rather than a hard cutoff: a threshold at σ=30 would score σ=5 and σ=17 identically
and then fall off a cliff the policy can learn to sit beneath.

### 4.3 Per-arm reward

```
C1  R = r_tg
C2  R = 1 if the candidate passes SV → TSD → DD → PV, else 0
C3  R = w_a · r_tg + w_v · pv_pass + w_n · novel          (weights in config, not code)
C4  R = 1 if (|T̂g − T_target| ≤ tol) and (SA ≤ sa_max) and novel, else 0
```

### 4.4 Drift is monitored, not prevented

Logged every epoch: distribution of max-Tanimoto to the labelled Tg set, and the **auditor gap**
(auditor prediction vs reward-ensemble prediction). If the policy leaves the predictor's support we
observe it and report it, rather than quietly forbidding it.

The unweighted reward is logged alongside the weighted one, so the gate's effect on the learning signal
is measurable rather than assumed.

## 5. Algorithm

GRPO — chosen because it needs **no value network**. For a 7.5M-parameter model on a 12 GB card, avoiding
a second network of comparable size is a material saving, and the critic is the usual source of PPO
instability.

**Group-relative advantage.** For prompt $x$, sample $y_1 \ldots y_G \sim \pi_{\theta_{old}}$:

$$A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G}) + \varepsilon}$$

This suits a noisy reward: our predictor has 28.7 K held-out MAE, and group-relative normalization
cancels per-prompt bias — a predictor that reads systematically high at 500 K shifts every group member
equally and leaves the advantage unchanged.

**Clipped surrogate, length-normalized:**

$$\mathcal{L}_{clip} = -\frac{1}{G}\sum_i \frac{1}{|y_i|}\sum_t \min\big(\rho_{i,t}A_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)A_i\big)$$

Length normalization is not optional here: sequences run 4–200 tokens and mean length already varies
56–100 across sampling configurations. Without $1/|y_i|$ the gradient is dominated by long sequences and
the policy learns to pad.

**KL anchor to the frozen supervised checkpoint**, using GRPO's k3 estimator (always positive, low
variance):

$$\mathcal{L} = \mathcal{L}_{clip} + \beta\Big(\frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1\Big)$$

Without it the policy finds degenerate strings that score well and forgets how to write polymers. β is
the primary knob; KL is logged every step.

**Log-probability source.** `generation.generate` already returns per-token log-probs under the
*unmodified* distribution — before temperature/top-p filtering. Those are the correct $\pi_{\theta_{old}}$
values for the ratio; the filtered distribution is only the proposal.

## 6. Architecture — synchronous

```
src/polyt5/rl/
    rollout.py           group sampling; returns sequences, per-token logprobs, masks
    rewards.py           assembles arm rewards from polyt5.rewards components
    advantages.py        group-relative advantage
    grpo.py              clipped surrogate + k3 KL
    reference_policy.py  frozen π_ref loader
    trainer.py           the synchronous loop

src/polyt5/rewards/
    validity.py          RDKit parse + PV terminus/valency gate
    tg.py                ensemble mean/std + confidence weighting
    novelty.py           ScalableNoveltyIndex lookup
    sa.py                synthetic accessibility
    constraints.py       C4 conjunction
    composite.py         weighted assembly
```

**Dependency direction is one-way:** `rl/ → {model, tokenization, chemistry, generation, evaluation,
inference}`. Nothing in the supervised codebase imports `rl/`. `training/checkpoint.py` is reused
unchanged.

**[CORRECTION] Torch is not, and was never meant to be, confined to `rl/` and `model/`.** `training/`,
`generation/`, `data/`, `inference/` and `evaluation/sweep.py` have imported `torch` since Phase 1/2 and
still do — that is correct, existing behaviour, not something Task 9 changed. The two contracts this
project actually verifies (Task 9, `tests/test_dependency_direction.py` and the pre-existing torch-purity
tests in `tests/test_chemistry.py` / `tests/test_rewards.py`) are narrower and both hold: (1) no
`transformers` dependency anywhere in the repo, and (2) nothing outside `rl/` imports `rl/` (checked by
AST, package by package). `polyt5.chemistry` and `polyt5.rewards` specifically are torch-free — that is a
real, narrower guarantee (reward workers must run CPU-only) — but that is not the same claim as "torch only
appears in `rl/` and `model/`", which is false and must not be repeated.

Rejected alternatives: **pipelined async reward workers** (~30–40% faster, but introduces a queue and
nondeterminism in a study whose whole point is trustworthy comparison — the torch-free chemistry layer
leaves this seam open for later); **an external RL library** (our model is not a HuggingFace model, the
repo avoids heavy frameworks, and GRPO's core is ~80 lines).

## 7. Measured cost

On the RTX 4080 Laptop, with the real checkpoints:

| Stage | Throughput | Per step (512 candidates) |
|---|---|---|
| Rollout, batch 128 | 125 cand/s | ~4.1 s |
| Reward, 1 model beam-4 | 296 cand/s | — |
| Reward, 4-model ensemble | 74 cand/s | ~6.9 s |
| Policy update | — | ~1–2 s |

**~12–15 s/step.** At G=16 × 32 prompts × 2,000 steps: **~7 h per arm, ~29 h for four.**

Rollout batches at 128 regardless of group size — measured, larger is worse (512 → 36 cand/s).

## 8. Training configuration

| Setting | Value | Rationale |
|---|---|---|
| Group size G | 16 | Enough for a stable group baseline; 512 candidates/step at 32 prompts |
| Prompts per step | 32 | |
| Target distribution | **uniform 250–600 K** | Forces genuine conditioning; the policy cannot score by emitting the data mode |
| Evaluation targets | **300 / 400 / 500 K, n=500** | Fixed to Arm B's protocol so comparison is apples-to-apples |
| ε (clip) | 0.2 | Standard |
| β (KL) | tunable, logged | Primary knob |
| Steps | ~2,000 | ≈1.02M candidates per arm |

## 9. Testing

Unit: advantage computation against hand-computed groups (including the degenerate all-equal-reward case,
which must yield zero advantage, not NaN); clipped surrogate against hand-worked ratios; k3 KL positivity;
each reward component in isolation; the validity gate zeroing every axis.

Integration: one full GRPO step on a tiny model end-to-end; determinism under a fixed seed; a reward-hacking
canary — a fake predictor with a known exploitable region, asserting the confidence gate suppresses it.

Guard: `rl/` must not be importable from any supervised module (subprocess test, matching the existing
torch-purity tests).

## 10. Entry gate

Before any RL run: **freeze and tag the baseline** — Arm A and Arm B numbers recorded, the supervised
checkpoints tagged immutable, `docs/baseline.md` final. Every RLVR result is reported against that frozen
record.

## 11. Reporting discipline

Three sentences that never merge: *"the paper reports"*, *"our reproduction obtains"*, *"our RLVR
extension obtains"*. The ensemble and auditor are **our** extension — the paper trains a Tg predictor but
never releases one, never ensembles, and never uses uncertainty. The five-split protocol that produced our
five models *is* the paper's; the use we put them to is not.
