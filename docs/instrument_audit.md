# Tg instrument audit

How much can the Phase 3 reward signal actually be trusted? Measured, not assumed.

Produced by `scripts/audit_ensemble_leakage.py`. Phase 3 — **not** part of the published
polyT5 method. Every number here is `[OURS]`.

## Why this exists

Phase 3 rewards a generator using a Tg estimate from four fine-tuned predictors
(splits 0–3). Every claim the Tg arms can support is bounded by how good that estimate
is. The reported 5-split figure — MAE 28.67 ± 0.76 K — is honest **for a single model**:
the paper's protocol is five independent random 80/20 splits, and `train ∪ val ∪ test`
covers every index, so each model's test set is genuinely unseen data.

The **ensemble** was never measured that way. For a given polymer, typically only one of
the four never trained on it, so the 4-model mean is largely a memory check.

## Method

For each polymer, a member is **clean** when the polymer was in neither its train nor its
val portion (val counts as seen — it drove checkpoint selection). Two predictions are
computed and compared against experimental Tg:

- **contaminated** — mean over all four reward members, which is what the reward uses
- **honest** — mean over only the clean members

Split 4 is the held-out auditor and is deliberately never scored here.

Clean-member distribution over 7,354 polymers: 3,021 have none, 3,005 have one, 1,123
have two, 187 have three, 18 have four.

## Results

| | ≥1 clean (n=4,333) | ≥2 clean (n=1,328) |
|---|---|---|
| Contaminated MAE | 20.38 K | 23.65 K |
| **Honest MAE** | **28.82 K** | **28.66 K** |
| Optimism | 8.43 K (41%) | 5.01 K (21%) |
| Contaminated σ | 16.40 K | 17.03 K |
| Honest σ | 3.65 K¹ | 11.92 K |
| corr(σ, \|error\|) contaminated | 0.30 | 0.31 |
| **corr(σ, \|error\|) honest** | **0.07** | **0.15** |

¹ Degenerate: most polymers in this bucket have exactly one clean member, and the spread
over a single value is 0. Only the ≥2 column is meaningful for σ.

## What this establishes

**1. The ensemble adds nothing over a single model.** Honest ensemble MAE is 28.82 K
against the single-model 28.67 K. Averaging four predictors, once the memorisation
advantage is removed, buys no accuracy. Each member is also handicapped by training on
80% of the data; one model trained on all of it would likely do better than this
ensemble does.

**2. The ensemble looks 41% better than it is.** 20.38 K contaminated versus 28.82 K
honest. This is not a defect in the predictors — it is what happens when models trained
on five overlapping subsets are later reused as an ensemble over the same data. Any
evaluation of the reward signal against this corpus must use the honest figure.

**3. σ is a weak error signal, and half of its apparent strength is contamination.**
The reward's confidence weight is `1/(1 + σ/σ₀)`, which presumes σ predicts error. It
does, but barely: r = 0.15 on clean predictions, so σ explains roughly 2% of error
variance. The contaminated r = 0.30 is inflated by a mechanism unrelated to uncertainty —
when some members memorised a polymer and others did not, the spread is wide because of
*split membership*, not because the molecule is hard.

An earlier hypothesis in this project's discussion was that σ might *anti*-correlate with
error, making the confidence weight actively harmful. **That is wrong.** The correlation
is positive at every threshold measured. The weight is weakly justified, not harmful, and
the spec's description of it steering the policy "toward hits it can back" overstates what
a 2%-of-variance signal can do.

## Consequences

- **The global-holdout retrain is not warranted.** A trigger was fixed before these
  numbers were seen: revisit if honest ensemble MAE exceeded ~40 K. It is 28.82 K. The
  instrument is weaker than advertised but not broken, and the study's headline claims —
  PV pass rate, novelty, SR, diversity, SA — are verifiable and do not use it at all.
- **Report the honest figure wherever the reward's accuracy is characterised.** The 5-split
  28.67 K remains the correct number for comparison with the paper, because it is the
  paper's protocol. The two answer different questions and must never be merged.
- **The confidence weight is open.** Keeping it, dropping it, or replacing it with a
  genuine uncertainty estimate (MC dropout, or a same-data seed ensemble) are all
  defensible. Changing it mid-round would invalidate the round in flight, so it is
  recorded here rather than acted on.

## Limitation

Only 18 polymers have all four members clean, so the honest MAE of the *full* 4-member
ensemble cannot be measured this way — the figures above are for 1- to 3-member clean
subsets. Measuring it directly would require retraining with a global holdout.

**No arm has been trained to completion. This document reports on the measuring
instrument, not on any RLVR result.**
