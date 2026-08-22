# Group-contribution Tg oracle — design

**Status:** approved in discussion 2026-08-21, pending spec review
**Phase:** 3, round 2. Not part of the published polyT5 method — see [`rlvr_plan.md`](../../rlvr_plan.md).

## 1. Purpose

Every Tg number in the reward today comes from a model trained on our substitute
corpus. The four reward models and the split-4 auditor are independent random 80%
draws from **one** dataset, so they share roughly 80% of their training data. That
makes the auditor a weak instrument against corpus-wide error: it detects
ensemble-specific mistakes well and shared mistakes barely at all.

This oracle is a Tg estimator with **no training data at all**. It computes Tg from
molecular structure, so its errors come from a different source than any learned
model's. That independence — not its accuracy — is the thing being bought.

It serves the two-round study:

| Round | Tg reward | Oracle's role |
|---|---|---|
| 1 | 4-model learned ensemble | none during training; post-hoc scoring only |
| 2 | the same ensemble, plus an agreement term | shapes the gradient every step |

Both rounds are scored by the same untouched split-4 auditor. That is what makes
round 1 versus round 2 a controlled contrast rather than two unrelated runs.

## 2. The research question this makes answerable

> Does adding a **verifiable**, training-data-free component to a learned reward
> reduce reward hacking?

Today that is an assumption the apparatus quietly encodes. Round 1 versus round 2
turns it into a measurement with one variable changed.

## 3. Method: Bicerano, not Van Krevelen

**[DECISION]** Bicerano's connectivity-index correlation, not Van Krevelen's additive
group sums.

Van Krevelen computes `Tg = Yg / M` by summing group molar transition functions. It
is simpler and its tables are well documented, but an unrecognised group leaves `Yg`
undefined — a **hard** coverage failure.

Bicerano is a QSPR over topological connectivity indices (⁰χ, ¹χ, ⁰χᵛ, ¹χᵛ) plus
structural correction terms. The connectivity indices are purely topological and
compute for any parseable molecule; only the corrections are group-specific. Coverage
therefore degrades **gracefully**: a missing correction biases the estimate instead of
destroying it.

That property is decisive here. Our inputs are *generated* polymers, which sit near
the edge of any parameter table by construction — and, worse, sit furthest from it
exactly when the policy is doing something we want to catch. A method whose failure
mode is "no answer" would go blind precisely when it is most needed.

## 4. Interface

```
src/polyt5/chemistry/bicerano.py
```

A pure structure→property function: PSMILES in, `(tg_kelvin, coverage)` out. No
torch, no checkpoints, no training data, no network.

It belongs in `chemistry/` because that is what it is. `rewards/` consumes it. The
existing one-way dependency rule is unchanged and still enforced by
`tests/test_dependency_direction.py`: nothing outside `rl/` imports `rl/`, and
`chemistry/` imports neither `rewards/` nor `rl/`.

`coverage` is the fraction of applicable correction terms the molecule's structure
actually resolves — `1.0` when every relevant group is recognised, `0.0` when none is.

## 5. Reward integration: an agreement term

**[DECISION]** The oracle enters as an agreement factor. It does **not** become a
fifth ensemble member.

```
r_tg = closeness × confidence × agreement
```

`closeness` and `confidence` are **unchanged from round 1**, which is the whole point:

```
agreement = 1 / (1 + |Tg_oracle − Tg_ensemble| / δ₀)
```

**Why a check rather than a member.** A fifth member would change the reward's
*target*. Then "round 2 is more accurate" becomes ambiguous — did the policy improve,
or did the yardstick move? As a check, the target is identical across rounds, so both
reward-side and auditor-side accuracy compare cleanly and any difference is
attributable to the oracle's pressure.

It also contains the blast radius. Group contribution has a *different error
character* than a neural net: systematically biased near its table's edge, accurate in
the middle. Averaging it into the target treats those errors as interchangeable with
the models'. As a check, a miscalibrated oracle can lower trust but can never move the
target.

**Why soft rather than a threshold.** A hard cutoff is a cliff a policy learns to sit
just beneath. This is the same reasoning that chose a soft σ weight over a hard
uncertainty cutoff in the original design; the argument has not changed.

**δ₀ is measured, not chosen.** It is the observed mean `|Tg_oracle − Tg_ensemble|`
across the 7,367 labelled polymers, fixed once and frozen — exactly as σ₀ = 17 K came
from observed ensemble disagreement rather than a guess. A δ₀ picked by hand would be
a free parameter with no defence.

## 6. Coverage policy

**[DECISION]** An uncoverable candidate is penalised, never treated as neutral.

```
coverage == 0        →  agreement = pessimistic floor (maximal disagreement)
0 < coverage < 1     →  agreement scaled by coverage
coverage == 1        →  agreement as computed
```

**This is the single most important rule in this document.** Because `agreement ≤ 1`
always, a neutral default of `1.0` would make escaping the check strictly better than
passing it — an ascending gradient toward chemistry the oracle cannot read, which is
the chemistry we least want the policy exploring.

That is Ruling F verbatim. In the earlier build, `σ = 0.0` meant both "perfect
agreement" and "only one model could parse this", so a molecule three of four models
choked on scored **1.0000** against 0.8095 for full consensus. The confidence weight
had been inverted into a reward-hacking gradient. The fix scaled by
`n_contributing / n_total`; this scales by `coverage`, for the same reason.

Getting this wrong twice, in the same codebase, in the same quarter, would be a
process failure and not merely a bug.

## 7. Acceptance gate — pre-registered

**[DECISION]** All three thresholds are written to the frozen record **before** the
oracle is measured, alongside the existing success criterion.

| Quantity | Threshold | Rationale |
|---|---|---|
| Coverage on the labelled set | ≥ 70% | below this the discount is an indiscriminate tax, not a signal |
| MAE vs experimental Tg | ≤ 50 K | it need not beat the ensemble's 28.67 K — it is a check, not a target |
| Error correlation with the ensemble | Pearson r ≤ 0.5 | above this it largely re-measures what the ensemble already knows |

Measured against the 7,367 experimentally-labelled polymers in
`data/external/LAMALAB_CURATED_Tg.csv` (Tg 134–768 K, mean 417.1 K, sd 112.7 K).

**Independence is the criterion that matters most, and the one usually omitted.** For
a *check*, uncorrelated error beats low error. A 45 K oracle whose mistakes are
unrelated to the ensemble's is a better hacking detector than a 30 K oracle whose
mistakes track it — because correlated errors mean both estimates agree confidently on
exactly the molecules where both are wrong, and that agreement region is what a policy
optimises into.

**Failing the gate is a result, not an obstacle.** If the oracle misses any threshold
it does not enter round 2's reward. Round 2 then either runs without it or does not
run, and the miss is reported. Lowering a threshold after seeing the number would
destroy the reason for pre-registering it.

## 8. Round 1 use: read-only

Once built, the oracle post-hoc scores round 1's saved candidates as additional
`compare_arms` columns. No gradient involvement whatsoever.

This is how it earns trust: we observe its behaviour on real RLVR output — including
whatever degenerate chemistry the arms produce — **before** it is allowed to influence
anything. An oracle that behaves strangely on round-1 candidates is one we learn about
for free.

## 9. Testing

- Connectivity indices against hand-worked values for molecules computed by hand.
- The acceptance gate as an executable test against the labelled set, so a regression
  in the oracle fails the suite rather than silently degrading a reward.
- **The Ruling F regression, in its new home:** an uncoverable candidate must score
  strictly below a covered one at equal `|Tg_oracle − Tg_ensemble|`. This test exists
  to stop the same inversion recurring in the same codebase.
- δ₀ is read from the frozen record, never a literal — with a test that fails if the
  two disagree, matching how `ARM_METRIC` is bound to the pre-registration today.
- Coverage is reported per candidate and aggregated per run, never silently averaged
  away.

## 10. Known risk: sourcing the parameters

The correction-term table must come from published literature with a citation in the
module docstring. **Coefficients will not be invented.**

If a defensible table cannot be sourced, the oracle does not ship, round 2 runs
without it, and that is recorded as a finding. This repository has never fabricated
data — substitutes are documented as substitutes and gaps are documented as gaps — and
an oracle built on plausible-looking numbers would be worse than no oracle, because it
would carry the authority of a "computed" value while being a guess.

## 11. Reporting discipline

Three sentences that never merge: *"the paper reports"*, *"our reproduction obtains"*,
*"our RLVR extension obtains"*. The oracle is **ours** — the paper trains Tg
predictors but never releases one, never ensembles, never uses uncertainty, and never
uses a group-contribution method.

**No arm has been trained to completion. This design describes apparatus, not
results.**
