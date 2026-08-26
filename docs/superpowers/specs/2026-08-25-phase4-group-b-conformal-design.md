# Phase 4 Group B — conformal prediction intervals

**NOT part of the published polyT5 method.** Sahu et al. report point predictions only. This
is ours, and it is the fourth of the four proposed Tg ideas — the only one that was never
tested, because unlike the other three it requires no training.

## What it buys

`412 K` does not tell a user whether to run an experiment. `412 ± 31 K, 90% coverage` does.
For an endpoint this is worth more than A1's 1.5 K of MAE: it converts a number into a
decision. The guarantee is distribution-free and finite-sample — it holds for ANY underlying
model, including one as imperfect as ours, provided calibration and test data are exchangeable.

## Method: split conformal

Standard split conformal regression.

1. Score a **calibration** set the model never trained on. Residual `s_i = |y_i - ŷ_i|`.
2. For miscoverage `α`, take `q̂ = ⌈(n+1)(1−α)⌉ / n` empirical quantile of `{s_i}`.
3. Interval for a new input: `ŷ ± q̂`.

Coverage `P(y ∈ Ĉ(x)) ≥ 1 − α` holds by exchangeability alone. No distributional assumption,
no assumption that the model is good.

`⌈(n+1)(1−α)⌉` is not a rounding detail — it is the finite-sample correction that makes the
guarantee hold at small `n`. Using the plain `(1−α)` quantile under-covers, and that error is
invisible without a coverage test, which is why one is mandatory below.

## Where the data comes from

The shipped A1 (`artifacts/splits/full_corpus.json`) holds out **400** polymers that entered
neither training nor model selection. Split them **200 calibration / 200 coverage-validation**,
disjoint, fixed seed.

**This is the design's weak point and it is recorded, not hidden.** 200 calibration points make
`q̂` coarse — one residual moves the 90% quantile — and 200 validation points give an
empirical coverage estimate with roughly ±4 points of binomial standard error. The
guarantee is not affected: split conformal is valid at any `n`. The *estimate of whether we
achieved it* is what is imprecise. If the measured coverage CI turns out too wide to
distinguish 0.90 from 0.85, the honest response is to rerun A1 with a larger held-out block
(costing training data), not to quote the interval as if it were tight.

**Adaptive intervals are explicitly out of scope.** A single global `q̂` gives every polymer
the same width, which is wrong — some are far easier than others. Normalized conformal
(`s_i = |y−ŷ|/σ(x)`) would fix that, but `instrument_audit.md` already measured
`corr(σ, |error|) = 0.15` for our ensemble disagreement: σ explains ~2% of error variance,
so dividing by it would add noise, not signal. **A constant-width interval that is honestly
calibrated beats a variable-width one built on a σ we have measured to be uninformative.**

## Interface

```python
class ConformalRegressor:
    @classmethod
    def calibrate(cls, predictions, targets, alpha=0.1) -> ConformalRegressor
    def interval(self, prediction: float) -> tuple[float, float]
    def coverage(self, predictions, targets) -> CoverageReport
```

Wraps any predictor's outputs — it takes numbers, not a model, so it never imports torch and
stays testable without a GPU.

## Tests that must be able to fail

- `q̂` uses `⌈(n+1)(1−α)⌉/n`, not the plain quantile: assert the exact index on a hand-built
  residual set where the two differ.
- Coverage on synthetic exchangeable data lands within binomial tolerance of `1−α`.
- Calibrating and validating on the SAME data is refused — that is the circularity this
  whole apparatus exists to prevent.
- `alpha` outside `(0,1)` is refused.
- `n` too small for the requested `alpha` (`⌈(n+1)(1−α)⌉ > n`) is refused rather than
  silently returning an infinite interval.

## What this does NOT claim

Coverage is guaranteed over the LAMALAB Tg distribution the calibration set was drawn from.
A polymer unlike anything in those 7,354 gets no guarantee — exchangeability is exactly what
fails there. This is a calibration method, not an extrapolation detector.
