# Phase 4 Group A results — Tg prediction ablation, seed 0

**Our extension obtains** the numbers below. They are improvements to OUR OWN reproduction,
not to the paper's Tg figures, which are a different quantity (5,130 withheld labels) on a
different dataset. Three claims never merge — see [`reproduction.md`](reproduction.md) §9.
polyT5 (Sahu et al., npj AI 2026) contains none of this.

Produced by `scripts/run_group_a.py` on 2026-08-25 over the frozen five splits
(`results/tg_prediction_5splits_medium92m/splits.json`). Seven arms warm-started from the
same 92.3M checkpoint, each held to an identical 9,930 optimizer steps. **One seed (0).**

## The harness reproduces the baseline

B0 is the in-harness rerun of the frozen configuration, and it is the check that makes every
other row readable. Outlier-corrected, it lands at **28.67 K** against the frozen **28.6733 K**
— a three-decimal match. Had it missed, no arm's delta would mean anything.

## The matrix

| arm | switch | MAE (K) | Δ vs 28.6733 | per-split s0…s4 | verdict |
|---|---|---|---|---|---|
| B0 | — (control) | 29.18 / **28.67** corrected | +0.51 / 0.00 | 27.85 32.51 28.56 29.83 27.17 | no effect |
| **A1** | **regression head** | **27.17** | **−1.50** | 25.30 27.72 27.48 29.15 26.20 | **helps** |
| A2 | descriptor auxiliaries | 29.19 | +0.52 | 30.75 30.04 28.58 29.44 27.17 | no effect |
| A3 | invariance augmentation | 29.38 | +0.70 | 29.38 30.50 29.10 29.89 28.03 | no effect |
| A4 | reliability weighting | 28.79 | +0.11 | 28.35 29.80 28.85 29.54 27.40 | no effect |
| A5 | multi-task shared encoder | 30.56 | +1.89 | 30.13 30.72 30.36 31.76 29.84 | hurts |
| A6 | all five combined | 29.71 | +1.04 | 28.99 31.64 30.01 29.77 28.13 | hurts |

Pre-registered threshold for `helps`: 27.9142 K, one baseline standard deviation below
28.6733. **A1 is the only arm that clears it**, and it beats B0 on all five splits
individually — not an average rescued by one lucky split.

## What A1 actually fixed

Across the whole run there was exactly **one** catastrophic prediction: B0 split 1 emitted
**4293.10 K** where the truth was 481.15 K. One extra digit. That single token took split 1's
r² from 0.828 to 0.051 and its RMSE from 46.75 to 109.83.

| B0 split 1 | MAE | RMSE | r² |
|---|---|---|---|
| all 1,471 | 32.51 | 109.83 | 0.051 |
| minus that one point | 29.94 | 46.75 | 0.828 |
| frozen split_1 | 29.10 | 45.90 | 0.834 |

Not a harness bug: the frozen run used the same protocol and simply did not roll that slip.
It is the free-form-text regression failure mode, and A1 removes it by construction — a float
head cannot emit an extra digit. **A1 produced zero outliers.** Its RMSE is 41.32 against
B0's 56.33 for the same reason.

This also means B0's RMSE and r² are hostage to single decodes and should carry no weight in
any comparison. MAE is the pre-registered metric and is robust: the outlier moved one split by
+2.6 K, so ~+0.5 K on a five-split mean.

## Generation: A5 and A6 do not help the generator

A5 and A6 touch the shared encoder, so they are the only arms that could change the generator.
`run_group_a.py` records Tg prediction only, so generation was scored separately with
`scripts/check_group_a_generation.py` at the frozen 300/400/500 protocol, 500 samples per
target, TP judged by the four **external** split-0…3 predictors (the arm's own head is refused
as a circular instrument).

**The control is what makes this readable.** The frozen generator's own weights, loaded into
the same `PolyT5MultiTask` wrapper, read **PV 0.6100 / TP 0.6164** — where
`sampling_sweep.py` recorded **0.5580 / 0.5878** for that same checkpoint. The checker sits
+0.052 PV and +0.029 TP above the sweep. Comparing an arm to the sweep row is therefore a
cross-script comparison that manufactures a gain; arms are compared to the control.

| arm | PV | vs control | TP | vs control |
|---|---|---|---|---|
| control (frozen generator) | 0.6100 | — | 0.6164 | — |
| A5 | 0.6153 | +0.005 | 0.4561 | **−0.160** |
| A6 | 0.5893 | −0.021 | 0.4457 | **−0.171** |

**PV flat, TP down ~16–17 points.** Not a validity-for-conditioning trade — the shared encoder
costs conditioning accuracy and buys nothing. With prediction also down 1.04–1.89 K, A5 and A6
are worse on every axis measured.

Before the control existed this looked like a PV *gain* of +0.057/+0.031. It was the
instrument. Recorded here because the uncontrolled reading is the one a reader would
reconstruct from the raw check files.

## Consequences

- **A1 is the prediction-side result.** It is not in the RL path: Tg was dropped from every
  reward still being trained, so a better Tg model sharpens the auditor, not the rewards.
- **The generator does not change.** A5/A6 regress it, so the Phase 3 arms stay comparable to
  the existing baseline and no RLVR arm needs re-running.
- **Three of the four proposed ideas placed:** regression head wins, descriptor auxiliaries and
  invariance augmentation do nothing. Conformal intervals (Group B) are untested here and need
  no training — they wrap an already-trained model.

## Caveats

**One seed.** `seed: 0` throughout. The across-seed unanimity criterion is not exercised, and
A1's −1.50 K is not protected against seed-to-seed variance. It wins on 5/5 splits, which is
evidence, but splits are not seeds.

**Generation is split 0 only.** A5/A6 generation used each arm's split-0 checkpoint, one
sampling seed. The TP gap is large and consistent across all three targets, but it is one draw.

**A2's switch did fire.** 99 of the 100 descriptor columns survived into training in every A2
and A6 split (one dropped as degenerate), so A2's "no effect" is a real null about descriptor
auxiliaries at `descriptor_lambda: 0.1`, not a harness no-op. The `lambda` sensitivity was
never swept, so the null is at that one weight.

## Reproducing

```bash
python scripts/run_group_a.py
python scripts/check_group_a_generation.py --checkpoint <arm>/checkpoints/best.pt --arm A5 \
    --n-samples 500 --target-property 300 \
    --predictor-checkpoint results/tg_prediction_5splits_medium92m/split_0/checkpoints/best.pt
```

Outputs `results/group_a/ablation_matrix.json` and
`results/group_a/generation_check_3target.json`.
