# Phase 4 Group A results — Tg prediction ablation, seeds 0 and 1

**Our extension obtains** the numbers below. They are improvements to OUR OWN reproduction,
not to the paper's Tg figures, which are a different quantity (5,130 withheld labels) on a
different dataset. Three claims never merge — see [`reproduction.md`](reproduction.md) §9.
polyT5 (Sahu et al., npj AI 2026) contains none of this.

Produced by `scripts/run_group_a.py` on 2026-08-25 over the frozen five splits
(`results/tg_prediction_5splits_medium92m/splits.json`). Seven arms warm-started from the
same 92.3M checkpoint, each held to an identical 9,930 optimizer steps. **Two seeds (0, 1)**,
70 training runs. Splits stay frozen across seeds, so only initialisation and batch order
change.

## The harness reproduces the baseline

B0 is the in-harness rerun of the frozen configuration, and it is the check that makes every
other row readable. Outlier-corrected, it lands at **28.67 K** against the frozen **28.6733 K**
— a three-decimal match. Had it missed, no arm's delta would mean anything.

## The matrix

MAE in K, outlier-corrected, averaged over the five frozen splits.

| arm | switch | seed 0 | seed 1 | mean | spread | clears 27.9142 on both? | splits beating own B0 |
|---|---|---|---|---|---|---|---|
| B0 | — (control) | 28.67 | 28.53 | 28.60 | 0.14 | no | 0/10 |
| **A1** | **regression head** | **27.17** | **27.05** | **27.11** | **0.12** | **yes** | **10/10** |
| A2 | descriptor auxiliaries | 29.19 | 28.29 | 28.74 | 0.90 | no | 5/10 |
| A3 | invariance augmentation | 29.38 | 29.99 | 29.68 | 0.61 | no | 0/10 |
| A4 | reliability weighting | 28.79 | 28.35 | 28.57 | 0.44 | no | 4/10 |
| A5 | multi-task shared encoder | 30.56 | 29.63 | 30.10 | 0.94 | no | 1/10 |
| A6 | all five combined | 29.71 | 30.74 | 30.23 | 1.03 | no | 2/10 |

## A1 meets the pre-registered criterion

The criterion was across-seed unanimity plus a minimum margin: the arm must sit below
27.9142 K — one baseline standard deviation under the frozen 28.6733 K — on **every** seed.
With one seed that criterion was vacuous, since a single run cannot disagree with itself.

**A1 clears it on both seeds** (27.17 and 27.05), with a 0.12 K across-seed spread, and beats
its own B0 on **all ten** arm/split pairs. No other arm clears it on either seed.

**No other arm survives.** A3, A5 and A6 land above baseline on both seeds. A2 and A4 are
nulls whose across-seed spread — 0.90 K and 0.44 K — is as large as or larger than the effect
they would need to show, and each beats B0 on roughly half its splits (5/10 and 4/10, a coin
flip). A2's sign even reverses between seeds: +0.52 K against the frozen baseline on seed 0,
−0.38 K on seed 1. **A switch whose sign flips between seeds has not been measured.**

That is a weaker and more honest claim than the seed-0 write-up's "no effect". At n=5 splits
per seed these two switches are indistinguishable from noise; they are not shown to be inert.

## The baseline held, and so did the environment

B0 landed at 28.67 and 28.53 against the frozen 28.6733 K. Seed 1 also ran under a different
interpreter — Windows Smart App Control began blocking the project venv's `python.exe`
mid-study, so seed 1 used the uv-managed CPython 3.12.13 that the venv was built from, with
the same site-packages and the same torch 2.9.0+cu129. B0 moving 0.14 K across that change
confirms it is inert, which is the only reason seed 1's numbers may sit in the same table as
seed 0's.

## What A1 actually fixed

Across both seeds there was exactly **one** catastrophic prediction: seed 0's B0 split 1
emitted **4293.10 K** where the truth was 481.15 K. One extra digit. That single token took
split 1's r² from 0.828 to 0.051 and its RMSE from 46.75 to 109.83. Seed 1 produced none, in
any arm.

| B0 split 1 | MAE | RMSE | r² |
|---|---|---|---|
| all 1,471 | 32.51 | 109.83 | 0.051 |
| minus that one point | 29.94 | 46.75 | 0.828 |
| frozen split_1 | 29.10 | 45.90 | 0.834 |

Not a harness bug: the frozen run used the same protocol and simply did not roll that slip.
It is the free-form-text regression failure mode, and A1 cannot exhibit it **by construction** —
a float head has no digit to add. That mechanism is the claim; the empirical support for it is
thin, since the event fired once in ten B0 runs, so A1's zero across ten runs is what a rate
that low predicts anyway. The argument rests on the mechanism, not on the count.

The RMSE gap is the same story with more data behind it: A1 averages 40.69–41.32 against B0's
43.46–56.33, and the seed-0 spread in that B0 figure is one decode.

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
| seed 0 A5 | 0.6153 | +0.005 | 0.4561 | **−0.160** |
| seed 0 A6 | 0.5893 | −0.021 | 0.4457 | **−0.171** |
| seed 1 A5 | 0.5707 | −0.039 | 0.4346 | **−0.182** |
| seed 1 A6 | 0.5807 | −0.029 | 0.4856 | **−0.131** |

Four measurements across two seeds, all four below the control on both axes. TP is down
13–18 points every time. PV's single positive — seed 0's A5 at +0.005 — is contradicted by
seed 1's −0.039, so it was noise.

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

**Two seeds, not more.** A1 clears the criterion as written, on 0 and 1. Two is the minimum
that makes unanimity meaningful, not a comfortable margin — a third seed would be the first
one able to break it.

**The nulls are underpowered, not disproven.** A2 and A4 move 0.90 K and 0.44 K between seeds
against effects of similar size. Calling them "no effect" overstates what five splits per seed
can resolve.

**Generation is split 0 only.** A5/A6 generation used each arm's split-0 checkpoint on both
seeds, one sampling seed each. The TP gap is large and consistent across all three targets and
both training seeds, but the split-0 restriction stands.

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
