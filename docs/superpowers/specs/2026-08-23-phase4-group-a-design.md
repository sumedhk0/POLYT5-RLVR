# Phase 4, Group A — fine-tuning improvements

**Status:** approved in discussion 2026-08-23, pending spec review
**Phase:** 4. Not part of the published polyT5 method — Phase 3's RLVR extension is
[`rlvr_plan.md`](../../rlvr_plan.md); this improves the models Phase 3 consumes.

## 1. Purpose

Phase 3 rewards a generator using a Tg predictor, and both come from the same
fine-tuning stage. Everything Phase 3 can claim is bounded by how good that stage is.
Group A improves it, using signal the repository already has and is discarding.

Two measurements from 2026-08-23 motivate this:

- **The 4-model ensemble adds nothing.** Honest MAE 28.82 K against a single model's
  28.67 K (`docs/instrument_audit.md`). Averaging four predictors buys no accuracy once
  the memorisation advantage is removed, and each member is handicapped by training on
  80% of the data.
- **σ is a weak signal.** `corr(σ, |error|)` is 0.15 on clean predictions — σ explains
  roughly 2% of error variance, so the confidence weight built on it is doing little.

A third fact reframes the problem entirely: **the paper's Tg dataset was 5,130
polymers; ours is 7,367.** We are not data-starved relative to the published work — we
have 44% more Tg labels. The gap between our RMSE 44.45 K and the paper's 40.82 K is
therefore unlikely to be label quantity. It is more plausibly the substitute
pretraining corpus, or how we use the labels we have.

**[CORRECTION 2026-08-23]** 7,367 is the *label count* in the CSV. The frozen
`splits.json` indexes **7,354 prepared examples** — 13 are lost to parse failures,
terminus rules, length limits and deduplication. 7,354 is the number that governs split
reuse, and the loader asserts it rather than trusting the CSV row count.

Group A attacks the second of those.

## 2. Scope

**In:** five changes to the fine-tuning stage, all cheap. One Tg fine-tune costs
6.7 minutes, so the whole ablation is roughly 3.5 hours.

**[CORRECTION 2026-08-24, whole-branch review finding 5]** This estimate implicitly
assumes every arm trains for the same number of OPTIMIZER STEPS as B0. Fixed epochs
alone does not give that: augmentation multiplies the batch count per epoch (~4x for
A3/A6) and the interleaved generation stream does too (~2x for A5, and -- before a
companion fix, finding 4 -- effectively ~4x more for A6 on top of that), so at a fixed
epoch count A3/A5/A6 would train for roughly 4x/2x/8x B0's steps, an 8-10 hour run, not
3.5. `scripts/run_group_a.py` now caps every arm's total optimizer steps at B0's
30-epoch budget (`_reference_step_budget`), which restores the original ~3.5-4 hour
order of magnitude — arms with more batches per epoch simply take fewer PASSES over
the data to reach the same step count. Per-arm wall-clock still varies a little
(A6 alternates two forward-pass shapes; A1/A6's regression path skips the decoder), so
treat 3.5-4 hours as the right order of magnitude, not an exact figure.

**Out, deliberately, each becoming its own spec:**

- **Group B — conformal prediction intervals.** Post-hoc calibration, needs a trained
  model first, minutes to run.
- **Group C — descriptor pretraining on the 92.3M corpus.** Days of GPU. Only worth
  paying for if Group A demonstrates that descriptor supervision helps at small scale.

Ordering is deliberate: Group A answers cheaply whether descriptors help at all,
before Group C spends days injecting the same information.

## 3. Architecture: one encoder, two heads

Today prediction and generation are **separate fine-tunes from the same pretrained
checkpoint** — siblings that share nothing after pretraining. Anything learned about
what makes a polymer high-Tg is discarded from the generator's perspective.

That separation is a consequence of the paper training them as distinct tasks, not a
property of T5, which is a multi-task architecture by design.

```
pretrained polyT5-medium encoder        (shared; receives gradients from both tasks)
├── regression head   → Tg as a scalar              (prediction)
├── descriptor heads  → the 100 LAMALAB features    (auxiliary, prediction side)
└── decoder           → PSELFIES                    (generation)
```

**[DECISION]** Shared encoder with two heads, rather than pure text-to-text
multi-task. The regression head is expected to predict better than text decoding, and
the asymmetry is acceptable: cycle consistency still works, since generation uses the
decoder and scoring uses the regression head.

Prediction and generation are mirror images — `polymer → Tg` and `Tg → polymer` — so
the encoder learning both directions is the point, not a side effect.

## 4. The five changes

Each is independently switchable. That is a requirement, not a convenience: without
it, a combined gain cannot be attributed.

### 4.1 Regression head

Tg is currently emitted as **text**, one character at a time, under beam search. The
model must learn digit place-value, can emit non-numeric output (tracked as
`non_numeric_rate`), and token cross-entropy gives it no sense that 236 and 237 are
adjacent while 236 and 936 are not.

```python
h      = encoder(input_ids, attention_mask).last_hidden_state
mask   = attention_mask.unsqueeze(-1).float()
pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)     # masked mean
tg     = linear(pooled).squeeze(-1)
```

- **Pooling:** masked mean over non-pad tokens. Not the first token — T5 has no `[CLS]`
  and never trained one.
- **Target scaling:** standardise using **train-split statistics only**, invert at
  inference. Raw Kelvin (μ≈417, σ≈113) makes the loss landscape awkward.
- **Loss:** Huber, not MSE. See 4.4 — some labels carry up to 145 K of experimental
  spread, and MSE would let those dominate the gradient.

### 4.2 Descriptor auxiliaries

`data/external/LAMALAB_CURATED_Tg.csv` already contains **100 precomputed descriptor
columns** — 30 backbone-level, 42 sidechain-level, 28 full-polymer. We load `PSMILES`
and `Exp_Tg(K)` and discard the rest.

```
L = L_Tg + λ · L_descriptors
```

The backbone/sidechain split is the physically meaningful one for Tg: backbone
rigidity raises it, sidechain length lowers it. Rather than hoping the model infers
this from 7,367 Tg values, it is supervised directly.

Descriptors are standardised per column on train statistics. Columns constant or
absent across the train split are dropped and the drop is logged, never silently
imputed.

### 4.3 Invariance augmentation

One polymer has many valid PSELFIES writings. Register entry A-05 verified that
canonicalisation collapses 14 different SMILES writings of 4 polymers to exactly one
PSELFIES each, deterministically across processes.

Train on N writings per polymer, same target. The model learns that Tg is a property
of the molecule, not the string, and effective training data multiplies at no
labelling cost.

**Augmentation must respect split boundaries** — every writing of a polymer belongs to
the same split as the original. A writing of a train polymer appearing in test would be
leakage indistinguishable from memorisation.

### 4.4 Label weighting by measurement reliability

Every row carries `num_of_points`, `std`, and `reliability`, all currently unused:

- 279 polymers have **multiple independent measurements**, up to 10
- their experimental spread: **median 5.6 K, mean 12.1 K, max 145 K**
- reliability flags: 7,088 `black`, 143 `gold`, 132 `yellow`, 4 `red`

That median 5.6 K is the **irreducible noise floor on the label itself**. Our 28.67 K
sits about 5× above it, so there is real headroom — but a 145 K label is close to
noise, and we currently train on it at full weight.

Weight each example by `1 / max(std, floor)` and drop `reliability == red`. Free
accuracy from data already on disk. The floor prevents a single-measurement polymer
(std = 0) from acquiring infinite weight.

**[CORRECTION 2026-08-23] The drop applies to train and val ONLY, never to test.**
The original wording did not say. Dropping rows from test would change the evaluation
set, and every Group A number is compared against a frozen 28.6733 K measured on the
full test split — a different denominator would make the comparison meaningless while
looking like an improvement. `n_red_in_test` is reported so the residual stays visible.

**[CORRECTION 2026-08-24, whole-branch review finding 6] The drop runs for EVERY arm,
not just A4.** `polyt5.data.multitask._drop_red_for_split` is called unconditionally in
`assemble_split`, so B0's in-harness rerun and A1/A2/A3/A5/A6 all drop the same ~2-3
train rows and ~1 val row per split that A4 does — only the *weighting* half of this
section is gated on the `reliability_weighting` switch. So A4 tests "weight by
1/max(std,floor)" alone, not the conjunction this section originally implied; the drop
is common corpus hygiene applied to every configuration, the way the red flag itself is
a property of the row rather than of any one arm. Impact is negligible (roughly 0.05%
of train rows per split) and was left as-is rather than gated, since gating would
perturb every arm's train/val pool for a change already covered by
`test_red_rows_leave_train_but_the_test_split_is_untouched`, which pins the
unconditional behaviour as deliberate.

### 4.5 Multi-task fine-tuning

Train prediction and generation together on the shared encoder, alternating batches.

**Cycle consistency is deliberately OPTIONAL and OFF by default.** Generating a polymer
for a target Tg and then scoring it with the model's own regression head is a signal the
model can satisfy by being *consistently wrong* — generate something odd, confidently
mispredict it as 500 K, incur zero loss. Anchored by 7,367 real labels it is a
legitimate semi-supervised regulariser, in the same way back-translation is anchored by
real parallel text. As a primary objective it would be circular. It ships behind a flag
and is measured as an ablation, never assumed.

## 5. Ablation protocol

Seven configurations, each on the **same five splits** the frozen baseline used, so
every number is directly comparable to 28.67 ± 0.76 K.

| id | configuration |
|---|---|
| B0 | baseline — current text head, single task (**rerun in-harness**, see below) |
| A1 | + regression head |
| A2 | + descriptor auxiliaries |
| A3 | + invariance augmentation |
| A4 | + label weighting |
| A5 | + multi-task shared encoder |
| A6 | all five combined |

Seven configurations × five splits × 6.7 min ≈ **3.5 hours**, valid once every arm's
optimizer-step budget is equalised to B0's -- see the finding-5 correction in Sec 2;
without it the real cost is closer to 8-10 hours because A3/A5/A6 otherwise train for
several times B0's step count on the same data.

**[CORRECTION 2026-08-23] B0 is rerun, not carried over.** The table originally
annotated it "already measured: 28.67 K" while this same section required all
configurations to run on the same five splits. Those contradict. Reusing the frozen
number would leave any harness difference — data loading, seeding, evaluation path —
silently attributed to whichever change was under test. The frozen artifact remains the
source of the 27.91 K *threshold*; B0's rerun is what the other six are compared against.

Individual ablations run **as well as** the combination, not instead of it. A combined
gain with no per-change attribution cannot tell you which idea to keep, and the runs are
cheap enough that there is no reason to guess.

## 6. Success criterion — fixed before any run

A change **helps** if its five-split mean MAE is below **27.91 K** — the baseline's
28.67 K minus one standard deviation (0.76 K). A change that lands inside the baseline's
own spread is recorded as **no effect**, not as a small win.

Reported for every configuration, whatever the outcome: MAE, RMSE, R², and
`non_numeric_rate` where applicable. A configuration that *hurts* is reported with the
same prominence as one that helps — the ablation's value is telling you which ideas were
wrong.

**Generation is evaluated separately and must not regress.** Any configuration touching
the shared encoder is checked against the frozen generation baseline (Arm B: PV 55.8%,
TP 58.8%). A prediction gain bought with a generation loss is a trade to surface, not a
success to report.

## 7. What this does not do

**It does not make Tg verifiable.** Every number here is a model predicting a property
of a molecule that, for generated candidates, nobody has synthesised. Group A improves
the *instrument*; it does not change the class of evidence the Tg arms can produce.
Phase 3's verifiable claims — PV rate, novelty, SR, diversity — do not depend on any of
this.

**It does not touch the 92.3M pretrained checkpoint**, the frozen baseline, or any
Phase 3 code. It produces new fine-tuned models alongside the existing ones. The
existing five predictors and their 28.67 K remain exactly as they are: they are what
Group A is measured against.

**It does not disturb Phase 3 training in flight.** The RLVR arms train the generator
against the *current* predictor. Swapping predictors mid-round would make arms
incomparable, so no configuration from Group A enters a reward path until a full round
is rerun on it deliberately.

## 8. Risks

**Multi-task can trade one task against the other.** The shared encoder serves two
objectives, and a gain on prediction may cost generation. §6's generation check exists
to catch this; if it fires, A5 is reported as a trade rather than an improvement.

**Invariance augmentation could reduce effective diversity.** N writings of the same
polymer are still one polymer. If N is large the model sees fewer distinct chemistries
per epoch. N is a config value, and A3 measures whether it helps rather than assuming.

**Descriptor auxiliaries could dominate the loss.** 100 auxiliary targets against one
Tg target risks λ swamping the objective we care about. λ is configurable and its
sensitivity is reported.

## 9. Reporting discipline

Three claims never merge: *"the paper reports"*, *"our reproduction obtains"*, *"our
extension obtains"*. The paper reports Tg prediction on 5,130 labels; our reproduction
obtains 28.67 ± 0.76 K on 7,367; Group A results are a **fourth** category — our
extension's improvements to our own reproduction — and must be labelled as such rather
than presented as closing the gap to the paper.

**No Group A configuration has been trained. This spec describes an experiment, not a
result.**
