# Round 3: supervised replay against conditioning collapse

**NOT part of the published polyT5 method.** Ours, like everything in Phase 3.

## The problem this addresses

Round 1 scored every Tg-free verifiable reward as destroying conditioning. Against the
frozen baseline's TP of 0.693:

| arm | TP | auditor MAE | mean Tg produced |
|---|---|---|---|
| baseline (Arm B) | 0.693 | 36.6 K | 376 K |
| `composite` | 0.112 | 149.8 K | 251 K |
| `constraint` | 0.060 | 153.1 K | 256 K |
| `validity` | 0.047 | 150.4 K | 248 K |

Targets averaged 400 K; every Tg-free arm produced 248–282 K whatever was asked.
`control` stayed flat, so this is the rewards and not GRPO.

`scripts/conditioning_frontier.py` showed the damage has two modes on different
schedules: offset drift begins immediately, slope collapse only around step 800. At
`composite` step 400 the slope is still 0.914 (baseline 0.896) while PV has already
gained 14 points — the skill is intact there and gone by step 2000 (slope 0.274).

## Why the skill is fragile, and why replay is the right shape of fix

The conditioning skill lives entirely in **6,619 supervised `(target, polymer)` pairs**
(`data/processed/tg/generation/train.jsonl`). The 92.3M-polymer pretraining never saw a
Tg. GRPO then runs 2,000 steps of pure policy gradient — `src/polyt5/rl/trainer.py` has
no supervised-loss path at all — against an objective that never mentions the target.
The newest and thinnest skill in the model is overwritten by an objective indifferent to
it. That is catastrophic forgetting, and it is exactly what the slope collapse measures.

Round 2's `kl_coef` sweep attacks this with a blunt instrument. KL says *"do not move
from the reference"*, which also resists the structural gains the reward is buying. Early
evidence says it is a weak lever: at matched step 1200, `kl_coef` 0.05 gave KL 0.0398
against 0.02's 0.0435 — 8% tighter for a 2.5× stronger penalty.

Replay says something narrower: *"do not forget this specific mapping."* It pins the
conditioning skill while leaving the policy free to improve PV, novelty, SA and
diversity. That asymmetry is precisely what the round-1 frontier says is needed.

Precedent: InstructGPT's PPO-ptx mixes pretraining gradients into RLHF to counter the
"alignment tax" — capability loss on axes the reward does not measure. Same shape.

**Replay reads no Tg predictor.** It trains on measured labels, so verifiability is
untouched. This is not a Tg reward through the back door: no reward term changes, the
arm's reward stays exactly `composite`'s four structural terms.

## Design

One extra loss term in `GRPOTrainer.step`:

```
loss = grpo_loss  +  replay_coef * cross_entropy(supervised batch)
```

### Interface

`GRPOTrainerConfig` gains two fields, defaulting to today's behaviour:

- `replay_coef: float = 0.0` — weight on the supervised term. **0.0 reproduces round 1
  exactly**, which is the sanity check that the change is inert when disabled.
- `replay_batch_size: int = 16` — supervised pairs per step.

`GRPOTrainer.__init__` gains `replay_dataset: Sequence[tuple[str, str]] | None = None`
— `(source, target)` pairs, the same `(Tg-as-text, PSELFIES)` shape the supervised
generation fine-tune used. `None` with a non-zero `replay_coef` is a **hard error**, not
a silent no-op: a run that believes it is doing replay and is not would look like
evidence that replay fails.

`scripts/train_grpo.py` loads `data/processed/tg/generation/train.jsonl` when
`train.replay_coef > 0`, and records the file's SHA-256 in the run manifest.

### Batch selection

Sampled with a generator seeded from `(config.seed, step_index)` — the same rule the
rollout already uses, so a resumed step draws exactly the replay batch an uninterrupted
run would have. Resume must stay bit-identical; that property is verified in
`docs/rl_runbook.md` and must not regress.

### Where it goes in the step

After `grpo_loss` and BEFORE `loss.backward()`, so both terms share one backward pass
and one optimizer step. Adding a second `optimizer.step()` would double the effective
learning rate on replay batches and desynchronise the step budget from every other arm.

The supervised forward runs in `eval()` like the rest of `step` — see the trainer's
module docstring on why the recompute must not switch to `train()`. Dropout noise in the
replay term would inject variance unrelated to any policy update.

### Metrics

`replay_loss`, `replay_coef` and `replay_batch_size` join the per-step row. Without
`replay_loss` logged there is no way to tell a run where replay is doing work from one
where it silently contributes nothing.

## Experiment

`configs/rl/composite_replay{01,05,20}.yaml` — identical to `composite.yaml` except
`experiment_name` and `train.replay_coef` ∈ {0.1, 0.5, 2.0}. Same single-variable
discipline the `kl_coef` sweep uses, enforced by a test.

`kl_coef` stays at round 1's 0.02 so replay is measured against the same anchor
`composite` had, not confounded with round 2.

### Success criterion, pre-registered

Replay works if some `replay_coef` gives, versus `composite`'s step-2000 numbers:

- **conditioning slope ≥ 0.70** (composite 0.274, baseline 0.904), measured by
  `scripts/conditioning_frontier.py`, AND
- **PV ≥ 0.75** (composite 0.933, baseline 0.658) — i.e. it keeps most of the structural
  gain rather than merely reverting toward the reference.

Failing the second while passing the first means replay has just reproduced the baseline
expensively, which is a null result and must be reported as one.

## What would falsify the whole idea

If every `replay_coef` either keeps conditioning while losing the structural gain, or
keeps the gain while losing conditioning, then the two objectives are not separable by
this mechanism — consistent with round 1's finding that the structural rewards are
chemically ANTI-correlated with high Tg (asking for higher Tg cost 17 points of PV). In
that case the remaining path is a reward that prices the target directly, and the only
verifiable version of that is Bicerano group contribution
(`docs/superpowers/specs/2026-08-21-bicerano-oracle-design.md`, still unbuilt).

## Tests that must be able to fail

- `replay_coef=0.0` produces a step numerically identical to today's trainer.
- `replay_coef>0` with `replay_dataset=None` raises, rather than training without replay.
- The replay batch for a given `(seed, step_index)` is identical across two trainers,
  so resume stays exact.
- `replay_loss` appears in the metrics row when replay is on, and is absent or zero when off.
- One backward pass and one `optimizer.step()` per step regardless of `replay_coef`.
- The shipped configs differ from `composite.yaml` in exactly `{experiment_name, train}`
  and within `train` in exactly `{replay_coef}`.
