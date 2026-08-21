# Phase 3 plan: GRPO + RLVR extension

> **⚠️ APPARATUS IMPLEMENTED (Tasks 1–9). NO ARM HAS BEEN TRAINED YET. NOT PART OF THE PUBLISHED polyT5
> METHOD.**
>
> polyT5 (Sahu et al., npj Artificial Intelligence 2026) is **supervised throughout**: span-corruption
> pretraining, supervised fine-tuning, then sampling and screening. It contains no reinforcement learning
> of any kind. Everything in this document is **our proposed research extension** and must never be
> attributed to the paper. Results from it are reported as *"our RLVR extension obtains"* — a third
> category, distinct from both *"the paper reports"* and *"our reproduction obtains"*.
>
> This document predates the implementation and is kept as the original design record. The as-built spec
> and task-by-task plan are `docs/superpowers/specs/2026-08-20-grpo-rlvr-design.md` and
> `docs/superpowers/plans/2026-08-20-grpo-rlvr.md`; §8 below records entry-gate status.

The reward primitives (`src/polyt5/rewards/`) and RL core (`src/polyt5/rl/`) described below now exist,
built exactly per the constraint stated when this file was written: nothing in
`src/polyt5/{model,data,training,tokenization,chemistry}` imports either (verified by
`tests/test_dependency_direction.py`).

---

## 1. Research question

Not *"can polyT5 generate polymers?"* — the paper already shows that. The question is:

> **Can verifiable reinforcement learning improve property-conditioned polymer generation beyond
> supervised polyT5?**

Three arms, all evaluated with the same metrics on the same held-out targets:

| Arm | Description |
|---|---|
| **A — supervised baseline** | Fine-tuned polyT5 + default sampling |
| **B — tuned sampling** | Same checkpoint, sampling hyperparameters tuned on validation (the paper's own sweep: top_p ∈ {0.75, 0.95}, T ∈ [0.1, 2.0]) |
| **C — RLVR** | Arm A's checkpoint as initial policy, then GRPO against verifiable rewards |

Arm B matters: much of what RL appears to buy can often be had by turning the temperature down. If C does
not beat a properly tuned B, there is no result.

## 2. Why *verifiable* rewards

Polymer generation is unusually well suited to RLVR because the reward is **computed, not learned**:

- Chemical validity is an RDKit parse — a fact, not a prediction.
- The `[At]` terminus rule (exactly two, each valency 1 — the paper's PV filter) is a graph check.
- SELFIES reproducibility is a string round-trip.
- Novelty is a set-membership test against the training corpus.

Only the *property* term needs a model, and that model is the supervised polyT5 property predictor from
Phase 1 — which introduces the central methodological risk (§7).

**We will not train a learned reward model** unless a demonstrated need appears. The first experiment uses
externally computed rewards exclusively.

## 3. Module layout (to be created in Phase 3, not now)

```
src/polyt5/rl/
    rollout.py            group sampling from a prompt; returns sequences + per-token logprobs + masks
    rewards.py            reward aggregation and normalization; composes src/polyt5/rewards/*
    advantages.py         group-relative advantage computation
    grpo.py               the clipped policy objective + KL regularization
    reference_policy.py   frozen copy of the supervised checkpoint
    trainer.py            the RL training loop (mirrors training/ structure, shares checkpoint format)

src/polyt5/rewards/
    validity.py           RDKit parse + PV terminus rule        (wraps polyt5.chemistry.validity)
    tg.py                 distance from target Tg               (uses the Phase-1 property model)
    dielectric.py         ε target                              (deferred: no property model yet)
    bandgap.py            Eg target                             (deferred)
    solubility.py         solubility constraints                (deferred)
    novelty.py            novelty vs training corpus            (wraps polyt5.chemistry.novelty)
    constraints.py        structural/chemical hard constraints
```

**Dependency direction is one-way and non-negotiable:**

```
rl/  →  model/, tokenization/, chemistry/, generation/, evaluation/
rl/  ←  nothing
```

The supervised implementation must never import from `rl/` or `rewards/`, and must never know RL exists.
The RL system *consumes* a supervised checkpoint; it does not require the supervised code to accommodate it.

## 4. What the supervised phase must provide (design constraints already honoured)

These are the hooks Phase 3 needs. They are being built into Phase 1 **because they are also good
supervised engineering**, not as RL special-cases:

| Requirement | Where it lives | Why RL needs it |
|---|---|---|
| Checkpoints carry config + tokenizer identity | `training/checkpoint.py` | The policy and reference model must be provably the same architecture and vocabulary |
| Model returns raw logits, loss optional | `model/transformer.py` | RL needs per-token log-probs, not a scalar loss |
| Cache-friendly incremental decoding | `model/transformer.py` | Group rollouts are the dominant cost |
| Batch generation with explicit seeds | `generation/` | Reproducible rollouts |
| Chemistry layer with zero torch dependency | `chemistry/` | Reward workers must run without a GPU or a model in the process |
| Evaluators return structured results, never print | `evaluation/` | Rewards are assembled from evaluator outputs |
| Tokenizer is a hashed on-disk artifact | `tokenization/` | Policy, reference, and reward all must agree byte-for-byte |
| Padding/EOS handling explicit and tested | `data/collate.py`, `generation/` | Sequence-level rewards over variable-length outputs |

## 5. The GRPO loop (target design)

```
for each conditioning prompt (a target Tg value):
    sample G candidates from the policy      (group size G, configurable)
    for each candidate:
        decode PSELFIES → PSMILES
        compute verifiable reward components
        r_i = weighted sum
    A_i = (r_i - mean(r)) / (std(r) + eps)          group-relative advantage
    update the policy with a clipped objective + KL penalty toward the frozen reference
```

Required capabilities, all configurable: group size, rollouts per prompt, reward normalization scheme,
advantage computation, policy/reference separation, KL regularization coefficient, clipped objective,
sequence-level rewards over variable-length outputs, EOS handling, padding masks, gradient accumulation,
and checkpointing.

## 6. Reward design — deliberately unweighted for now

Candidate components:

| Component | Verifiable? | Notes |
|---|---|---|
| Chemical validity | ✅ RDKit | Invalid output strongly penalized |
| PV terminus rule | ✅ graph check | Exactly two `[At]`, each valency 1 |
| SELFIES reproducibility | ✅ round-trip | The paper's SR metric |
| Target property (Tg) | ⚠️ model-predicted | Distance from the requested value |
| Other properties (ε, Eg, solubility) | ⚠️ model-predicted | Deferred until those models exist |
| Novelty | ✅ set membership | Distance from the training corpus |
| Diversity | ✅ within-group | Penalize mode collapse onto a few polymers |
| Constraints | ✅ substructure rules | Hard structural requirements |

Conceptually `R = validity + property + novelty + diversity − penalties`.

> **No weights are chosen yet, on purpose.** Picking reward weights before the supervised baseline exists
> would be tuning against an unmeasured target. The reward module must make each component independently
> switchable so ablations over reward composition are cheap.

## 7. Known risks to design against

These are no longer hypothetical. The supervised baseline has been measured, and three of them are
quantified below with real numbers from `docs/baseline.md`.

1. **Reward hacking via the property model — measured, not speculative.** The Tg predictor emits only
   **146 distinct values across 1,471 test polymers**, with a large attractor at the most common training
   labels: **22.9% of scored generations land on exactly 503.1 K**. The label distribution is inherently
   lumpy (Celsius-to-Kelvin conversion piles 80.3% of training labels onto `.1` decimals, and 32 training
   polymers share the label 323.1 K). So a policy can satisfy "≈500 K" by steering into a modal bucket
   rather than by controlling Tg. Mitigations, in order of importance:
   - **Ensemble disagreement gate — built and measured.** Five Tg models come free from the paper's own
     five-split protocol (`results/tg_prediction_5splits/split_*/checkpoints/best.pt`).
     `EnsemblePropertyPredictor.predict_with_uncertainty` returns `(mean, std, n_contributing)` per
     candidate. On 200 real generated polymers: mean disagreement **16.7 K**, median 14.7 K, p90 29.7 K,
     max **45.2 K**. For scale, each member's own held-out MAE is 34.6 ± 1.2 K — so inter-model
     disagreement is roughly half the error magnitude and varies threefold across candidates, which is
     exactly the spread a gate needs. Candidates where five independently trained models disagree by 45 K
     are the ones a policy would otherwise farm.
   - **A held-out auditor** — one split's model never used in any reward, only for final scoring.
   - **A novelty ceiling**, so the policy cannot escape the predictor's support entirely.
   - **Log the predicted-value histogram every RL epoch.** Collapse onto modal values is the signature of
     this failure, and it is visible immediately if plotted.
2. **SELFIES reproducibility is a trap as a reward term.** SR is **100% for ring-free outputs and 16.5%
   for ring-bearing ones** — it measures whether a molecule survives ring-index re-derivation, not
   quality. Rewarding it directly pushes the policy toward aliphatic, ring-free polymers, while most
   high-Tg polymers are aromatic. Either condition it on structure class or keep its weight near zero.
3. **The circularity between generator and predictor.** Both were fine-tuned on the same LamaLab Tg data,
   so the predictor scoring the generator is partly self-referential *before* RL even starts. The auditor
   split is the only clean measurement; quote it separately in every result.
2. **Mode collapse.** Group-relative advantages plus a validity-heavy reward push toward a handful of
   trivially valid polymers. The diversity term and within-group duplicate penalties exist for this.
3. **Degenerate short sequences.** Short PSELFIES are easier to keep valid. Length-aware normalization is
   required, and length statistics must be logged from the first run.
4. **KL collapse or explosion.** The reference policy is the frozen supervised checkpoint; the KL
   coefficient needs a sweep, and the KL must be logged per step, not just the reward.
5. **Comparing against a weak baseline.** Arm B exists to prevent this. The sampling sweep must be run
   properly before any RL claim is made.

## 8. Entry criteria — none of this starts until all are true

**[SATISFIED, Task 9.]** All five criteria hold, recorded in `artifacts/baseline/frozen_baseline.json`
(frozen `2026-08-20T23:02:41Z`, commit `5ff1c44`) and `docs/baseline.md`. The Phase-3 apparatus this
criterion gates — reward components (`src/polyt5/rewards/`), RL core (`src/polyt5/rl/`), `GRPOTrainer`,
`scripts/train_grpo.py`, the four arm configs, and `scripts/compare_arms.py` — is now built and tested
(Tasks 1–9; see `docs/superpowers/plans/2026-08-20-grpo-rlvr.md`). **No arm has been trained yet** — this
marks the entry gate open, not a result.

- [x] Supervised pretraining runs end-to-end and its loss curve is recorded
- [x] Tg property prediction reproduces a sane MAE/RMSE/R² on the substitute dataset, over five splits
      (mean MAE 28.6733 ± 0.7591 K, `n=5`, per `frozen_baseline.json`'s `tg_prediction_5split`)
- [x] Tg-conditioned generation works, with the full evaluation layer (SV/TSD/DD/PV, SR, SA, novelty)
- [x] Arm A **and** Arm B are measured, frozen, and written to `results/` (`arm_a_default_sampling` /
      `arm_b_tuned_sampling` in `frozen_baseline.json`)
- [x] The frozen baseline checkpoint is tagged and its manifest recorded (all 8 artifacts SHA-256 verified
      in `frozen_baseline.json`; splits 0–3 are the reward ensemble, split 4 is the auditor and never
      enters a reward path, per `success_criterion` and `auditor_note`)

Only then does Phase 3 begin.
