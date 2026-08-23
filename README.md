# polyT5-RLVR

A clean-room research reproduction of **polyT5** — an encoder–decoder chemical language model for
generative polymer design — plus a **separate, clearly-labelled** research extension exploring GRPO/RLVR.

> Sahu, H., Xiong, W., Savit, A., Shukla, S. S., Ramprasad, R.
> *POLYT5: an encoder-decoder foundation chemical language model for generative polymer design.*
> npj Artificial Intelligence **2**, 30 (2026). [doi:10.1038/s44387-026-00087-1](https://doi.org/10.1038/s44387-026-00087-1)
> · preprint [arXiv:2510.18860](https://arxiv.org/abs/2510.18860)

---

## Two things live here, and they are kept apart

| | Phase 1–2 — **reproduction** | Phase 3 — **extension** |
|---|---|---|
| What | Span-corruption pretraining → supervised fine-tuning → sampling and screening | GRPO from the supervised checkpoint. Rewards are verifiable for `validity`/`control`, learned-model-scored for the Tg arms — see [below](#not-all-of-it-is-actually-verifiable--and-the-distinction-is-load-bearing) |
| Source | The published polyT5 method | **Ours. Not in the paper.** |
| Status | In progress; Arm A/B measured and frozen for RLVR comparison (see [`docs/baseline.md`](docs/baseline.md)) | Round 1 partially trained; **no final result yet** (see [RLVR extension](#phase-3-grporlvr-our-extension) below) |

The published polyT5 work is **entirely supervised**. It contains no reinforcement learning. Nothing in
this repository may describe GRPO or RLVR as part of the paper.

---

## Read this first: the paper's data and code are not public

> "The datasets generated and/or analysed during the current study are **not publicly available due to IP
> protection being considered at authors' institution.**" — npj AI 2, 30 (2026)

There is no code-availability statement, no repository, no released weights, and no released tokenizer
(verified 2026-08-17). Therefore:

- We reproduce the **method**, on **documented public substitute data**, at **reduced scale**.
- We never claim replication of the paper's numbers. See [`docs/reproduction.md`](docs/reproduction.md) §9.
- Every detail the paper leaves unspecified has an entry in
  [`docs/ambiguity_register.md`](docs/ambiguity_register.md) recording the choice we made and why.

---

## The pipeline

```
polymer structure (PSMILES, "*CCO*")
    → [*] replaced by [At]           chain-end marker; SELFIES cannot encode "*"
    → PSELFIES                       "[At][C][C][O][At]"
    → custom 458-token tokenizer
    → T5 encoder-decoder
    → span-corruption pretraining
    → domain-adapted polymer LM
    → supervised fine-tuning
         ├─ property prediction:      PSELFIES → "236.0"
         └─ conditional generation:   "500.0"  → PSELFIES
    → candidate sampling and screening (SV → TSD → DD → PV, SR, SA, novelty)
```

---

## Install

Requires Python 3.10–3.12 (3.12 recommended; RDKit and CUDA-torch wheel coverage is best there).

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev,track]"
# CUDA build of torch (adjust the index for your driver):
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
```

Verify:

```bash
python -c "import torch, rdkit, selfies; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

> This machine's environment lives outside the project tree (`C:\Users\sumedh\.venvs\polyt5-rlvr`)
> deliberately: the repository sits in a OneDrive-synced folder, and syncing a multi-gigabyte `.venv`
> is slow and failure-prone.

---

## Quickstart

```bash
# 1. fetch the substitute corpora (small: ~71 MB total)
python scripts/download_data.py --dataset pi1m --dataset tg

# 2. build the deterministic tokenizer artifact (458 tokens, content-hashed)
python scripts/build_tokenizer.py --out artifacts/tokenizer/polyt5_vocab.json

# 3. convert PSMILES → PSELFIES → token ids
python scripts/prepare_pretraining_data.py --config configs/pretrain/dev.yaml

# 4. the smallest end-to-end slice, on real polymers:
#    chemistry -> tokenizer -> span corruption -> batch -> polyT5-small
#    -> forward -> loss -> backward -> optimizer step -> checkpoint -> reload
python scripts/run_vertical_slice.py --model-config configs/model/polyt5_small.yaml --steps 5

# (synthetic-only variant, no data or tokenizer needed)
python scripts/smoke_train.py --model-config configs/model/polyt5_tiny.yaml --steps 3

# 5. span-corruption pretraining
python scripts/pretrain.py --config configs/pretrain/dev.yaml
python scripts/pretrain.py --config configs/pretrain/pi1m_small.yaml   # full PI1M, polyT5-small
```

```bash
# 6. downstream fine-tuning, both paper tasks
python scripts/prepare_tg_data.py --config configs/finetune/tg_prediction.yaml

python scripts/finetune.py --task prediction \
    --init-checkpoint results/<pretrain-run>/checkpoints/best.pt   # PSELFIES -> "236.0", beam-4
python scripts/finetune.py --task generation \
    --init-checkpoint results/<pretrain-run>/checkpoints/best.pt   # "500.0" -> PSELFIES, top-p

# the paper's central ablation: same run, random initialisation
python scripts/finetune.py --task prediction --no-pretrained
```

Every script takes `--set key.subkey=value` overrides, so a run can be re-pointed without editing YAML:

```bash
python scripts/pretrain.py --config configs/pretrain/pi1m_small.yaml \
    --set train.physical_batch_size=8 train.gradient_accumulation_steps=56
# 8 x 56 = 448; set target_effective_batch_size to have the mismatch caught, not guessed
```

---

## Layout

```
src/polyt5/
    tokenization/   458-token vocabulary spec + deterministic tokenizer + artifact I/O
    data/           span corruption, collators, dataset loading, splits
    model/          T5 from scratch: relative position bias, attention, layers, seq2seq LM
    training/       training loop, AMP, gradient accumulation, checkpointing, resume
    generation/     greedy / temperature / top-p sampling, beam search, batch generation
    chemistry/      PSMILES ↔ PSELFIES, validity, canonicalization, novelty  (no torch dependency)
    evaluation/     SV/TSD/DD/PV, SELFIES reproducibility, SA, property metrics
    utils/          seeding, device, config, run directories, metric logging
    rl/             Phase 3 (ours) — group rollout, advantages, clipped GRPO surrogate, trainer
    rewards/        Phase 3 (ours) — reward components + the five arm definitions

configs/            every experimental setting; no settings live in Python
scripts/            CLI entry points
tests/              tokenizer, span corruption, model, training, generation, chemistry
docs/               reproduction notes, ambiguity register, data provenance, baseline, RLVR plan
results/            one directory per experiment: config, manifest, metrics, predictions, checkpoints
artifacts/          the tokenizer artifact
```

Architectural rules that make the Phase-3 extension possible without rewriting Phase 1:

- **Chemistry and rewards never import torch.** Reward workers must run without a model in the process
  (verified by a subprocess test — see `tests/test_chemistry.py`, `tests/test_rewards.py`). This is
  narrower than "torch is confined to `rl/`/`model/`" — it is not, and never has been: `training/`,
  `generation/`, `data/`, `inference/` and `evaluation/sweep.py` all import torch too, same as `rl/` and
  `model/` do. No `transformers` dependency exists anywhere in this repository.
- **Model code never imports chemistry.** Validity is an evaluation concern, not a layer of the network.
- **Dependencies point one way:** `rl/ → {model, tokenization, chemistry, generation, evaluation,
  inference, training}`, never the reverse — nothing outside `rl/` imports `rl/`, verified by
  `tests/test_dependency_direction.py`. The supervised code does not know RL exists.
- **The tokenizer is a hashed on-disk artifact**, identical across pretraining, fine-tuning, evaluation,
  generation, and future rollouts.
- **Checkpoints always carry their config and tokenizer identity.** A checkpoint without its configuration
  is not a checkpoint.

---

## Phase 3: GRPO/RLVR (our extension)

**This is our research extension, not part of the published polyT5 method.** polyT5 (Sahu et al., npj
Artificial Intelligence 2026) is supervised throughout — span-corruption pretraining, supervised
fine-tuning, then sampling and screening. It contains no reinforcement learning of any kind, no reward
ensemble, no uncertainty estimate, and never releases a Tg predictor. Nothing here may be described as part
of the paper's method; results from it are reported as **"our RLVR extension obtains …"** — a third
category, distinct from *"the paper reports …"* and *"our reproduction obtains …"* (see
[Scientific integrity](#scientific-integrity)).

### Not all of it is actually "verifiable" — and the distinction is load-bearing

RLVR means the reward can be **checked**. Ours only partly can, and the arms differ sharply:

| arm | reward | checkable without a lab? |
|---|---|---|
| `validity` | RDKit parse, terminus valency, deduplication | **yes** — arithmetic with a right answer |
| `control` | uniform random, candidate-independent | **yes** — trivially |
| `accuracy` | Tg closeness, from a learned predictor | **no** |
| `composite` | weighted mix including Tg | partly |
| `constraint` | conjunction including a Tg window | partly |

For a **generated** polymer there is no experimental Tg and never will be — the molecule has not been
synthesised. So a Tg-based reward is a model's opinion about a molecule nobody has made. That is a
legitimate and common way to do generative materials design — the paper screens its own 6.17M candidates
with polyT5-based predictors the same way — but it is **RL against a learned reward**, not RLVR, and this
repository labels it as such rather than letting "verifiable" cover the whole study.

The instrument behind those Tg arms is characterised in [`docs/instrument_audit.md`](docs/instrument_audit.md):
the 4-model ensemble adds nothing over a single model (28.82 K honest vs 28.67 K), it appears 41% better
than it is when scored on data three of its four members trained on, and σ explains roughly 2% of error
variance. **Phase 4 (`docs/superpowers/specs/2026-08-23-phase4-group-a-design.md`) exists to improve that
instrument**; until it lands, Tg-arm results are reported as model-scored secondary observations.

**Status: round 1 partially trained.** The entry gate (`docs/rlvr_plan.md` §8) is satisfied — the
supervised baseline is frozen (`artifacts/baseline/frozen_baseline.json`, verified SHA-256) and Arm A /
Arm B are measured against it.

- `accuracy` — **complete** (2000 steps). In-training diagnostics show the reward-ensemble-scored
  conditioning error falling 52.5 → 31.4 K **while `unique_fraction` fell 0.951 → 0.535**: mode collapse.
  The cost is verified (deduplication is structural); the benefit is not (the reward ensemble scoring the
  policy trained to satisfy it). Reported as a motivating negative result, not a success.
- `validity` — in training.
- `composite`, `constraint`, `control` — not started.

No arm has been through `compare_arms.py`, so **this repository still reports no final RLVR result.**

What is built on top of the frozen baseline:

- **Reward components** (`src/polyt5/rewards/`) — validity gate, Tg closeness with confidence weighting,
  novelty, and the five reward arms (`accuracy`, `validity`, `composite`, `constraint`, `control`). Deliberately
  torch-free, so reward workers run CPU-only. The confidence weight scales by how much of the ensemble
  could actually score a candidate (`n_contributing / n_total`) and substitutes the maximum observed
  disagreement, not zero, when only one member of several answered — without that, a candidate three of
  four reward models cannot parse outscores one all four agree on.
- **RL core** (`src/polyt5/rl/`) — group rollout, group-relative advantages, the clipped GRPO surrogate
  with a k3 KL anchor to a frozen reference policy, and `GRPOTrainer`, the synchronous training loop.
- **Training CLI** (`scripts/train_grpo.py`) and **five arm configs** (`configs/rl/*.yaml`) — one GRPO run
  per arm, differing only in reward. `control` earns a uniform random reward independent of the
  candidate: if a meaningless reward moves the same metrics, then no other arm's movement is
  attributable to its reward design. Without it the other four are uninterpretable.
- **Drift monitoring** (`src/polyt5/rl/drift.py`) — spec §4.4's max-Tanimoto-to-training distribution,
  logged every 50 steps, ON by default. The held-out split-4 auditor gap is OFF by default and opt-in via
  `--drift-auditor`: σ (ensemble disagreement) is itself optimized against — the Tg reward's confidence
  weight explicitly rewards the policy for landing where the four reward models agree — so σ cannot double
  as a trustworthy drift signal, and the auditor gap was the proposed check on that. But split 4 shares
  ~80% of its training data with each reward model (independent random 80% draws from one corpus), so it
  detects ensemble-specific error well and corpus-wide error barely; given that limited power, the default
  is to never open its checkpoint at all — held out of the training *process*, not merely the reward path.
  When loaded (`--drift-auditor`), the containment is enforced by construction and pinned by a test that
  the step's rewards, advantages and loss are identical with and without it. max-Tanimoto needs no auditor
  and stays on regardless, since it is not itself optimized against.
- **Arm-comparison matrix** (`scripts/compare_arms.py`) — samples fresh candidates from every trained arm
  under the frozen evaluation protocol and scores them twice: once by the reward ensemble (splits 0–3,
  "the metric it optimized") and once by the held-out auditor (split 4, never used in any reward path).

**Pre-registered success criterion** (`frozen_baseline.json`'s `success_criterion` and
`pre_registered_metrics`): an RLVR arm succeeds only if it beats Arm B on **the metric it actually
optimized** — `accuracy_score`, `pv_rate`, `composite_score`, `constraint_satisfaction_rate`, each pinned
in the frozen record with a minimum effect size — under the reward ensemble, *and* that gain survives
scoring by the auditor, *and* it was sampled at Arm B's temperature/top-p. "Beats" requires both an
improvement of at least the pre-registered `min_margin` and a 95% bootstrap CI over candidates that
excludes zero; a bare inequality on one generation seed cannot separate an 0.1 K win from noise.
`compare_arms` refuses to run if its code and the frozen record disagree. This is `[OURS]` — the paper
defines no such criterion, ensembles nothing, and never audits with a held-out model.

> **What the auditor can and cannot establish.** `[OURS]` `scripts/run_splits.py` builds **five
> independent random 80/20 splits of the same corpus**, explicitly not a partitioning k-fold. Split 4
> therefore trains on a random 80% of the same LamaLab Tg data and shares ~80% of its training set with
> each reward model in expectation, and its own held-out 20% is *not* held out from the reward ensemble.
> The auditor tests **"is this gain an artifact of *these four particular models'* idiosyncrasies?"** — a
> real and worthwhile test. It cannot test "is the Tg claim true": a reward hack exploiting a genuinely
> data-sparse region of chemical space fools all five models identically, because all five are sparse
> there. The auditor is held out of the reward **path**, not statistically independent of the reward
> models.

> **Known limitation: novelty is exact-canonical-match.** `[OURS]` The novelty term C3 rewards and C4
> requires is the absence of the candidate's exact canonical PSMILES from the training index. **A
> one-atom edit of a memorised training polymer therefore scores `novel = 1.0`.** No reward term anywhere
> can distinguish that from genuinely new chemistry. This is not fixed — changing the reward now would
> change what the arms optimize — but it is measured: the drift monitor reports max-Tanimoto to the
> labelled set and the fraction of candidates whose nearest known neighbour is at Tanimoto ≥ 0.9, so a
> C3/C4 arm rediscovering the training set is visible during the run rather than assumed away. Read the
> C3 and C4 columns of the matrix with this in mind.

**No outcome is claimed here.** No arm has been trained; this section describes the apparatus and the
criterion it will be measured against, not a result. See `docs/rlvr_plan.md` for the original design
record, `docs/superpowers/specs/2026-08-20-grpo-rlvr-design.md` for the as-built spec, and
`docs/superpowers/plans/2026-08-20-grpo-rlvr.md` for the task-by-task plan.

---

## Findings so far

Things this reproduction established that the paper does not state. Each is recorded with evidence in
[`docs/reproduction.md`](docs/reproduction.md) and [`docs/ambiguity_register.md`](docs/ambiguity_register.md).

1. **The original is T5 v1.0 with tied embeddings.** Rebuilding all three sizes from Table S2 reproduces
   the reported parameter counts to within 0.19% / 0.04% / 0.005% — but only with `feed_forward_proj="relu"`
   and `tie_word_embeddings=True`. The paper never says which T5 variant it used; the parameter counts do.
2. **The base vocabulary must be corpus-derived, not library-derived.** A `selfies`-library alphabet passed
   every curated test and then produced 2.1% unknown tokens across 22.4% of real sequences, missing
   stereochemistry (`[/C]`, `[\C]`), bonded-terminus forms (`[=At]`, `[/At]`) and explicit-hydrogen species.
   Frequency-ranking from the training split drops that to zero, and generalizes to a second dataset it was
   not derived from (1 unknown token in 371k).
3. **The paper's PSMILES→PSELFIES pipeline cannot run on vinyl polymers.** Joining the two chain ends needs
   a one- or two-membered ring whenever both termini sit on the same atom or on two bonded atoms — that is
   polyethylene, polypropylene, PVC and every other vinyl polymer. RDKit cannot represent such rings. The
   paper does not mention this class.
4. **Canonicalizing before cleaving does what the paper claims.** 14 different SMILES writings of 4 polymers
   — reversed direction, branch-first, kekulized vs aromatic — collapse to exactly one PSELFIES each.
5. **PV's valency rule is load-bearing, not decorative.** `[At][C][Ring1][C][At]` decodes to `[At]=C[At]`:
   RDKit-valid with exactly two astatines, but the first is divalent, so it must fail PV. A terminus *count*
   alone is not enough.

## Documentation

| File | Contents |
|---|---|
| [`docs/reproduction.md`](docs/reproduction.md) | Every implementation detail the paper specifies, quoted and tagged `[PAPER]` / `[IMPLIED]` / `[OURS]`, plus what cannot be reproduced |
| [`docs/ambiguity_register.md`](docs/ambiguity_register.md) | Every unspecified detail, our choice, and its justification |
| [`docs/data.md`](docs/data.md) | The withheld datasets, the public substitutes, their licenses and provenance |
| [`docs/baseline.md`](docs/baseline.md) | Staged build plan, hardware reality, fidelity ledger, results tables |
| [`docs/rlvr_plan.md`](docs/rlvr_plan.md) | The original Phase-3 GRPO/RLVR design record; entry gate now satisfied |
| [`docs/superpowers/specs/2026-08-20-grpo-rlvr-design.md`](docs/superpowers/specs/2026-08-20-grpo-rlvr-design.md) | The as-built Phase-3 spec — algorithm, architecture, and pre-registered success criterion |
| [`docs/superpowers/plans/2026-08-20-grpo-rlvr.md`](docs/superpowers/plans/2026-08-20-grpo-rlvr.md) | The task-by-task Phase-3 implementation plan (Tasks 1–9) |

---

## Scientific integrity

Three sentences that are never merged:

- **"the paper reports …"** — a published number, cited
- **"our reproduction obtains …"** — our supervised run, on substitute data, with the register version
- **"our RLVR extension obtains …"** — Phase 3, never attributed to polyT5

We do not fabricate datasets. We do not silently invent unspecified details. Where a detail is missing, it
gets a register entry rather than a quiet default.

## Hardware

Developed against a single **NVIDIA RTX 4080 Laptop GPU (12 GB)**. The paper used an NVIDIA L40S (48 GB),
so the published pretraining batch size of 450 is reached via gradient accumulation rather than assumed to
fit. No published batch size is hard-coded anywhere.

## License

MIT for the code in this repository. The substitute datasets carry their own licenses — see
[`docs/data.md`](docs/data.md). No dataset is redistributed here; `scripts/download_data.py` fetches them
from their original sources and records URL, size, and SHA-256.
