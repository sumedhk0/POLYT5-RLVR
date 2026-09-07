# polyT5-RLVR

A clean-room research reproduction of **polyT5** — an encoder–decoder chemical language model for
generative polymer design — plus a separate, clearly-labelled extension asking what happens when you
post-train it with **verifiable-reward reinforcement learning**.

> Sahu, H., Xiong, W., Savit, A., Shukla, S. S., Ramprasad, R.
> *POLYT5: an encoder-decoder foundation chemical language model for generative polymer design.*
> npj Artificial Intelligence **2**, 30 (2026). [doi:10.1038/s44387-026-00087-1](https://doi.org/10.1038/s44387-026-00087-1)
> · preprint [arXiv:2510.18860](https://arxiv.org/abs/2510.18860)

T5 built from scratch in PyTorch — no `transformers`, no pretrained weights. 7.46M parameters matching
the paper's Table S2, pretrained on 92.3M polymers, fine-tuned for property prediction and conditional
generation, then used as the starting point for a five-arm GRPO study.

## The headline result

Post-training a conditional generator with a verifiable reward reliably **destroys the conditioning it
was fine-tuned for** — and that turns out to be catastrophic forgetting, fixable without giving up
verifiability.

| | validity (PV) | conditioning slope | on-target rate |
|---|---|---|---|
| supervised baseline | 0.665 | 0.904 | **0.738** |
| `composite` (best structural reward) | **0.915** | 0.268 | 0.138 |
| `+ stronger KL anchor` | 0.918 | 0.295 | 0.192 |
| **`+ supervised replay`** | **0.790** | **0.973** | **0.582** |

Five different verifiable rewards each optimised exactly what they measured and collapsed the
target-property response from 0.904 to ~0.27. A uniform-random control arm stayed flat throughout, so
the cause is the rewards and not the algorithm. Raising the KL penalty 2.5× recovered under 5% of the
loss; **adding a supervised-replay term recovered 99% of it** while keeping +12.5 points of validity
over the baseline — and reads no learned model, so the reward stays verifiable.

Full tables, per-arm trajectories and the negative results: [`docs/rlvr_results.md`](docs/rlvr_results.md).

## Two things live here, and they are kept apart

| | Phase 1–2 — **reproduction** | Phase 3 — **extension** |
|---|---|---|
| What | Span-corruption pretraining → supervised fine-tuning → sampling and screening | GRPO from the frozen supervised checkpoint |
| Source | The published polyT5 method | **Ours. Not in the paper.** |
| Status | Complete; baseline frozen with verified SHA-256s | Rounds 1–3 complete and scored |

> [!IMPORTANT]
> The published polyT5 work is **entirely supervised**. It contains no reinforcement learning, no reward
> ensemble, and no released Tg predictor. Nothing in this repository may describe GRPO or RLVR as part of
> the paper.

## The paper's data and code are not public

> "The datasets generated and/or analysed during the current study are **not publicly available due to IP
> protection being considered at authors' institution.**" — npj AI 2, 30 (2026)

No repository, no released weights, no released tokenizer (verified 2026-08-17). So we reproduce the
**method**, on documented public substitute data, at reduced scale. We never claim replication of the
paper's numbers, and every detail the paper leaves unspecified has an entry in
[`docs/ambiguity_register.md`](docs/ambiguity_register.md) recording the choice made and why.

## Pipeline

```
polymer structure (PSMILES, "*CCO*")
    → [*] replaced by [At]           chain-end marker; SELFIES cannot encode "*"
    → PSELFIES                       "[At][C][C][O][At]"
    → custom 458-token tokenizer
    → T5 encoder-decoder
    → span-corruption pretraining    92.3M polymers
    → supervised fine-tuning
         ├─ property prediction:      PSELFIES → "236.0"
         └─ conditional generation:   "500.0"  → PSELFIES
    → screening (SV → TSD → DD → PV, SR, SA, novelty)
    → [ours] GRPO against a verifiable reward
```

Both directions are text-to-text: the model emits the temperature one character at a time. That is why a
single extra digit once produced `4293.10 K` for a polymer whose true Tg was 481 K — and why the numeric
regression head below removes that failure mode by construction.

## Install

Requires Python 3.10–3.12 (3.12 recommended — RDKit and CUDA-torch wheel coverage is best there).

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev,app]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
```

Verify:

```bash
python -c "import torch, rdkit, selfies; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

> [!TIP]
> On Windows, prefer WSL2. Smart App Control blocks unsigned binaries by per-file reputation and will
> eventually refuse venv interpreters and vendored `.dll`s mid-project. WSL2 sidesteps it entirely and
> reproduces the frozen baseline bit-for-bit. See [`docs/rl_runbook.md`](docs/rl_runbook.md).

## Quickstart

```bash
# 1. fetch the substitute corpora (~71 MB)
python scripts/download_data.py --dataset pi1m --dataset tg

# 2. build the deterministic tokenizer artifact (458 tokens, content-hashed)
python scripts/build_tokenizer.py --out artifacts/tokenizer/polyt5_vocab.json

# 3. smallest end-to-end slice on real polymers:
#    chemistry → tokenizer → span corruption → forward → loss → step → checkpoint → reload
python scripts/run_vertical_slice.py --model-config configs/model/polyt5_small.yaml --steps 5

# 4. pretrain, then fine-tune both paper tasks
python scripts/pretrain.py --config configs/pretrain/pi1m_small.yaml
python scripts/finetune.py --task prediction  --init-checkpoint results/<run>/checkpoints/best.pt
python scripts/finetune.py --task generation  --init-checkpoint results/<run>/checkpoints/best.pt
```

Every script takes `--set key.subkey=value`, so a run can be re-pointed without editing YAML:

```bash
python scripts/pretrain.py --config configs/pretrain/pi1m_small.yaml \
    --set train.physical_batch_size=8 train.gradient_accumulation_steps=56
```

### Running the RL study

```bash
python scripts/train_grpo.py --arm composite            # one arm
python scripts/run_round1.py --then novelty composite   # chain several, resuming any partial run
python scripts/compare_arms.py                          # score every arm against the frozen protocol
python scripts/watch_rl.py                              # live dashboard
```

## Results

### Reproduction (Phase 1–2)

| | our reproduction |
|---|---|
| polyT5-medium parameters | 7.46M — matches Table S2 |
| Tg prediction, 5 splits | **28.6733 ± 0.7591 K** |
| Conditional generation (tuned sampling) | 47.316 K MAE, PV 0.558, TP 0.588 |

Frozen in [`artifacts/baseline/frozen_baseline.json`](artifacts/baseline/frozen_baseline.json) with
verified SHA-256s, and reproduced **bit-for-bit** on a second platform — delta exactly 0 on all five
splits.

### Prediction improvements (Phase 4, ours)

A seven-arm, two-seed ablation over five candidate improvements — 70 training runs. One worked:

- **Numeric regression head** — Tg MAE **28.60 → 27.11 K**, clearing a pre-registered threshold on both
  seeds and winning all 10 arm/split pairs. It also makes the digit-slip failure structurally impossible.
- **Conformal intervals** — the model now returns `412 ± 54 K` at 90% confidence, measured coverage
  0.900, distribution-free and independent of model quality.
- Descriptor auxiliaries, invariance augmentation, reliability weighting and a shared multi-task encoder
  each showed no effect or hurt. Details and the nulls: [`docs/group_a_results.md`](docs/group_a_results.md).

> [!WARNING]
> There is no clean held-out test set — only 7 of 7,354 polymers are held out by all five splits, which
> overlap ~20% pairwise. The **deltas** are sound because every arm ran an identical protocol; the
> absolute numbers should not be quoted against other published Tg models.

## Layout

```
src/polyt5/
    tokenization/   458-token vocabulary spec, deterministic tokenizer, artifact I/O
    data/           span corruption, collators, splits
    model/          T5 from scratch: relative position bias, attention, seq2seq LM
    training/       loop, AMP, gradient accumulation, checkpointing, resume
    generation/     greedy / temperature / top-p sampling, beam search
    chemistry/      PSMILES ↔ PSELFIES, validity, canonicalization, novelty  (no torch)
    evaluation/     SV/TSD/DD/PV, SELFIES round-trip, SA, conformal intervals
    inference/      property predictors, ensembles, the injection point for evaluation
    rl/             [ours] group rollout, advantages, clipped GRPO surrogate, trainer
    rewards/        [ours] reward components and the arm definitions

configs/            every experimental setting; no settings live in Python
scripts/            CLI entry points
docs/               reproduction notes, ambiguity register, results, runbook
artifacts/          tokenizer, frozen baseline, conformal calibration, splits
```

Architectural rules that let Phase 3 be added without rewriting Phase 1:

- **Chemistry and rewards never import torch** — reward workers run without a model in the process,
  verified by a subprocess test.
- **Model code never imports chemistry** — validity is an evaluation concern, not a network layer.
- **Dependencies point one way**: nothing outside `rl/` imports `rl/`, pinned by
  `tests/test_dependency_direction.py`. The supervised code does not know RL exists.
- **The tokenizer is a hashed on-disk artifact**, identical across every stage.
- **Checkpoints carry their config and tokenizer identity.** A checkpoint without its configuration is
  not a checkpoint.

## What "verifiable" means here, and why it is load-bearing

RLVR means the reward can be **checked**. Every arm still trained satisfies that:

| arm | reward | checkable without a lab? |
|---|---|---|
| `validity` | RDKit parse, terminus valency, deduplication | yes — arithmetic with a right answer |
| `novelty` | absent from the training corpus, and valid | yes — set membership |
| `synthesisability` | RDKit SA score ≤ threshold | yes — a deterministic heuristic |
| `composite` | PV + novelty + SA + within-batch diversity | yes — no term is model-scored |
| `constraint` | conjunction of the above | yes — no term is model-scored |
| `control` | uniform random, candidate-independent | yes — trivially |
| `accuracy` *(retired)* | Tg closeness from a learned predictor | **no** |

A generated polymer has no experimental Tg and never will — it has not been synthesised. So a Tg reward
is a model's opinion about a molecule nobody has made: legitimate for screening, but **RL against a
learned reward, not RLVR**. The `accuracy` arm is retired rather than deleted, kept as the motivating
negative result: its reward-scored error fell 52.5 → 31.4 K while unique fraction collapsed 0.951 → 0.535.

That single distinction is what the whole study turns on. The conditioning objective is the one thing
that *cannot* be verified without a lab, and leaving it out of the reward is precisely what costs the
conditioning.

## Findings

Things this reproduction established that the paper does not state — each recorded with evidence in
[`docs/reproduction.md`](docs/reproduction.md):

1. **The original is T5 v1.0 with tied embeddings.** Rebuilding all three sizes reproduces the reported
   parameter counts to within 0.19% / 0.04% / 0.005% — but only with `feed_forward_proj="relu"` and
   `tie_word_embeddings=True`. The paper never says which variant; the parameter counts do.
2. **The base vocabulary must be corpus-derived.** A library alphabet passed every curated test, then
   produced 2.1% unknown tokens across 22.4% of real sequences. Frequency-ranking from the training split
   drops that to zero.
3. **The paper's PSMILES→PSELFIES pipeline cannot run on vinyl polymers.** Joining both chain ends needs a
   one- or two-membered ring whenever the termini sit on the same or bonded atoms — polyethylene,
   polypropylene, PVC. RDKit cannot represent such rings.
4. **PV's valency rule is load-bearing.** `[At][C][Ring1][C][At]` decodes to `[At]=C[At]`: RDKit-valid with
   exactly two astatines, but the first is divalent, so it must fail. A terminus *count* is not enough.
5. **The 4-model Tg ensemble adds nothing** over a single model (28.82 K honest vs 28.67 K), and appears
   41% better than it is when scored on data three of its four members trained on.

## Documentation

| File | Contents |
|---|---|
| [`docs/reproduction.md`](docs/reproduction.md) | Every implementation detail the paper specifies, tagged `[PAPER]` / `[IMPLIED]` / `[OURS]` |
| [`docs/ambiguity_register.md`](docs/ambiguity_register.md) | Every unspecified detail, our choice, and its justification |
| [`docs/data.md`](docs/data.md) | Withheld datasets, public substitutes, licenses and provenance |
| [`docs/baseline.md`](docs/baseline.md) | Staged build plan, hardware reality, fidelity ledger |
| [`docs/rlvr_results.md`](docs/rlvr_results.md) | The three-round RL study, all arms, all negative results |
| [`docs/group_a_results.md`](docs/group_a_results.md) | The 70-run prediction ablation |
| [`docs/instrument_audit.md`](docs/instrument_audit.md) | What the Tg ensemble can and cannot measure |
| [`docs/rl_runbook.md`](docs/rl_runbook.md) | Stopping, resuming, chaining runs; environment pitfalls |

## Scientific integrity

Three sentences that are never merged:

- **"the paper reports …"** — a published number, cited
- **"our reproduction obtains …"** — our supervised run, on substitute data
- **"our RLVR extension obtains …"** — Phase 3, never attributed to polyT5

We do not fabricate datasets, and we do not silently invent unspecified details. Where a detail is
missing it gets a register entry rather than a quiet default. Negative results are reported as negative
results — including the ones that contradict an earlier optimistic reading in this repository's own
history.

## Hardware

Developed against a single **NVIDIA RTX 4080 Laptop GPU (12 GB)**. The paper used an L40S (48 GB), so the
published pretraining batch size of 450 is reached by gradient accumulation rather than assumed to fit. No
published batch size is hard-coded anywhere.

## License

MIT for the code here. The substitute datasets carry their own licenses — see
[`docs/data.md`](docs/data.md). No dataset is redistributed; `scripts/download_data.py` fetches them from
their original sources and records URL, size, and SHA-256.
