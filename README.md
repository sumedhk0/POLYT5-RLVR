# polyT5-RLVR

A reproduction of **polyT5** — an encoder–decoder chemical language model for generative
polymer design — plus an extension asking what happens when you post-train it with verifiable-reward
reinforcement learning.

T5 built from scratch in PyTorch (no `transformers`, no pretrained weights): 7.46M parameters matching
the paper's Table S2, pretrained on 92.3M polymers, fine-tuned for property prediction and conditional
generation, then used as the starting point for a five-arm GRPO study.

> Sahu, H., Xiong, W., Savit, A., Shukla, S. S., Ramprasad, R. *POLYT5: an encoder-decoder foundation
> chemical language model for generative polymer design.* npj Artificial Intelligence **2**, 30 (2026).
> [doi:10.1038/s44387-026-00087-1](https://doi.org/10.1038/s44387-026-00087-1)

> [!IMPORTANT]
> polyT5 is **entirely supervised**. The RL work here is ours, not part of the paper. The paper's data and
> code are not public, so this reproduces the *method* on public substitute data at reduced scale — never
> the paper's numbers.

## Headline

Post-training a conditional generator with a verifiable reward **destroys the conditioning it was
fine-tuned for** — and that is catastrophic forgetting, fixable without giving up verifiability.

| | validity (PV) | conditioning slope | on-target rate |
|---|---|---|---|
| supervised baseline | 0.665 | 0.904 | **0.738** |
| `composite` (best structural reward) | **0.915** | 0.268 | 0.138 |
| `+ stronger KL anchor` | 0.918 | 0.295 | 0.192 |
| **`+ supervised replay`** | **0.790** | **0.973** | **0.582** |

Five different verifiable rewards each optimised exactly what they measured and collapsed the
target-property response; a uniform-random control arm stayed flat, so the cause is the rewards, not the
algorithm. A 2.5× KL penalty recovered under 5% of the loss. Supervised replay recovered **99%** while
keeping +12.5 points of validity — and reads no learned model, so the reward stays verifiable.

## Install

Python 3.10–3.12 (3.12 recommended).

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev,app]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
pytest -q
```

> [!TIP]
> On Windows, prefer WSL2 — Smart App Control blocks unsigned binaries by per-file reputation and will
> eventually refuse venv interpreters mid-project.

## Quickstart

```bash
# data + tokenizer
python scripts/download_data.py --dataset pi1m --dataset tg
python scripts/build_tokenizer.py --out artifacts/tokenizer/polyt5_vocab.json

# smallest end-to-end slice on real polymers
python scripts/run_vertical_slice.py --model-config configs/model/polyt5_small.yaml --steps 5

# pretrain, then both paper tasks
python scripts/pretrain.py --config configs/pretrain/pi1m_small.yaml
python scripts/finetune.py --task prediction --init-checkpoint results/<run>/checkpoints/best.pt
python scripts/finetune.py --task generation --init-checkpoint results/<run>/checkpoints/best.pt

# the RL study
python scripts/train_grpo.py --arm composite
python scripts/compare_arms.py      # score every arm against the frozen protocol
python scripts/watch_rl.py          # live dashboard
```

Every script takes `--set key.subkey=value` overrides, so a run can be re-pointed without editing YAML.

## Results

**Reproduction** — polyT5-medium at 7.46M parameters (matches Table S2); Tg prediction
**28.6733 ± 0.7591 K** over five splits; conditional generation 47.316 K MAE. Frozen in
`artifacts/baseline/frozen_baseline.json` with verified SHA-256s, and reproduced bit-for-bit on a second
platform.

**Prediction (ours)** — a seven-arm, two-seed ablation, 70 runs. One of five candidate improvements
worked: a **numeric regression head**, Tg MAE 28.60 → **27.11 K**, clearing a pre-registered threshold on
both seeds and winning all 10 arm/split pairs. Conformal intervals now return `412 ± 54 K` at 90%
confidence, measured coverage 0.900. Descriptor auxiliaries, invariance augmentation, reliability
weighting and a shared multi-task encoder showed no effect or hurt.

**RL (ours)** — three rounds, summarised above.

> [!WARNING]
> There is no clean held-out test set: only 7 of 7,354 polymers are held out by all five splits, which
> overlap ~20% pairwise. The **deltas** are sound because every arm ran an identical protocol; the
> absolute numbers should not be quoted against other published Tg models.

## Docs

[`reproduction.md`](docs/reproduction.md) (what the paper specifies, tagged) ·
[`ambiguity_register.md`](docs/ambiguity_register.md) (every unspecified detail and our choice) ·
[`data.md`](docs/data.md) (substitutes, licenses, provenance) ·
[`rlvr_results.md`](docs/rlvr_results.md) (all arms, all negative results) ·
[`group_a_results.md`](docs/group_a_results.md) (the 70-run ablation) ·
[`rl_runbook.md`](docs/rl_runbook.md) (stopping, resuming, environment pitfalls)

Developed on a single RTX 4080 Laptop (12 GB); the paper used an L40S (48 GB), so its batch size of 450 is
reached by gradient accumulation. MIT for the code; datasets carry their own licenses and none is
redistributed here.
