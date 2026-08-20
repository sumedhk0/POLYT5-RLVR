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
| What | Span-corruption pretraining → supervised fine-tuning → sampling and screening | GRPO with verifiable rewards, initialized from the supervised checkpoint |
| Source | The published polyT5 method | **Ours. Not in the paper.** |
| Status | In progress | Designed only — **no code** (see [`docs/rlvr_plan.md`](docs/rlvr_plan.md)) |

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
    rl/             Phase 3 — empty by design
    rewards/        Phase 3 — empty by design

configs/            every experimental setting; no settings live in Python
scripts/            CLI entry points
tests/              tokenizer, span corruption, model, training, generation, chemistry
docs/               reproduction notes, ambiguity register, data provenance, baseline, RLVR plan
results/            one directory per experiment: config, manifest, metrics, predictions, checkpoints
artifacts/          the tokenizer artifact
```

Architectural rules that make the Phase-3 extension possible without rewriting Phase 1:

- **Chemistry never imports torch.** Reward workers must run without a model in the process.
- **Model code never imports chemistry.** Validity is an evaluation concern, not a layer of the network.
- **Dependencies point one way:** `rl/ → {model, tokenization, chemistry, generation, evaluation}`, never
  the reverse. The supervised code does not know RL exists.
- **The tokenizer is a hashed on-disk artifact**, identical across pretraining, fine-tuning, evaluation,
  generation, and future rollouts.
- **Checkpoints always carry their config and tokenizer identity.** A checkpoint without its configuration
  is not a checkpoint.

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
| [`docs/rlvr_plan.md`](docs/rlvr_plan.md) | The Phase-3 GRPO/RLVR design — **not implemented** |

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
