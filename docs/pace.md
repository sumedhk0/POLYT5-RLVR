# Running polyT5 pretraining on Georgia Tech PACE

Everything except pretraining runs comfortably on a laptop GPU. Pretraining at the paper's scale does not.
This document covers moving that one stage to PACE (Slurm, H100 nodes) and bringing the checkpoint back.

## Why bother

Measured on an RTX 4080 Laptop (12 GB): polyT5-medium trains at ~3.0 optimizer steps/s at effective batch
450, i.e. ~1,350 sequences/s. Extrapolating to the paper's corpus scale:

| Run | 4080 Laptop (12 GB) | 1× H100 (80 GB), estimated |
|---|---|---|
| Corpus prep to ~100M (CPU only) | ~3.5 h | ~2–3 h on 24 cores |
| **medium**, 2 epochs @ ~100M | **~37 h** | **~7–8 h** |
| large, 2 epochs @ ~100M | ~8.5 days | ~1.5–2.5 days |

Two things make the H100 gain larger than raw FLOPs suggest:

1. **80 GB holds the paper's batch of 450 as a single physical batch.** On 12 GB it must be factored
   (150 × 3), and the accumulation loop costs Python overhead per micro-batch.
2. **polyT5 is small.** A 7.5M-parameter model at sequence length 200 underuses tensor cores; larger
   batches recover much of that, and only a big-memory card allows them.

**Before spending large-model hours, read this:** the paper reports that polyT5-large "showed marginal
improvements over the medium variant" and that they "selected the polyT5-medium model for subsequent
analyses" on accuracy/cost grounds. Every headline number in the paper — Tg RMSE 40.82, the generation
sweep, the dielectric screening cascade — is **medium**. Matching the paper's best model means medium.
Large is worth running only to test that claim.

## What lives where

| Stage | Where | Why |
|---|---|---|
| Corpus download + prep | PACE (CPU job) or locally | CPU-bound; ~9 GB in, ~17 GB out |
| Span-corruption pretraining | **PACE (H100)** | The only stage that needs it |
| Tg fine-tuning (30 epochs) | Laptop | ~10 min |
| Generation + evaluation + sweep | Laptop | minutes to hours |
| The demo app | Laptop | interactive |

So the only artifact that needs to cross the boundary is one checkpoint file (~30 MB for medium,
~240 MB for large), plus the tokenizer artifact, which is tiny and already in git.

## One-time setup

On a **login** node:

```bash
git clone <your-repo> ~/polyt5-rlvr && cd ~/polyt5-rlvr
bash scripts/pace/setup_env.sh
```

That builds a venv in `$SCRATCH/polyt5-venv` (not `$HOME` — the home quota is small), installs a CUDA 12.x
torch wheel, and prints the detected device so a CPU-only wheel cannot slip through unnoticed.

Then edit the allocation line in each sbatch file:

```bash
#SBATCH --account=gts-REPLACE-ME     # e.g. gts-<PI-username>
```

## Getting the data there

The polyOne corpus is ~9 GB of raw text. Either re-download it on PACE (simplest, it is a public Zenodo
record) or copy what you already have:

```bash
# option A — fetch on PACE, into $SCRATCH
python scripts/download_data.py --dataset polyone_train --dataset polyone_dev \
    --dest "$SCRATCH/polyt5-data/external" --yes-large

# option B — copy from the laptop
rsync -avP "C:/Users/sumedh/polyt5-data/external/" \
    <user>@login-phoenix.pace.gatech.edu:/storage/scratch1/<n>/<user>/polyt5-data/external/
```

Either way the download script writes a provenance sidecar with URL, byte count and SHA-256, so the corpus
version is pinned and checkable on both machines.

## Submitting

```bash
mkdir -p logs
sbatch scripts/pace/prepare_corpus.sbatch          # CPU only, no GPU requested
sbatch scripts/pace/pretrain_medium.sbatch         # 1x H100
# optional, only to test the medium-vs-large claim:
sbatch scripts/pace/pretrain_large.sbatch
```

`prepare_corpus.sbatch` deliberately requests **no GPU** — it is pure RDKit and tokenizer work, and holding
an H100 idle for hours would be waste.

### Walltime and resuming

Both training jobs checkpoint every epoch and resume cleanly:

```bash
sbatch --export=ALL,RESUME=1 scripts/pace/pretrain_medium.sbatch
```

Resume restores model, optimizer, scheduler, epoch, global step **and RNG state**, so a resumed run
reproduces an uninterrupted one — that equivalence is asserted by a test
(`tests/test_training.py`, 4 steps + checkpoint + 4 steps ≡ 8 uninterrupted steps).

Corpus preparation is chunk-checkpointed too, so a walltime kill mid-prep is resumable with `--resume`
rather than starting over.

## Bringing the checkpoint home

```bash
rsync -avP <user>@login-phoenix.pace.gatech.edu:/storage/scratch1/<n>/<user>/polyt5-results/pretrain_medium_polyone_100m/checkpoints/best.pt \
    "C:/Users/sumedh/polyt5-data/checkpoints/"
```

Then fine-tune locally exactly as before:

```bash
python scripts/finetune.py --task prediction \
    --init-checkpoint C:/Users/sumedh/polyt5-data/checkpoints/best.pt
```

The checkpoint carries its own `model_config` and `tokenizer_sha256`, and `finetune.py` **refuses** to
fine-tune across a vocabulary mismatch. That guard has already caught one real mismatch in this project, so
a checkpoint trained on PACE against a different tokenizer build will fail loudly rather than silently
producing wrong token ids.

## Sizing notes

- `physical_batch_size × gradient_accumulation_steps` must equal `target_effective_batch_size` (450), and
  `TrainerConfig` raises naming both numbers if it does not. Change the factorisation freely per node; do
  not change the product without recording it.
- The H100 medium config uses **450 × 1** — the paper's batch, unfactored.
- The large config uses 225 × 2 for headroom. Probe with `--max-steps 50` and check reported peak VRAM
  before committing to a multi-day run; raise to 450 × 1 if there is room.
- `--set` overrides anything from the command line, so node-specific tuning never requires editing YAML:
  `--set train.physical_batch_size=300 train.gradient_accumulation_steps=1 ...` (but then also set
  `train.target_effective_batch_size` to match, on purpose).

## What does *not* change on PACE

The reproduction's substance is identical: same tokenizer artifact and hash, same span-corruption
implementation with the paper's invariants asserted, same architecture configs from Table S2, same
evaluation. The only difference is wall-clock and the batch factorisation, both of which are recorded in
each run's `manifest.json` alongside the device description.
