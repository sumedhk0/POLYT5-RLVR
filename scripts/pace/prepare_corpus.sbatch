#!/bin/bash
#SBATCH --job-name=polyt5-prep
#SBATCH --account=gts-REPLACE-ME              # your PACE allocation, e.g. gts-rramprasad3
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/prep-%j.out
#SBATCH --error=logs/prep-%j.err
#
# CPU-only: convert raw polyOne PSMILES into a pre-tokenized memmap corpus.
# No GPU is requested on purpose — this stage is pure RDKit + tokenizer work,
# and asking for a GPU would waste an H100 for hours.
#
# Measured locally at 8,100 rows/s on 18 workers, so ~100M rows lands around
# 3.5 h. 24 cores here should beat that; the 6 h walltime is deliberate slack.
# The script checkpoints every chunk, so a walltime kill is resumable with
# --resume rather than fatal.
set -euo pipefail

: "${SCRATCH:=$HOME/scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
VENV="${POLYT5_VENV:-$SCRATCH/polyt5-venv}"
DATA_ROOT="${POLYT5_DATA:-$SCRATCH/polyt5-data}"

module purge
module load python/3.12 || module load anaconda3
# shellcheck disable=SC1091
source "$VENV/bin/activate"

mkdir -p "$DATA_ROOT/processed/polyone_full" logs

# Both polyOne shards together are ~100M polymers, essentially matching the
# paper's 100,145,883. Feed them as two --input flags so they land in one corpus.
srun python "$PROJECT_ROOT/scripts/prepare_large_corpus.py" \
    --input "$DATA_ROOT/external/generated_polymer_smiles_train.txt" \
    --input "$DATA_ROOT/external/generated_polymer_smiles_dev.txt" \
    --output-prefix "$DATA_ROOT/processed/polyone_full/corpus" \
    --tokenizer "$PROJECT_ROOT/artifacts/tokenizer/polyt5_vocab.json" \
    --workers "$((SLURM_CPUS_PER_TASK - 2))" \
    --dedup exact \
    --splits 0.9 0.05 0.05 \
    --seed 0 \
    --report-every 500 \
    --resume

echo "corpus written to $DATA_ROOT/processed/polyone_full/"
