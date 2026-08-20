#!/bin/bash
#SBATCH --job-name=polyt5-lrg
#SBATCH --account=gts-REPLACE-ME              # your PACE allocation
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/large-%j.out
#SBATCH --error=logs/large-%j.err
#
# polyT5-large span-corruption pretraining on the full ~100M polyOne corpus.
#
# The paper found large only "marginal[ly]" better than medium and selected
# MEDIUM for every downstream analysis. Run this to test that claim, not as the
# default path -- it costs roughly 5x the GPU-hours of medium.
#
# Estimated ~1.5-2.5 days on one H100. The 48 h walltime will likely need at
# least one resume; that is expected and supported:
#     sbatch --export=ALL,RESUME=1 scripts/pace/pretrain_large.sbatch
set -euo pipefail

: "${SCRATCH:=$HOME/scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
VENV="${POLYT5_VENV:-$SCRATCH/polyt5-venv}"
DATA_ROOT="${POLYT5_DATA:-$SCRATCH/polyt5-data}"
RESULTS="${POLYT5_RESULTS:-$SCRATCH/polyt5-results}"
RUN_NAME="pretrain_large_polyone_100m"

module purge
module load python/3.12 || module load anaconda3
# shellcheck disable=SC1091
source "$VENV/bin/activate"

mkdir -p logs "$RESULTS"
nvidia-smi

RESUME_ARG=()
if [[ "${RESUME:-0}" == "1" ]]; then
    CKPT="$RESULTS/$RUN_NAME/checkpoints/best.pt"
    [[ -f "$CKPT" ]] && RESUME_ARG=(--resume "$CKPT")
    echo "resuming from ${CKPT:-<none found>}"
fi

# Large is ~7.9x medium's parameters. The config uses 225 x 2 for headroom;
# raise to 450 x 1 if a short probe shows memory to spare. Either way
# target_effective_batch_size keeps the product honest at the paper's 450.
srun python "$PROJECT_ROOT/scripts/pretrain.py" \
    --config "$PROJECT_ROOT/configs/pretrain/polyone_large_h100.yaml" \
    "${RESUME_ARG[@]}" \
    --set \
        experiment_name="$RUN_NAME" \
        data.corpus_prefix="$DATA_ROOT/processed/polyone_full/corpus" \
        output.results_root="$RESULTS" \
        train.num_workers="$((SLURM_CPUS_PER_TASK - 2))"

echo "results in $RESULTS/$RUN_NAME"
