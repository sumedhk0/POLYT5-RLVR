#!/bin/bash
#SBATCH --job-name=polyt5-med
#SBATCH --account=gts-REPLACE-ME              # your PACE allocation
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/medium-%j.out
#SBATCH --error=logs/medium-%j.err
#
# polyT5-medium span-corruption pretraining on the full ~100M polyOne corpus.
# This is the run that matches the paper's headline model: every number the
# paper reports downstream (Tg RMSE 40.82, the generation sweep, the dielectric
# screen) comes from polyT5-medium, which the authors selected over large on
# accuracy/cost grounds.
#
# Estimated ~7-8 h on one H100 based on 4080-Laptop measurements scaled by
# throughput; the 12 h walltime is slack. If it is killed at walltime, resume:
#     sbatch --export=ALL,RESUME=1 scripts/pace/pretrain_medium.sbatch
set -euo pipefail

: "${SCRATCH:=$HOME/scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
VENV="${POLYT5_VENV:-$SCRATCH/polyt5-venv}"
DATA_ROOT="${POLYT5_DATA:-$SCRATCH/polyt5-data}"
RESULTS="${POLYT5_RESULTS:-$SCRATCH/polyt5-results}"
RUN_NAME="pretrain_medium_polyone_100m"

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

# An H100 has 80 GB, so the paper's batch of 450 fits as ONE physical batch --
# no gradient accumulation, which is the closest we can get to their setup.
# Keep target_effective_batch_size so a mis-set factorisation is caught.
srun python "$PROJECT_ROOT/scripts/pretrain.py" \
    --config "$PROJECT_ROOT/configs/pretrain/polyone_medium_h100.yaml" \
    "${RESUME_ARG[@]}" \
    --set \
        experiment_name="$RUN_NAME" \
        data.corpus_prefix="$DATA_ROOT/processed/polyone_full/corpus" \
        output.results_root="$RESULTS" \
        train.num_workers="$((SLURM_CPUS_PER_TASK - 2))"

echo "results in $RESULTS/$RUN_NAME"
