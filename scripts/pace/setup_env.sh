#!/bin/bash
# One-time environment setup for polyt5-rlvr on Georgia Tech PACE (Phoenix).
#
# Run this on a LOGIN node once, before submitting any jobs:
#     bash scripts/pace/setup_env.sh
#
# It builds a self-contained virtualenv in $SCRATCH so that compute nodes never
# touch your home quota, which is small and shared. Everything the training code
# needs is pip-installable; there is no compiled extension to build.
set -euo pipefail

# PACE sets $SCRATCH; fall back to a sane path if running elsewhere.
: "${SCRATCH:=$HOME/scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="${POLYT5_VENV:-$SCRATCH/polyt5-venv}"

echo "project : $PROJECT_ROOT"
echo "venv    : $VENV"

# PACE ships several Python builds; 3.11/3.12 both work. Adjust if `module
# avail python` shows different names on your allocation.
module purge
module load python/3.12 || module load anaconda3
module load cuda/12.4 || echo "note: no cuda module loaded; the torch wheel bundles its own runtime"

if [[ ! -d "$VENV" ]]; then
    python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel

# CUDA 12.x wheel. H100 is compute capability 9.0 and needs cu12x; do not fall
# back to the CPU wheel, which will silently train ~100x slower.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

python -m pip install -e "$PROJECT_ROOT[dev,track]"

python - <<'PY'
import torch
print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(0))
    print("capability :", torch.cuda.get_device_capability(0))
    print("bf16       :", torch.cuda.is_bf16_supported())
PY

echo
echo "Done. Submit jobs with:  sbatch scripts/pace/pretrain_medium.sbatch"
echo "Remember to set your allocation in the sbatch files: #SBATCH -A gts-<PI>-<tier>"
