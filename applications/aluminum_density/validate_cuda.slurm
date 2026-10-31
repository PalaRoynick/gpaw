#!/usr/bin/env bash
#SBATCH --job-name=gpaw-cuda-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=gpaw-cuda-%j.out
set -euo pipefail
# Adjust partition/account and GPU resource name to the site's configuration.
# Submit with GPAW_CUDA_BUILD pointing to the completed isolated CUDA build.
source "${GPAW_CUDA_BUILD:-$HOME/gpaw-cuda-v100}/activate.sh"
mkdir -p "$GPAW_CUDA_STATE/work"
cd "$GPAW_CUDA_STATE/work"
python "${GPAW_SOURCE_ROOT:?Set GPAW_SOURCE_ROOT to the updated checkout}/tools/deepdft/check_gpaw_cuda.py" \
  --require-v100 --output "$PWD/validation-$SLURM_JOB_ID" "$@"
