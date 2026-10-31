#!/usr/bin/env bash
#SBATCH --job-name=al-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=al-reference-%j.out
set -euo pipefail
# Set GPAW_CUDA_BUILD; pass absolute --input, --settings and --output paths.
# Adjust partition/account and GPU resource spelling for the site.
source "${GPAW_CUDA_BUILD:?Set GPAW_CUDA_BUILD}/activate.sh"
mkdir -p "$GPAW_CUDA_STATE/work"
cd "$GPAW_CUDA_STATE/work"
exec python -m gpaw.density_learning.cli reference "$@"
