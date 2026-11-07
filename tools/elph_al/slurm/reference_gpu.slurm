#!/usr/bin/env bash
#SBATCH --job-name=Al-PAW-elph-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=Al-elph-GPU-%j.out
set -euo pipefail
: "${GPAW_VENV:?Export the absolute path of your CUDA GPAW virtual environment}"
source "$GPAW_VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the GPAW repository root}"
export GPAW_MPI_BACKEND=cgpaw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GPAW_CPUPY
run_root=${ELPH_RUN_ROOT:-"$PWD/runs"}
repeat=${ELPH_REPEAT:-2}
case "$repeat" in
    2) k_super=6 ;;
    3) k_super=4 ;;
    4) k_super=3 ;;
    *) echo 'ELPH_REPEAT must be 2, 3, or 4' >&2; exit 2 ;;
esac
output="$run_root/al-c$repeat-gpu"
export MPLCONFIGDIR="$run_root/.matplotlib-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$run_root/.cache-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
nvidia-smi
python - <<'PY'
import cupy as cp
assert cp.cuda.runtime.getDeviceCount() >= 1
print(cp.cuda.runtime.getDeviceProperties(0)['name'])
PY
python -m gpaw info
python tools/elph_al/check_backend_gauge.py --gpu \
    --output "$output-gauge-check"
# SLURM sets CUDA_VISIBLE_DEVICES; do not replace it with a physical index.
python tools/elph_al/conventional.py --gpu-displacements \
    --repeat "$repeat" --k-super "$k_super" --ecut 600 --h .16 \
    --output "$output"
# Postprocessing intentionally uses the CPU PW diagnostic.
python tools/elph_al/pw_reference.py "$output" --output "$output-pw"
python tools/elph_al/plot.py "$output-pw" \
    --reference tools/elph_al/reference/chaput-figure2.csv
python tools/elph_al/check_translation.py "$output-pw/path.gpw" \
    --output "$output-pw/translation-check.json"
