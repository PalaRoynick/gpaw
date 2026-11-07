#!/usr/bin/env bash
#SBATCH --job-name=Al-PAW-elph-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=Al-elph-CPU-%j.out
set -euo pipefail
: "${GPAW_VENV:?Export the absolute path of your CPU GPAW virtual environment}"
source "$GPAW_VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the GPAW repository root}"
export GPAW_MPI_BACKEND=cgpaw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_root=${ELPH_RUN_ROOT:-"$PWD/runs"}
repeat=${ELPH_REPEAT:-2}
case "$repeat" in
    2) k_super=6 ;;
    3) k_super=4 ;;
    4) k_super=3 ;;
    *) echo 'ELPH_REPEAT must be 2, 3, or 4' >&2; exit 2 ;;
esac
output="$run_root/al-c$repeat"
export MPLCONFIGDIR="$run_root/.matplotlib-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$run_root/.cache-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python -m gpaw info
mpiexec -n "$SLURM_NTASKS" python tools/elph_al/conventional.py \
    --repeat "$repeat" --k-super "$k_super" --ecut 600 --h .16 \
    --output "$output"
python tools/elph_al/pw_reference.py "$output" --output "$output-pw"
python tools/elph_al/plot.py "$output-pw" \
    --reference tools/elph_al/reference/chaput-figure2.csv
python tools/elph_al/check_translation.py "$output-pw/path.gpw" \
    --output "$output-pw/translation-check.json"
