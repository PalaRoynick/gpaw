#!/usr/bin/env bash
# Repeat functional checks without reinstalling the environment.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
state_dir="$root/.deepdft-build/cpu"
mode=cpu
mpi_ranks=2
while (($#)); do
    case "$1" in
        --state-dir) state_dir=${2:?}; shift 2 ;;
        --mode) mode=${2:?}; shift 2 ;;
        --mpi-ranks) mpi_ranks=${2:?}; shift 2 ;;
        --help) echo 'Usage: check_linux.sh [--state-dir DIR] [--mode cpu|v100] [--mpi-ranks N]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ $mode == cpu || $mode == v100 ]] || exit 2
[[ $mpi_ranks =~ ^[1-9][0-9]*$ ]] || exit 2
state_dir=$(cd "$state_dir" && pwd)
source "$state_dir/activate.sh"
cd "$root"
# Refuse incomplete installs rather than accepting a pytest run with skips.
python - <<'PY'
import torch, asap3, lz4.frame, gpaw_data  # noqa: F401
from pathlib import Path
assert Path('third_party/DeepDFT/densitymodel.py').is_file(), 'Initialize the submodule'
assert Path('third_party/DeepDFT/pretrained_models/qm9_painn/best_model.pth').is_file()
PY
mkdir -p "$state_dir/work"
# GPAW's CLI writes test.txt; keep calculation outputs outside source directories.
(
    cd "$state_dir/work"
    gpaw info
    timeout 180s gpaw test
) 2>&1 | tee "$state_dir/gpaw-serial.log"
if [[ $(id -u) == 0 ]]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
(
    cd "$state_dir/work"
    timeout 180s mpiexec --bind-to none -n "$mpi_ranks" gpaw test
) 2>&1 | tee "$state_dir/gpaw-mpi.log"
python - "$state_dir/gpaw-mpi.log" "$mpi_ranks" <<'PY'
from pathlib import Path
import sys
assert f'cores: {sys.argv[2]}' in Path(sys.argv[1]).read_text(), 'MPI rank count mismatch'
PY
python -m pytest -q gpaw/test/deepdft/test_smoke.py \
    --junitxml="$state_dir/smoke.xml" 2>&1 | tee "$state_dir/smoke.log"
python - "$state_dir/smoke.xml" <<'PY'
import sys
from xml.etree import ElementTree as ET
cases = ET.parse(sys.argv[1]).findall('.//testcase')
assert len(cases) == 2, f'Expected 2 inference tests, got {len(cases)}'
assert all(c.find(tag) is None for c in cases
           for tag in ['failure', 'error', 'skipped']), 'An inference test failed or was skipped'
PY
device=cpu
gpu_args=()
if [[ $mode == v100 ]]; then device=cuda; gpu_args=(--require-v100); fi
python tools/deepdft/check_training.py --device "$device" "${gpu_args[@]}" \
    --report "$state_dir/training.json"
echo "All $mode checks passed; reports: $state_dir"
