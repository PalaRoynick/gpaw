#!/usr/bin/env bash
# Build V100 GPAW in a separate wheel/environment without sudo or CPU overwrite.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
base=$root/.deepdft-build/user-v100
state=$root/.deepdft-build/gpaw-cuda-v100
cuda=
compile_only=0
download_cuda=0
while (($#)); do
    case "$1" in
        --base-state) base=${2:?}; shift 2 ;;
        --state-dir) state=${2:?}; shift 2 ;;
        --cuda-root) cuda=${2:?}; shift 2 ;;
        --download-cuda) download_cuda=1; shift ;;
        --compile-only) compile_only=1; shift ;;
        --help)
            echo 'Usage: build_gpaw_cuda.sh [--base-state DIR] [--state-dir NEW_DIR]'
            echo '       [--cuda-root CUDA_12_DIR | --download-cuda] [--compile-only]'
            echo 'Requires a completed rootless CPU/v100 build. Prefers nvcc on PATH; otherwise provisions CUDA 12.6 components.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -z $cuda || $download_cuda == 0 ]] || {
    echo 'Use --cuda-root or --download-cuda, not both.' >&2; exit 2;
}
source "$root/tools/deepdft/select_cuda.sh"
# Capture the module-provided toolkit before activation sanitizes MPI paths.
if (( ! download_cuda )); then
    cuda=$(deepdft_select_cuda "$cuda")
fi
if [[ -n $cuda ]]; then
    deepdft_check_cuda_version "$cuda"
fi
base=$(cd "$base" && pwd)
[[ -f $base/activate.sh && -f $base/toolchain/.gpaw-deepdft-toolchain ]] || {
    echo 'Select a completed rootless build state with its private toolchain.' >&2; exit 2;
}
[[ ! -e $state ]] || { echo 'Choose a new state directory; existing builds are preserved.' >&2; exit 2; }
mkdir -p "$state"
state=$(cd "$state" && pwd)
source "$base/activate.sh"
exec > >(tee "$state/build.log") 2>&1
unset GPAW_CPUPY GPAW_USE_GPUS
export GPAW_NO_GPU_MPI=1
if [[ -z $cuda ]]; then
    cuda=$state/cuda
    # Bound slow metadata requests without changing the user's Conda config.
    CONDA_REMOTE_CONNECT_TIMEOUT_SECS=${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-30} \
    CONDA_REMOTE_READ_TIMEOUT_SECS=${CONDA_REMOTE_READ_TIMEOUT_SECS:-120} \
    CONDA_REMOTE_MAX_RETRIES=${CONDA_REMOTE_MAX_RETRIES:-5} \
    "$CONDA_EXE" create --yes --prefix "$cuda" --override-channels \
        --channel nvidia/label/cuda-12.6.3 --channel conda-forge \
        cuda-version=12.6 cuda-nvcc=12.6 cuda-cudart-dev=12.6 \
        cuda-nvrtc=12.6 cuda-cccl=12.6 libcublas-dev libcufft libcurand \
        libcusolver libcusparse
    "$CONDA_EXE" list --prefix "$cuda" --explicit > "$state/cuda-explicit.txt"
fi
cuda=$(cd "$cuda" && pwd)
deepdft_check_cuda_version "$cuda" | tee "$state/nvcc-version.txt"
host=$(command -v "${CXX:-g++}")
"$host" --version > "$state/host-compiler-version.txt"
git -C "$root" rev-parse HEAD > "$state/source-commit.txt"
python "$root/tools/deepdft/prepare_cuda_build.py" "$root" "$state" "$base" "$cuda" "$host"
"$base/toolchain/bin/python" -m venv "$state/venv"
python_bin=$state/venv/bin/python
"$python_bin" -m pip install 'pip==25.1.1' 'setuptools==80.9.0' 'wheel==0.45.1' \
    'pybind11==3.0.1' 'numpy==2.2.6' 'scipy==1.15.3' 'ase==3.29.0' \
    'gpaw-data==1.2.1' 'cupy-cuda12x==13.6.0'
export CUDA_PATH="$cuda"
export PATH="$cuda/bin:$PATH"
export LD_LIBRARY_PATH="$cuda/lib64:$cuda/lib:$cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH-}"
export GPAW_CONFIG="$state/siteconfig.py"
cd "$state/source"
"$python_bin" -m pip wheel --verbose --no-build-isolation --no-deps --wheel-dir "$state/wheels" .
"$python_bin" -m pip install --no-deps "$state"/wheels/gpaw-*.whl
"$python_bin" -m pip check
"$python_bin" -m pip freeze > "$state/python-packages.txt"
"$python_bin" - "$base" "$state" "$cuda" <<'PY'
from pathlib import Path
from shlex import quote as q
import sys
base, state, cuda = sys.argv[1:]
lines = [f'source {q(base + "/activate.sh")}',
         f'source {q(state + "/venv/bin/activate")}',
         f'export GPAW_CONFIG={q(state + "/siteconfig.py")}',
         f'export CUDA_PATH={q(cuda)}',
         f'export PATH={q(cuda + "/bin")}:"$PATH"',
         f'export LD_LIBRARY_PATH={q(cuda + "/lib64:" + cuda + "/lib:" + cuda + "/targets/x86_64-linux/lib")}:"${{LD_LIBRARY_PATH-}}"',
         'export GPAW_NO_GPU_MPI=1',
         'unset GPAW_CPUPY GPAW_USE_GPUS',
         f'export GPAW_CUDA_STATE={q(state)}']
Path(state, 'activate.sh').write_text('\n'.join(lines) + '\n')
PY
source "$state/activate.sh"
mkdir "$state/work"
cd "$state/work"
python - <<'PY'
import gpaw.cgpaw as c
assert hasattr(c, 'gpaw_gpu_init'), 'CUDA kernels are missing from the extension'
print('CUDA extension compiled; hardware execution is not established by this check.')
PY
python "$state/source/tools/deepdft/check_gpaw_cuda.py" --cpu-only --output "$state/work/cpu"
if ((compile_only)); then
    echo 'BUILD ONLY: GPU validation was not run. Run check_gpaw_cuda.py on an allocated V100.'
else
    python "$state/source/tools/deepdft/check_gpaw_cuda.py" --require-v100 --output "$state/work/cpu-gpu"
fi
echo "Activate with: source '$state/activate.sh'"
echo 'Run outside any GPAW source checkout to avoid shadowing the installed CUDA wheel.'
