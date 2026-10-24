#!/usr/bin/env bash
# Provision a complete user-space toolchain, then build and check the sources.
# No sudo, apt, driver installation, or shell startup file edits.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mode=cpu
state_dir=
venv_dir=
toolchain=
conda_base=
mpi_ranks=2
while (($#)); do
    case "$1" in
        --mode) mode=${2:?}; shift 2 ;;
        --state-dir) state_dir=${2:?}; shift 2 ;;
        --env) venv_dir=${2:?}; shift 2 ;;
        --toolchain) toolchain=${2:?}; shift 2 ;;
        --conda-base) conda_base=${2:?}; shift 2 ;;
        --mpi-ranks) mpi_ranks=${2:?}; shift 2 ;;
        --help)
            echo 'Usage: build_user_linux.sh [--mode cpu|v100] [--state-dir DIR]'
            echo '       [--env DIR] [--toolchain DIR] [--conda-base DIR] [--mpi-ranks N]'
            echo 'Installs Python, compilers, MPI and numerical libraries into user space.'
            echo 'Requires Bash, Git, bootstrap Python >=3.8, network access and writable storage.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ $mode == cpu || $mode == v100 ]] || exit 2
[[ $mpi_ranks =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $(uname -m) == x86_64 ]] || { echo 'This profile requires Linux x86_64.' >&2; exit 1; }
[[ $(git -C "$root" branch --show-current) == DeepDFT ]] || {
    echo 'Check out DeepDFT first.' >&2; exit 1;
}
for cmd in python3 git timeout; do command -v "$cmd" >/dev/null; done
source "$root/tools/deepdft/isolate_toolchain.sh"
deepdft_clean_host_toolchain
python3 - <<'PY'
import platform
name, version = platform.libc_ver()
if name != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 28):
    raise SystemExit('This wheel profile requires glibc >= 2.28. Use a supported host or site-provided container; do not replace system glibc.')
PY
if [[ $mode == v100 ]]; then
    command -v nvidia-smi >/dev/null || {
        echo 'An administrator-provided NVIDIA driver and an allocated GPU are required.' >&2
        exit 1
    }
    nvidia-smi
fi
state_dir=${state_dir:-$root/.deepdft-build/user-$mode}
venv_dir=${venv_dir:-$root/venv-deepdft-user-$mode}
mkdir -p "$state_dir"
state_dir=$(cd "$state_dir" && pwd)
toolchain=${toolchain:-$state_dir/toolchain}
toolchain=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$toolchain")
export CONDA_REGISTER_ENVS=false
export XDG_CACHE_HOME="$state_dir/cache"
export CONDA_PKGS_DIRS="$state_dir/conda-pkgs"
export CONDA_ENVS_PATH="$state_dir/conda-envs"
export CONDARC="$state_dir/condarc.yml"
cat > "$CONDARC" <<'YAML'
channels:
  - conda-forge
channel_priority: strict
ssl_verify: truststore
register_envs: false
YAML
if [[ -z $conda_base ]]; then
    conda_base="$state_dir/miniforge"
    python3 "$root/tools/deepdft/bootstrap_miniforge.py" "$conda_base" --cache "$state_dir/bootstrap"
fi
conda_base=$(cd "$conda_base" && pwd)
[[ -x $conda_base/bin/conda && -f $conda_base/etc/profile.d/conda.sh ]] || {
    echo 'The selected Conda base must supply bin/conda and etc/profile.d/conda.sh.' >&2
    exit 1
}
# Do not adopt or alter a pre-existing unrelated Conda prefix.
if [[ -e $toolchain && ! -f $toolchain/.gpaw-deepdft-toolchain ]]; then
    echo 'Toolchain directory already exists and is not managed by this script.' >&2
    exit 1
fi
if [[ ! -d $toolchain ]]; then
    "$conda_base/bin/conda" create --yes --override-channels \
        --strict-channel-priority --channel conda-forge --prefix "$toolchain" \
        python=3.12 pip gcc_linux-64=13 gxx_linux-64=13 gfortran_linux-64=13 \
        openmpi=5 'libxc-c=6.2.2=cpu*' fftw=3.3.10 scalapack=2.2 'libblas=*=*openblas' \
        2>&1 | tee "$state_dir/toolchain-create.log"
    touch "$toolchain/.gpaw-deepdft-toolchain"
fi
"$conda_base/bin/conda" list --prefix "$toolchain" --explicit > "$state_dir/toolchain-explicit.txt"
"$conda_base/bin/conda" run --no-capture-output --prefix "$toolchain" bash "$root/tools/deepdft/build_linux.sh" \
    --mode "$mode" --python "$toolchain/bin/python" --dependency-prefix "$toolchain" \
    --env "$venv_dir" --state-dir "$state_dir" --mpi-ranks "$mpi_ranks"
# Restore Conda compiler/MPI activation as well as the Python venv in future shells.
python3 - "$state_dir/activate.sh" "$conda_base" "$toolchain" "$state_dir" "$root" <<'PY'
from pathlib import Path
from shlex import quote
import sys
path, base, toolchain, state, root = sys.argv[1:]
p = Path(path)
body = p.read_text()
helper = Path(root) / 'tools/deepdft/isolate_toolchain.sh'
header = (f'source {quote(str(helper))}\n'
          'deepdft_clean_host_toolchain\n'
          'export CONDA_REGISTER_ENVS=false\n'
          f'export CONDA_PKGS_DIRS={quote(state + "/conda-pkgs")}\n'
          f'export CONDA_ENVS_PATH={quote(state + "/conda-envs")}\n'
          f'export CONDARC={quote(state + "/condarc.yml")}\n'
          f'source {quote(base + "/etc/profile.d/conda.sh")}\n'
          f'conda activate {quote(toolchain)}\n')
p.write_text(header + body)
PY
echo "Rootless build passed. Activate with: source '$state_dir/activate.sh'"
