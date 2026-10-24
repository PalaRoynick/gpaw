#!/usr/bin/env bash
# Explicit opt-in system package installation for Debian/Ubuntu with OpenMPI.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo 'Usage: bash tools/deepdft/install_system_deps.sh'
    echo 'Installs signed distribution packages via apt; requires root or sudo.'
    exit 0
fi
source /etc/os-release
case "$ID" in
    debian|ubuntu) ;;
    *) echo 'Use the equivalent development packages for your distribution.' >&2; exit 1 ;;
esac
privilege=()
if [[ $(id -u) != 0 ]]; then
    command -v sudo >/dev/null || { echo 'Root or sudo is required.' >&2; exit 1; }
    privilege=(sudo)
fi
"${privilege[@]}" apt-get update
"${privilege[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential gfortran pkg-config git ca-certificates \
    python3-dev python3-venv libblas-dev liblapack-dev libxc-dev \
    libfftw3-dev libopenmpi-dev openmpi-bin libscalapack-openmpi-dev
