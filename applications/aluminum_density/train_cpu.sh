#!/usr/bin/env bash
# Multicore CPU fallback using PyTorch CPU threads.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
exec python -m gpaw.density_learning.cli train --device cpu --threads "${DENSITY_THREADS:-8}" "$@"
