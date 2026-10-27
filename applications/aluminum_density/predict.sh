#!/usr/bin/env bash
# Large structures use tiled atom halos and a temporary disk grid.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
exec python -m gpaw.density_learning.cli predict --device "${DENSITY_DEVICE:-cpu}" \
    --threads "${DENSITY_THREADS:-8}" "$@"
