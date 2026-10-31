#!/usr/bin/env bash
# Large structures use tiled atom halos and a temporary disk grid.
set -euo pipefail
exec python -m gpaw.density_learning.cli predict --device "${DENSITY_DEVICE:-auto}" \
    --threads "${DENSITY_THREADS:-8}" "$@"
