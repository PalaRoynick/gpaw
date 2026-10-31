#!/usr/bin/env bash
set -euo pipefail
exec python -m gpaw.density_learning.cli evaluate --device "${DENSITY_DEVICE:-auto}" \
    --threads "${DENSITY_THREADS:-8}" "$@"
