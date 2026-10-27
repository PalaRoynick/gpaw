#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
exec python -m gpaw.density_learning.cli evaluate --device "${DENSITY_DEVICE:-cpu}" \
    --threads "${DENSITY_THREADS:-8}" "$@"
