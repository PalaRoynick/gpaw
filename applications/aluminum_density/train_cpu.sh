#!/usr/bin/env bash
# Multicore CPU fallback using PyTorch CPU threads.
set -euo pipefail
exec python -m gpaw.density_learning.cli train --device cpu --threads "${DENSITY_THREADS:-8}" "$@"
