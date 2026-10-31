#!/usr/bin/env bash
# Select a real CUDA PyTorch backend when available, with explicit CPU fallback.
set -euo pipefail
exec python -m gpaw.density_learning.cli train \
    --device "${DENSITY_DEVICE:-auto}" --threads "${DENSITY_THREADS:-4}" "$@"
