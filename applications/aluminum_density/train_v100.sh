#!/usr/bin/env bash
# One allocated V100 per training process. Preserve scheduler GPU visibility.
set -euo pipefail
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
    raise SystemExit('This launcher requires an allocated V100 (compute capability 7.0)')
if 'sm_70' not in torch.cuda.get_arch_list():
    raise SystemExit('Install the tested PyTorch cu118 wheel with sm_70 support')
print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)
PY
exec python -m gpaw.density_learning.cli train --device cuda --threads "${DENSITY_THREADS:-4}" "$@"
