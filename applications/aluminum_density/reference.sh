#!/usr/bin/env bash
# Run one MPI reference job on a shared filesystem, from this source checkout.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
exec mpiexec -n "${GPAW_RANKS:-4}" gpaw python -m gpaw.density_learning.cli reference \
    --settings "$root/applications/aluminum_density/reference.toml" "$@"
