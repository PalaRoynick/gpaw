#!/usr/bin/env bash
# One rank per GPU job; multiple ranks explicitly select a CPU/MPI fallback.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ranks=${GPAW_RANKS:-1}
[[ $ranks =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid MPI rank count' >&2; exit 2; }
command=(python -m gpaw.density_learning.cli reference
         --settings "$root/applications/aluminum_density/reference.toml")
if ((ranks == 1)); then
    exec "${command[@]}" "$@"
fi
exec mpiexec -n "$ranks" gpaw python -m gpaw.density_learning.cli reference \
    --settings "$root/applications/aluminum_density/reference.toml" "$@"
