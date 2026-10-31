#!/usr/bin/env bash
# Diagnose one-node MPI launch in the allocated job shell, never under srun/mpiexec.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
state=${1:-$root/.deepdft-build/user-v100}
[[ -f "$state/activate.sh" ]] || { echo "Missing activation: $state/activate.sh" >&2; exit 2; }
# Do not sanitize a live MPI process or nest another launcher inside it.
for name in OMPI_COMM_WORLD_RANK PMI_RANK PMIX_RANK; do
    if [[ -v $name ]]; then
        echo "Run this diagnostic from the allocation's job shell, outside an MPI/srun task ($name is set)." >&2
        exit 2
    fi
done
printf 'Launcher before activation: %s\n' "$(command -v mpiexec || true)"
source "$state/activate.sh"
hash -r
launcher=$(command -v mpiexec)
if [[ -x "$state/toolchain/bin/mpiexec" ]]; then
    launcher=$(cd -- "$state/toolchain/bin" && pwd)/mpiexec
fi
printf 'Selected launcher: %s\n' "$launcher"
"$launcher" --version
command -v python gpaw
# Report only allocation metadata; never dump the environment.
for name in SLURM_JOB_ID SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE PBS_NP; do
    printf '%s=%s\n' "$name" "${!name-unset}"
done
extension=$(python -c 'import _gpaw; print(_gpaw.__file__)')
printf 'Extension: %s\n' "$extension"
ldd "$extension" | awk '/libmpi|libpmix|libpmi|libopen-pal|libprrte|not found/'
# No oversubscription or scheduler override: require a real two-task allocation.
# Assertion prevents two successful singleton exits from being reported as MPI success.
timeout 60s "$launcher" -n 2 gpaw python -c \
  'from gpaw.mpi import world; print("rank", world.rank, "size", world.size, flush=True); assert world.size == 2, "Expected two communicating MPI ranks, not singletons"'
