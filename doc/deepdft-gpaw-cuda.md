# Native GPAW CUDA build for NVIDIA V100

This is a separate, rootless GPAW build for PW calculations. The existing
`build_user_linux.sh --mode v100` profile supplies CPU/MPI GPAW and CUDA PyTorch;
it does not compile GPAW's GPU kernels. Keep that environment for FD labels and
PaiNN training. This new environment contains GPAW/CuPy, not the PyTorch stack.

The baseline is Linux x86_64, the existing rootless GCC 13/OpenMPI toolchain,
CUDA Toolkit 12.6.3, CuPy 13.6.0 (`cupy-cuda12x`), and explicit `sm_70` plus
`compute_70` code generation. CUDA 13 removed offline compilation for Volta.
The CUDA toolkit is installed in user space; the NVIDIA driver remains a host
prerequisite. The reported V100 driver 580.178.04 supports CUDA 12.6 applications.
The CUDA version shown by nvidia-smi is driver capability, not the installed nvcc.

## Build

First complete the rootless CPU or V100 build from [the Linux guide](deepdft-linux.md).
Use its state directory, containing `activate.sh`, `siteconfig.py`, and `toolchain`.
Then, from the DeepDFT checkout:

```bash
bash tools/deepdft/build_gpaw_cuda.sh \
  --base-state "$PWD/.deepdft-build/user-v100" \
  --state-dir "$HOME/gpaw-cuda-v100"
```

When no site `nvcc` is found, the script downloads selected CUDA 12.6 components from NVIDIA's 12.6.3 labeled Conda channel
and uses conda-forge for dependencies. It avoids the full toolkit metapackage
and its unrelated graphics dependencies. Allow several GB for CUDA, compilers,
wheels and build intermediates. The script never installs a driver or uses sudo.
The script captures a module-provided `nvcc` before activating its private MPI
toolchain. To use an existing complete CUDA 12.6-12.9 installation, append
`--cuda-root /path/to/cuda-12.6` (nvcc, headers, runtime and cuBLAS development
libraries are required). Do not use CUDA library stub directories at runtime.

For a faster build, set `GPAW_BUILD_JOBS=4` before invoking the script if four
CPU cores and sufficient RAM are allocated. The default build is serial.

Use a fresh state directory for each build. A source snapshot includes current
local edits and a SHA256 manifest. A wheel is built and installed into a separate
venv; the working CPU extension is not overwritten. Build provenance includes
siteconfig, compiler version, Python packages and the provisioned CUDA package
lock. There is no implicit commit, push or upstream modification.

Without an allocated GPU, append `--compile-only`. This still compiles GPU kernels
and runs a CPU PW check, but explicitly does not establish GPU readiness. A
normal build fails if hardware validation fails rather than accepting emulation.

## Activate and validate on the GPU node

```bash
source "$HOME/gpaw-cuda-v100/activate.sh"
mkdir -p "$HOME/work/gpaw-cuda"
cd "$HOME/work/gpaw-cuda"
python "$GPAW_CUDA_STATE/source/tools/deepdft/check_gpaw_cuda.py" \
  --require-v100 --output "$PWD/validation-001"
```

Run outside all GPAW source checkouts. Otherwise their Python modules and CPU
`_gpaw.so` can shadow the installed CUDA wheel. Use a new validation output path.
The check rejects fake CuPy, checks GPU capability and compares CPU/CUDA results
for displaced periodic Al using identical PW/PBE settings. Acceptance tolerances
are 1e-5 eV total energy, 2e-5 eV/angstrom maximum force error, and 1e-6 relative
L1 pseudo-density error. These are backend consistency tests, not proof that the
300 eV test cutoff describes converged physical references. Do not loosen the
tolerances to hide a failure. Timings of this small check are not a speed benchmark.

## Generate periodic reference densities

Use one allocated GPU and one process. Preserve scheduler CUDA_VISIBLE_DEVICES.

```bash
python -m gpaw.density_learning.cli reference --gpu \
  --input /your/work/periodic-structures.extxyz \
  --settings "$GPAW_CUDA_STATE/source/applications/aluminum_density/reference-pw-cuda.toml" \
  --output /your/work/pw-cuda-references
```

The wrapper rejects fake GPU execution, explicit multi-rank CUDA requests and
nonperiodic PW inputs. FD CUDA is separately available with the explicit
`--experimental-fd-gpu` opt-in, using the host-staged electrostatic adapter; see
the aluminum application README for hardware acceptance before using it. Do not convert an isolated system or slab into a fully
periodic system just to use this profile: that changes the physical problem.
Do not pass the mixed pilot file directly; select periodic structures first.
The exported pseudo density keeps the existing units and PAW convention.
GPU arrays are explicitly transferred to host memory before cube export.
CUDA execution is recorded in metadata separately from physical settings.

## Slurm example

A ready-to-edit job is also available at
`applications/aluminum_density/validate_cuda.slurm`.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=gpaw-cuda-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=gpaw-cuda-%j.out
set -euo pipefail
# Add the site's GPU partition and account as required.
source "$HOME/gpaw-cuda-v100/activate.sh"
mkdir -p "$HOME/work/gpaw-cuda"
cd "$HOME/work/gpaw-cuda"
python "$GPAW_CUDA_STATE/source/tools/deepdft/check_gpaw_cuda.py" \
  --require-v100 --output "$PWD/validation-$SLURM_JOB_ID"
```

Adapt the GPU resource name to the site's configuration. The allocation and
memory above suit a small validation, not arbitrary production structures.
If the site requires srun to enter a GPU job step, follow that site policy for
this single-rank check; do not nest mpiexec inside srun.

## MPI and scope

GPU-aware MPI is disabled at build and runtime by default; it must not be assumed
from a working CPU MPI installation. This alone does not guarantee every
multi-GPU algorithm works. The current upstream tests contain skips for PBE with
some multi-rank GPU decompositions without GPU-aware MPI. Therefore this wrapper
supports one GPU per reference job initially. Multiple independent structures can
use separate scheduler jobs/array elements with disjoint output directories.

PW is the established profile here. The application now offers an experimental
FD CUDA adapter with CPU Poisson corrections; it must pass the new bulk/slab/finite
acceptance checks on real hardware before labels are used. This build does not
certify multi-GPU PBE.
PAW datasets, wheels, toolkit, density cubes and validation logs remain outside Git.

## Conda timeouts and site CUDA modules

A repodata ReadTimeout occurs before compilation. Loading `ml nvidia` supplies
CUDA but does not make older script revisions automatically use it. With those
revisions, pass `--cuda-root` explicitly (only CUDA 12.6 is accepted there).
The revised script detects module-provided nvcc before private activation and
supports CUDA 12.6-12.9. It rejects CUDA 13 rather than attempting Volta compilation.
Use `--download-cuda` to explicitly bypass a site toolkit and provision CUDA 12.6.

Inspect the toolkit after loading the site module:

```bash
ml nvidia
command -v nvcc
nvcc --version
source tools/deepdft/select_cuda.sh
CUDA_SITE_ROOT=$(deepdft_select_cuda)
bash tools/deepdft/build_gpaw_cuda.sh \
  --cuda-root "$CUDA_SITE_ROOT" \
  --state-dir "$HOME/gpaw-cuda-v100-site" --compile-only
```

Choose a fresh state directory after a failed attempt; do not delete working
builds. Site-toolkit mode skips Conda CUDA downloads, but Python packages may
still need PyPI access or an existing pip cache. Site modules may contain several
CUDA versions; select the actual compatible directory, not an arbitrary SDK root.
The module's MPI is not used in place of the private OpenMPI build.

When provisioning is necessary, metadata connect/read timeouts default to 30/120
seconds and retries to 5 for that one command. Override them via the corresponding
`CONDA_REMOTE_CONNECT_TIMEOUT_SECS`, `CONDA_REMOTE_READ_TIMEOUT_SECS`, and
`CONDA_REMOTE_MAX_RETRIES` environment variables. Longer timeouts help a slow
connection; they do not repair blocked network access.

NVIDIA HPC SDK can expose `compilers/bin/nvcc` as a wrapper. Its parent directory
is not necessarily the CUDA development root. Discovery checks the resolved
compiler root and sibling `cuda/<version>` and `cuda` directories for runtime
headers and cudart development libraries. cuBLAS is checked separately during
preparation. If none qualifies, it fails
before downloads rather than guessing an incompatible toolkit.

## Split NVIDIA HPC SDK libraries

The runtime and cuBLAS need not share a directory. Preparation searches the CUDA
root plus sibling `math_libs/<CUDA major.minor>` and `math_libs` roots. It checks
`cuda_runtime.h`, `cublas_v2.h`, `libcudart.so`, and `libcublas.so` independently.
Selected library directories are saved in `cuda-library-paths.json` and used both
for linking/rpath and generated runtime activation. Missing files are reported by
name together with the toolkit root and searched directories. Versioned runtime
files alone are not accepted as a substitute for development linker symlinks.
An explicit `--cuda-root` pointing at HPC SDK `compilers` is normalized to the
sibling toolkit when that layout can be identified.

The toolkit root need not contain `bin/nvcc`: HPC SDK may keep the compiler in
sibling `compilers/bin/nvcc` while headers and runtime live in `cuda`. The selected
compiler path is used explicitly in siteconfig; CUDA_PATH refers to the header/
runtime root. These paths are distinct from the private GCC host compiler.


The reference, convergence and ML commands now accept `--device auto` by default;
CPU selection is explicit in reference progress events. `--device cuda` (or the
reference alias `--gpu`) fails instead of silently accepting CPU fallback.
The FD adapter is opt-in because real-device acceptance has not yet been run by
the developer in this environment. It preserves the configured FD basis,
boundaries and electrostatic corrections; it does not convert open systems to PW.
Run the hardware comparison separately for `--mode fd --family bulk`, `slab`,
and `finite`. A fake-CuPy FD calculation is not an acceptance substitute.


## Update Python adapters without rebuilding CUDA

After `git pull`, run with the installed CUDA environment's Python:

```bash
/path/to/gpaw-cuda-build/venv/bin/python \
  /path/to/gpaw/tools/deepdft/update_python_adapters.py --dry-run
/path/to/gpaw-cuda-build/venv/bin/python \
  /path/to/gpaw/tools/deepdft/update_python_adapters.py
```

Only `gpaw/density_learning/*.py` is updated. `_gpaw.so` and compiled kernels are
preserved and their hashes verified; replacements are backed up inside the venv.
Applications/settings/test scripts remain in the checkout, not in site-packages.
Use the checkout's `tools/deepdft/check_gpaw_cuda.py` to test new adapter features,
not an older copy under the build state's source snapshot. Use a new Python
process after updating and perform updates between calculation jobs.


For the confirmed FD localized-function derivative crash, Python-only core fixes
are available through `update_python_adapters.py --include-fd-force-fix`.
This explicitly adds `gpaw/lfc.py` and `gpaw/new/fd/pot_calc.py` to the updater's
allowlist; it does not replace `_gpaw.so`. The derivative calculation is staged
to host NumPy because the existing C derivative routine is CPU-only. The FD
compensation-force accumulator uses its execution backend consistently. Run the
full real-GPU energy/force/density comparisons again after installation; array
emulation and CPU tests do not establish V100 acceptance.
