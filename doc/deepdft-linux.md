# Building GPAW + DeepDFT on Linux

This guide applies exclusively to **PalaRoynick/gpaw, branch `DeepDFT`**.
All committed documentation, comments, examples, messages, and filenames must
be in English. These build scripts never commit, push, create PRs/MRs, notify
maintainers, or change the external DeepDFT source.

The recommended route requires **no sudo**: install Python, compilers, MPI and
numerical libraries into user-writable directories, then compile GPAW and ASAP3
from source. DeepDFT is a pinned source submodule. PyTorch uses official binary
wheels, with separate CPU and NVIDIA V100 training profiles.

This is an initial integration environment, not a DeepDFT-backed GPAW
calculator, SCF initializer, or validated ML energy/force/MD potential.

## 1. Requirements and scope

| Component | Requirement |
|---|---|
| Platform | Linux x86_64; glibc >= 2.28 for the selected PyTorch wheels |
| Bootstrap | Bash, Git, Python >= 3.8 with working HTTPS certificates, and `timeout` |
| Storage | User-writable executable filesystem; about 10 GiB for CPU, 15 GiB for CUDA including caches |
| Memory | At least 8 GiB recommended for dependency resolution and compilation |
| Rootless toolchain | Python 3.12, GCC/G++/GFortran 13, OpenMPI 5, LibXC C 6.2.2, FFTW, ScaLAPACK, OpenBLAS |
| Python packages | Pinned in `tools/deepdft/requirements-linux.txt`; PyTorch 2.7.1 installed separately |
| V100 host prerequisites | An allocated V100 and a working administrator-provided NVIDIA driver |

Rootless installation does not need system Python development headers, a
system compiler, `apt`, a CUDA toolkit, or `nvcc`. It cannot install a kernel
GPU driver, grant GPU access, bypass a scheduler, or repair an unsupported host
OS. A newer driver retaining V100 support can run the selected CUDA 11.8 wheel.

Choose paths outside read-only or `noexec` filesystems. On a cluster, check
scratch quotas and retention policies before storing environments there.
Package resolution requires outbound HTTPS access to GitHub/release assets,
`conda.anaconda.org`, `pypi.org`, `files.pythonhosted.org`,
`download.pytorch.org`, and its official download CDN.

## 2. Clone the personal repository

```bash
git clone --branch DeepDFT --recurse-submodules https://github.com/PalaRoynick/gpaw.git
cd gpaw
git branch --show-current
git submodule status
```

For an existing checkout:

```bash
git checkout DeepDFT
git submodule update --init --recursive
```

Repeat submodule initialization after receiving a commit that changes its
revision. Do not use `git submodule update --remote`: use the exact dependency
commit recorded by this repository. The external MIT-licensed repository
contains approximately 129 MiB of pretrained checkpoints; GPAW tracks only its
Git submodule reference, not copied model binaries.

## 3. Recommended: automatic installation without sudo

From the checkout root, run:

```bash
bash tools/deepdft/build_user_linux.sh --mode cpu
source .deepdft-build/user-cpu/activate.sh
```

The script performs the following steps:

1. Downloads the pinned official Miniforge 25.3.0-3 installer, verifies its
   SHA-256 **before executing it**, and installs it noninteractively under the
   state directory. It never executes an unverified network response.
2. Creates a private Conda-forge toolchain containing Python, C/C++/Fortran
   compilers, OpenMPI, LibXC, FFTW, ScaLAPACK and OpenBLAS.
3. Creates an isolated Python venv based on that toolchain's Python.
4. Installs pinned build tools, the selected PyTorch wheel, and Python
   dependencies. ASAP3 and the current GPAW checkout are built from source.
5. Runs `pip check`, serial and two-rank MPI GPAW calculations, two pretrained
   inference checks, and three real Adam training steps for both SchNet/PaiNN.
6. Writes activation instructions, package manifests and numerical test logs.

It never invokes sudo or `apt`, installs to `/usr`, edits shell startup files,
or overwrites an unrelated existing environment. TLS and package checksums
remain enabled. Existing managed environments are reused on subsequent runs.
Compilers are used conservatively without aggressive parallel compilation.

The default locations are:

- Toolchain, cache and logs: `.deepdft-build/user-cpu/`.
- Python venv: `venv-deepdft-user-cpu/`.
- Activation: `.deepdft-build/user-cpu/activate.sh`.

For a server with separate user storage:

```bash
bash tools/deepdft/build_user_linux.sh \
  --mode cpu \
  --toolchain /your/writable/path/gpaw-toolchain \
  --env /your/writable/path/gpaw-python-cpu \
  --state-dir /your/writable/path/gpaw-build-cpu \
  --mpi-ranks 2

source /your/writable/path/gpaw-build-cpu/activate.sh
```

Replace the example paths with your actual user-writable directories. If an
administrator already provides a recent Conda installation, add
`--conda-base /path/to/conda-base` to skip the bootstrap download. The script creates its own toolchain; it does
not modify an unrelated existing Conda prefix. Keep CPU and V100 venvs separate.

The rootless path uses the **CPU build of Conda-forge `libxc-c`**, not its Python bindings package
`libxc`, avoiding an unnecessary Python ABI constraint. It also selects the
prefix library name `scalapack`, instead of Debian's `scalapack-openmpi`.
Compiler and MPI activation is restored by the generated activation script.

The rootless build and generated activation script isolate the private MPI
stack from inherited NVIDIA HPC-X/OpenMPI compiler and runtime settings.
They clear host compiler/linker overrides and MPI-specific settings, filter
known HPC-X/MPI directories out of runtime paths, then prioritize the private
prefix. CUDA driver/library paths outside those MPI directories,
`CUDA_VISIBLE_DEVICES`, and scheduler allocation variables are preserved.
The scripts do not alter system MPI configuration or shell startup files.
Sourcing activation changes the current shell's compiler/MPI environment; use
a subshell if you need to retain a separate host HPC SDK environment afterward.
Conda environment registration is disabled using its
supported `register_envs` setting for dependency installation and later use.
The official Miniforge bootstrap installer still expects `~/.conda` to exist
or to be creatable by the current user; this needs no administrator rights.
For a read-only home without that directory, use an existing `--conda-base`
provided by the site instead of bootstrapping an installer. Source the generated activation file in every new shell.

## 4. NVIDIA V100 training without sudo

The host administrator must already have installed a compatible driver and
made a V100 available to your process. Check this **inside your GPU allocation**:

```bash
nvidia-smi
```

Then run:

```bash
bash tools/deepdft/build_user_linux.sh --mode v100
source .deepdft-build/user-v100/activate.sh
```

Keep any scheduler-assigned GPU visibility; select a permitted device explicitly
only on standalone hosts. This profile uses
**PyTorch 2.7.1 + CUDA 11.8** and requires `sm_70` in the installed wheel.
Newer PyTorch/CUDA packages may omit Volta support; do not replace this profile
with the latest wheel without checking its architectures and running the tests.

The script tests the device name, compute capability 7.0, CUDA matrix
multiplication, finite losses/gradients, and actual weight updates for both
DeepDFT architectures. It does not silently fall back to CPU when CUDA fails.
GPAW remains a CPU/MPI build; PyTorch uses the GPU for density-model training.
CuPy, MAGMA, a GPU GPAW build and a local CUDA toolkit are not required.

For standard CUDA 11.8 configurations, use a Linux driver such as 520.61.05 or
newer **that still supports V100**, chosen by the server administrator. Consult
[NVIDIA's CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html).
The CUDA version shown by `nvidia-smi` is a driver capability; the Python runtime
version is reported separately by `torch.version.cuda`.

The baseline uses FP32. V100 does not have native BF16 or TF32 support; do not
enable them automatically. FP16/AMP needs separate numerical validation.
These scripts target one GPU, not distributed DeepDFT training.

On a scheduler-managed cluster, request an appropriate GPU job and at least two
CPU cores, then run the script there. Use the site's scheduler and network
policy; do not run training on a login node or assume every allocated GPU is a
V100. Package downloads must be possible during initial setup, or the site must
provide an appropriate package mirror/cache and a preprovisioned environment.
No sudo is needed for the user-space packages, but GPU driver and allocation
problems still require administrator or scheduler action.

### Reported target server

The user reported Ubuntu 22.04.5 LTS, glibc 2.35, NVIDIA driver 580.178.04,
and six Tesla V100-SXM2-32GB GPUs. These OS, architecture and driver versions
meet this profile's stated requirements. Ubuntu 22.04's default Python 3.10
can run the bootstrap; the private toolchain supplies Python 3.12 for GPAW.
No Python replacement or system development-package installation is needed.
This is a compatibility assessment from the reported configuration, not a
claim that training has run on the remote server.

Run the following inside your allocated GPU session:

```bash
bash tools/deepdft/build_user_linux.sh --mode v100
source .deepdft-build/user-v100/activate.sh
python tools/deepdft/check_training.py --device cuda --require-v100
```

Preserve the scheduler's `CUDA_VISIBLE_DEVICES` setting. The check uses logical
`cuda:0`, meaning the first GPU visible inside that allocation. On a standalone
server, select a GPU index only if it is assigned to you, for example:

```bash
CUDA_VISIBLE_DEVICES=2 bash tools/deepdft/build_user_linux.sh --mode v100
```

The presence of six cards does not make this a six-GPU training setup. The
initial profile uses one V100; multi-GPU training requires separate design and
validation.

## 5. Alternative: administrator-provided system libraries

If the server already has compatible C++/MPI development packages, use the
existing system-library route without installing anything system-wide:

```bash
bash tools/deepdft/build_linux.sh --mode cpu --python python3.12
source .deepdft-build/cpu/activate.sh
```

It requires Python 3.11-3.13, `mpicxx`, OpenMPI, BLAS/LAPACK, LibXC, FFTW and
ScaLAPACK development files. On Debian/Ubuntu, an administrator can optionally
run `bash tools/deepdft/install_system_deps.sh`. **Do not run that installer if
you lack sudo; use the rootless route instead.**

`--sysroot /path/to/prefix` supports trusted Debian packages already unpacked
under `usr/include` and `usr/lib/x86_64-linux-gnu`. Alternatively,
`--dependency-prefix /path/to/conda-prefix` selects a flat `include`/`lib`
layout with `libscalapack.so`. These options are mutually exclusive; neither
bypasses package authentication or supplies missing packages automatically.

A source checkout contains one editable GPAW C++ extension. Do not use it
concurrently with different Python versions or incompatible NumPy/toolchain
ABIs. Rebuild after switching ABI, toolchain or major dependencies.

## 6. Repeat checks and inspect results

For the recommended rootless CPU profile:

```bash
bash tools/deepdft/check_linux.sh \
  --mode cpu --state-dir "$PWD/.deepdft-build/user-cpu"
```

For V100:

```bash
bash tools/deepdft/check_linux.sh \
  --mode v100 --state-dir "$PWD/.deepdft-build/user-v100"
```

Pass your actual `--state-dir` if you used a custom location. Standalone checks,
after sourcing the generated activation file, are:

```bash
gpaw info
gpaw test
mpiexec --bind-to none -n 2 gpaw test
python -m pytest -q gpaw/test/deepdft/test_smoke.py
python tools/deepdft/check_training.py --device cpu --steps 3
# In a V100 allocation:
python tools/deepdft/check_training.py --device cuda --require-v100 --steps 3
```

Success requires a zero exit status, `cores: 2` in the MPI calculation,
**two passed inference tests with no skips**, finite training gradients, and
changed model parameters. `--mpi-ranks 1` is available for single-core hosts,
but does not establish multi-rank readiness. The old `gpaw -P` shortcut is
deprecated. Check scripts preserve failing command exit codes and use MPI
launch timeouts rather than treating a launched process as a passing test.

The density round-trip preserves two H2 electrons and explicitly converts
GPAW's e/angstrom^3 to the cube convention e/Bohr^3. The training check uses
GPAW-generated spin-summed all-electron density. Its nonperiodic molecular
model graph is an execution check, not a physical accuracy comparison against
the periodic reference calculation. Three optimization steps do not establish
convergence or predictive accuracy.

The state directory contains `build.log`, serial/MPI logs, `smoke.log`,
`smoke.xml`, `training.json`, `python-packages.txt`, `submodules.txt`,
`gpaw-commit.txt`, `source-status.txt`, `siteconfig.py`, and `activate.sh`.
Private-prefix builds also write `link-diagnostics.log`: MPI, PMIx, hwloc and
libevent dependencies must resolve inside that prefix. The preflight links and
runs a small executable with immediate symbol binding before package builds.
The rootless profile additionally saves `toolchain-create.log` and an explicit
Conda package manifest `toolchain-explicit.txt`. Only managed paths are written;
outputs, environments, downloaded packages and checkpoints must not be committed.
Default build/environment paths are ignored by Git.

The Python requirements pin primary versions, not every transitive dependency.
Keep the resulting package manifests for reproducibility and diagnose version
changes before changing pins. Never disable TLS/signature/checksum verification
or relax numerical assertions to make an installation pass.

## 7. Preparing real density training

After passing the execution checks, run the original DeepDFT CLI from its
working directory, saving outputs **outside the submodule and version control**:

```bash
cd third_party/DeepDFT
python runner.py --help
python runner.py \
  --dataset /absolute/path/to/train-density.tar \
  --split_file /absolute/path/to/splits.json \
  --output_dir /absolute/path/outside/git/deepdft-run \
  --device cuda --use_painn_model \
  --cutoff 4 --num_interactions 3 --node_size 64 --max_steps 1000
```

Use `--device cpu` for CPU training. This is a template for a **prepared dataset**,
not a validated thousand-step experiment on arbitrary user data. The upstream
runner has its own worker and probe-count settings. Its loader accepts supported
tar/directory/list inputs containing cube or CHGCAR densities. A split file can
contain `{"train": [0, 1, 2], "validation": [3]}`; indices refer to the dataset's
actual structure order. Keep train/validation/test disjoint and record ordering.

Specify the target density type (pseudo, all-electron, or PAW contribution),
spin convention, charge, units, normalization, grid, probe coordinates, cell,
periodic boundary conditions, exchange-correlation functional, PAW datasets and
reference-calculation convergence. The supplied QM9 models used VASP references
and nonperiodic molecules; passing inference does not demonstrate accuracy on
GPAW references. Do not equate a learned density with a self-consistent solution
or validated conservative energy/force/stress predictions.

## 8. Troubleshooting and validation limits

### NVIDIA HPC-X and Conda OpenMPI conflict

Errors such as `undefined reference to PMIx_Proc_construct` or
`PMIx_Info_destruct`, together with paths from both a private `toolchain/lib`
and `/opt/nvidia/hpc_sdk/.../hpcx/.../ompi/lib`, indicate incompatible MPI/PMIx
libraries in the same link. The failure can occur at the prerequisites check
before installing PyTorch.

The rootless scripts now sanitize inherited MPI/compiler settings before
Conda activation and verify private-prefix linkage explicitly. To resume an
existing managed V100 installation after obtaining the fix:

```bash
git checkout DeepDFT
git pull --ff-only origin DeepDFT
bash tools/deepdft/build_user_linux.sh --mode v100
source .deepdft-build/user-v100/activate.sh
```

The script reuses the existing managed toolchain, venv and package caches.
Preserve your original `--env`, `--state-dir` and `--toolchain` arguments if
you selected custom paths. Review `link-diagnostics.log` if another dependency
still resolves outside the intended prefix. Regression tests for environment
isolation can be run with:

```bash
python -m pytest -q tools/deepdft/tests/test_toolchain_environment.py
```

| Symptom | Action |
|---|---|
| No sudo or system development headers | Use `build_user_linux.sh`; do not run the system package installer |
| Toolchain solve fails on Python/libxc | Use `libxc-c=6.2.2`; do not substitute the Python bindings package `libxc` |
| Miniforge checksum mismatch | Stop; verify the official release or choose another destination; do not change the expected hash to match a failed download |
| HTTPS or package downloads denied | Ask for access to required official destinations or a supported site mirror; keep verification enabled |
| Environment path already exists | Choose another path; scripts refuse to adopt unrelated environments |
| `mpi.h`/`xc.h` missing | Ensure the selected prefix supplies headers and that its compiler/MPI activation is active |
| `undefined symbol` or C++ ABI errors | Restore matching compiler/runtime and rebuild GPAW; do not mix prefixes |
| `not enough slots` | Allocate two CPU cores or use `--mpi-ranks 1` and report the MPI limitation |
| MPI reports one core | Use explicit `mpiexec -n 2 gpaw test` with a matching active MPI backend |
| CUDA unavailable | Check the host driver, GPU allocation, container access, visibility and `cu118` wheel |
| Missing `sm_70` / no kernel image | Use the pinned V100 wheel, then check actual device execution |
| CUDA out of memory | Inspect GPU usage; reduce production probe count/model size; the smoke check uses only 64 probes |
| Read-only or `noexec` storage | Choose user-writable executable storage for toolchain, venv and state |

The system-library CPU profile was previously tested on Debian 13 / Python
3.12.14 with a trusted local sysroot, including repeated builds, serial/MPI
calculations, both inference models and training steps.

The rootless CPU profile was also built with newly provisioned Python 3.12.15,
GCC/G++ 13.4.0, OpenMPI 5.0.11 and LibXC 6.2.2 from Conda-forge. Source builds,
serial/two-rank calculations, both pretrained inference models and three Adam
steps for each architecture passed. Reusing the managed environment and running
the checks from a fresh shell both passed. These rootless builds installed no
system packages and used the private toolchain rather than the earlier sysroot.

An official PyTorch 2.7.1+cu118 CPython 3.12 Linux wheel was inspected and contains
`sm_70` in `libtorch_cuda.so`. **No NVIDIA GPU is available in this development
environment; CUDA execution and V100 training must be verified on the target
server.** Ubuntu system package installation, scheduler integration, full GPAW
regression suites and scientific ML accuracy benchmarks were not run here.

## 9. English-only commits

All committed files must use English prose, including documentation, comments,
examples, messages and filenames. Before each commit:

```bash
git diff --cached --check
python3 tools/check_staged_english.py
```

The guard reads complete staged text files and rejects Cyrillic; submodule Git
references are not text files. It does not detect every non-English language,
so manually review all staged files as well. Do not weaken the guard or add
Russian prose to tests or examples. Conversation language is independent of
repository language.
