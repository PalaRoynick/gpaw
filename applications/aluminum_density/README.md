# Aluminum density workflow with automatic GPU selection

Generate generic Al structures, calculate PAW density references, train PaiNN,
validate held-out grids, and predict larger structures. Keep data, checkpoints,
logs and calculations outside this checkout. Never change periodicity or basis
just to select a GPU. See [reference validation](REFERENCE_VALIDATION.md) for
physical assumptions, electrostatics, relaxation, spin and convergence.

## Backend roadmap

| Stage | Automatic policy | Explicit CPU route |
|---|---|---|
| Geometry generation / dataset splitting | CPU; no expensive GPU kernels | Same command |
| Periodic PW references / relaxation | Real CUDA when available, one rank | `--device cpu` |
| FD references / relaxation | CPU by default; optional experimental GPU with host Poisson | `--device cpu` |
| Fixed-geometry convergence | Same policy as the reference stage | `--device cpu` |
| PaiNN train / evaluate / predict | PyTorch CUDA when available | `--device cpu` |
| Cube serialization / host Poisson / KD trees | CPU, even in a GPU job | Same algorithms |

`--device auto` is the default. `--device cuda` requires a GPU and fails if it
cannot be used. The reference alias `--gpu` means `--device cuda`. Auto explains
why CPU was selected; it does not catch numerical failures or GPU OOM and rerun
on another backend. Physical parameters remain unchanged. Multiple MPI ranks use
CPU in auto mode; multi-rank GPU reference calculations remain unsupported here.

FD GPU kernels exist in this revision (see `gpaw/test/gpu/test_pw.py`), despite the
older GPU documentation describing PW only. Our previous wrapper's unconditional
FD rejection was too restrictive. Full GPU support for every Poisson correction
is not established. The opt-in FD adapter keeps Hamiltonian/density operations on
GPU and stages the charge/potential to CPU for the existing Poisson solver,
including the same moment and slab dipole corrections. It changes execution,
not physical boundary conditions. Transfers can reduce the achievable speedup.
The adapter is experimental and requires the real-hardware tests below before
using it for scientific labels. Array emulation does not certify GPU kernels.

## 1. Environments and installed code

Use separate rootless environments for reference generation and ML:

- Native GPAW/CuPy CUDA build: [build guide](../../doc/deepdft-gpaw-cuda.md).
- PaiNN/PyTorch V100 build: [Linux guide](../../doc/deepdft-linux.md),
  `build_user_linux.sh --mode v100`. This installs CPU GPAW plus CUDA PyTorch.
  The native CUDA environment alone does not install PyTorch/ASAP3.

A CUDA wheel is a source snapshot. After pulling Python-only changes, update
only the density-learning adapters using the destination environment's Python:

```bash
/path/to/gpaw-cuda-build/venv/bin/python \
  /path/to/your/gpaw/tools/deepdft/update_python_adapters.py --dry-run
/path/to/gpaw-cuda-build/venv/bin/python \
  /path/to/your/gpaw/tools/deepdft/update_python_adapters.py
```

The updater copies only `gpaw/density_learning/*.py`, backs up replacements under
`<venv>/.gpaw-adapter-backups/`, invalidates those modules' bytecode, and verifies
native-extension hashes remain unchanged. It imports neither GPAW nor CUDA and
can run from the checkout. Run it between jobs, then start a new Python process.
It does not copy the GPAW core, settings, applications or toolchain. Application
scripts and settings are used directly from the updated checkout. Recompile only
when native sources or build configuration actually change. The runtime-path
activation helper is also sourced from the checkout and updates independently.

Example shell variables (use your own persistent paths):

```bash
PROJECT=/path/to/your/gpaw
WORK=/scratch/your-account/al-density
CUDA_BUILD=/path/to/gpaw-cuda-build
mkdir -p "$WORK"
source "$CUDA_BUILD/activate.sh"
cd "$WORK"
```

All commands below run outside the checkout. Launchers preserve the working
directory. Preserve scheduler CUDA_VISIBLE_DEVICES. Allocate one GPU per job;
there is no multi-GPU DDP training or multi-GPU PBE certification in this workflow.
For CPU-only hosts, activate the rootless CPU environment instead.

## 2. Validate the selected GPAW backend

Start with the PW acceptance test (the user has verified this on a V100):

```bash
python "$PROJECT/tools/deepdft/check_gpaw_cuda.py" \
  --require-v100 --output "$WORK/check-pw-001"
```

Before opting into FD GPU, run each boundary family on the actual GPU:

```bash
for family in bulk slab finite; do
  python "$PROJECT/tools/deepdft/check_gpaw_cuda.py" \
    --mode fd --family "$family" --require-v100 \
    --output "$WORK/check-fd-$family-001"
done
```

These compare CPU and GPU energy, forces and smooth density at identical settings,
reject fake CuPy, and do not relax tolerances. `--cpu-only` checks the CPU baseline
but does not establish GPU support. The small tests are execution checks, not a
speed benchmark or a convergence study. No complete FD calculation in fake CuPy
should be substituted for real-device acceptance.

## 3. Smoke pipeline, retaining open boundaries

```bash
python "$PROJECT/applications/aluminum_density/generate.py" \
  --tier smoke --variants 1 --vacuum 3 --output "$WORK/smoke-geometries"

python -m gpaw.density_learning.cli reference \
  --input "$WORK/smoke-geometries/structures.extxyz" \
  --settings "$PROJECT/applications/aluminum_density/smoke.toml" \
  --experimental-fd-gpu --device auto --output "$WORK/smoke-references"

python -m gpaw.density_learning.cli split \
  --references "$WORK/smoke-references" --output "$WORK/smoke-split"
```

Remove `--experimental-fd-gpu` for the CPU FD route. This mixed bank contains
bulk, slab and finite structures. Do not switch its open structures to fully
periodic PW. The coarse grid, small boxes and broad occupations are smoke-only.

Activate the ML environment, keep the external working directory, then:

```bash
bash "$PROJECT/applications/aluminum_density/train.sh" \
  --splits "$WORK/smoke-split/splits.json" --output "$WORK/smoke-model" \
  --steps 20 --probes 64 --hidden 16 --interactions 2 --cutoff 3 --validate-every 5

bash "$PROJECT/applications/aluminum_density/evaluate.sh" \
  --splits "$WORK/smoke-split/splits.json" \
  --checkpoint "$WORK/smoke-model/best.pt" --output "$WORK/smoke-metrics"
```

These launchers default to auto. Set `DENSITY_DEVICE=cpu` or `cuda` to require a
specific backend. `train_cpu.sh` and the strict `train_v100.sh` remain available.
A 20-step smoke model is not a scientifically usable density predictor.

## 4. Pilot candidates and reference labels

```bash
python "$PROJECT/applications/aluminum_density/generate.py" \
  --tier pilot --variants 2 --seed 0 --vacuum 12 --output "$WORK/pilot-geometries"
```

This generates 52 unrelaxed candidates in 24 parent groups (maximum 243 atoms):
bulk, multiple facets/thicknesses, finite motifs, strains and displacements,
surface defects and periodic cavities. Candidate geometries are not calculated
labels or equilibrium minima. Relax representative parents and compare competing
isomers/spins; related variants remain in one data group.

In the native CUDA environment, after FD hardware acceptance:

```bash
python -m gpaw.density_learning.cli reference \
  --input "$WORK/pilot-geometries/structures.extxyz" \
  --settings "$PROJECT/applications/aluminum_density/reference.toml" \
  --experimental-fd-gpu --output "$WORK/pilot-fd-references" --plan-only

bash "$PROJECT/applications/aluminum_density/reference.sh" \
  --input "$WORK/pilot-geometries/structures.extxyz" \
  --experimental-fd-gpu --output "$WORK/pilot-fd-references" \
  --start 0 --stop 2 --relax-steps 200 --fmax 0.02 --resume-completed
```

The launcher defaults to one process and auto selection. For CPU/MPI explicitly
use `GPAW_RANKS=12 ... --device cpu`. Separate jobs must use nonoverlapping input
ranges/output files. For one-GPU Slurm jobs use `reference_gpu.slurm`, set
GPAW_CUDA_BUILD, and pass absolute input/settings/output paths as job arguments.
The script does not request permission or alter scheduler visibility.

For already fully periodic structures, native PW is the established CUDA route:
use `reference-pw-cuda.toml` with the same commands and omit the FD opt-in flag.
Select a periodic-only input first. The wrapper refuses nonperiodic PW inputs.
FD/PW protocols are kept in separate datasets until convergence is established.

## 5. Reference convergence and physical targets

Start with PBE / Al PAW with three valence electrons and recorded setup hashes.
FD h=0.16 angstrom, PW cutoff 500 eV, k-point density 4, Fermi width 0.01 eV and
130% bands are initial values, not certified convergence. Check grid/cutoff,
k sampling, width, empty bands, vacuum and slab thickness. Energy convergence
alone does not establish density or exterior-field convergence.

```bash
python -m gpaw.density_learning.cli converge \
  --input "$WORK/pilot-geometries/structures.extxyz" --index 0 \
  --settings "$PROJECT/applications/aluminum_density/reference.toml" \
  --experimental-fd-gpu --device auto \
  --parameter grid_spacing --values 0.18 0.16 0.14 \
  --output "$WORK/convergence-fd"
```

Convergence uses the same backend selection and preserves the chosen basis.
Its reports compare energy/atom, density MAE/RMSE, relative L1 and integrals,
including interpolation error; acceptance remains a scientific decision.
Relaxation is fixed-cell BFGS; an unconverged requested relaxation is not exported.
Moments 4 correct monopole and dipole, not all angular orders through l=4.

The target remains spin-summed smooth PAW pseudo density without compensation
charges. Internal units are electrons/angstrom^3; cubes use electrons/Bohr^3 and
contain ionic coordinates. Sidecars retain PBC, settings, execution and provenance.
Its integral need not be 3N. Do not renormalize it or combine it with point +3 ions
as an electrostatic source. Reconstructed valence/all-electron densities require
independent near-core convergence. Refer to REFERENCE_VALIDATION.md for details.

## 6. Full training and held-out evaluation

In the ML environment:

```bash
python -m gpaw.density_learning.cli split \
  --references "$WORK/pilot-fd-references" --output "$WORK/pilot-split"
bash "$PROJECT/applications/aluminum_density/train.sh" \
  --splits "$WORK/pilot-split/splits.json" --output "$WORK/painn-seed0" \
  --steps 20000 --probes 1024 --cutoff 5 --interactions 3 --hidden 64
bash "$PROJECT/applications/aluminum_density/evaluate.sh" \
  --splits "$WORK/pilot-split/splits.json" --checkpoint "$WORK/painn-seed0/best.pt" \
  --output "$WORK/test-metrics" --partition test
```

Grouping prevents parent/equivalent-supercell leakage. Add withheld-size and shape
experiments beyond random splits. Evaluation visits the full grid in bounded
batches and reports regional/planar/radial errors and electron integrals. Training
uses deliberately biased surface sampling; use `--surface-fraction 0` for uniform
sampling. Checkpoints record the resolved device. CUDA OOM is an error, not an
implicit CPU restart. Auto selection is convenient, not a promise of speedup.

## 7. Large-system prediction

```bash
python "$PROJECT/applications/aluminum_density/generate.py" \
  --shape sphere --radius 12 --vacuum 12 --output "$WORK/inference-geometry"
bash "$PROJECT/applications/aluminum_density/predict.sh" \
  --input "$WORK/inference-geometry/structure.extxyz" \
  --checkpoint "$WORK/painn-seed0/best.pt" --output "$WORK/prediction" \
  --spacing 0.2 --tile 24 --probe-batch 2048 --plan-only
```

After reviewing disk/voxel/halo budgets, remove `--plan-only`. Length parameters
are angstrom. Prediction uses CUDA automatically when available. Tiling bounds
working memory but does not reduce output size. The default grid budget is 100M
voxels; raise it only after planning. A 30 nm box at 0.2 angstrom is 1500^3 points,
not 150^3. Periodic tiled inference currently needs orthorhombic cells.

PaiNN's halo is (K+1)*cutoff for K interactions. Tests establish parity with the
complete local model, not physical accuracy at a new size. Local PaiNN cannot
guarantee nonlocal metallic polarization or global electrostatic response.
An ML density is not a self-consistent DFT solution or an energy/force potential.

## Verification

```bash
python -m pytest gpaw/test/deepdft -q
mpiexec -n 2 gpaw python -m pytest -q \
  gpaw/test/deepdft/test_density_workflow.py::test_aluminum_reference_export
```

Tests cover units, symmetry, leakage, halo parity, training/resume, CPU FD export,
backend selection and staged Poisson equivalence. Real FD CUDA hardware acceptance
remains separate. A complete fake-CuPy FD trial on this executor crashed and is
not counted as a passing test. No GPU results are fabricated or inferred from
CPU/array-emulation success. No datasets, weights, PAW files or builds enter Git.


## Localizing an FD CUDA native crash

A force table is not sufficient to locate a segfault: the original test runs CPU
first, then GPU, and extracts density after forces. Inspect both backend logs.
The checker now enables Python faulthandler and emits flushed started/completed
events for calculator creation, SCF, forces, density construction, host transfer,
gather and log closure. Use a fresh output directory for each attempt:

```bash
CUDA_LAUNCH_BLOCKING=1 python "$PROJECT/tools/deepdft/check_gpaw_cuda.py" \
  --mode fd --family bulk --backend cuda --sync-gpu --require-v100 \
  --output "$WORK/fd-cuda-diagnostic-001"
```

To isolate density from force evaluation, add `--skip-forces`. Single-backend
runs and runs skipping forces always report `gpu_validated: false`; they cannot
certify a force backend. Synchronization is diagnostic overhead, not a timing
benchmark. The current FD LFC derivative path enters CPU C code expecting NumPy
arrays; that is a candidate native-crash path when passed device arrays. No
kernel repair is claimed without real-device validation. FD CUDA ionic relaxation uses the same explicit experimental opt-in; validate
forces on the target GPU first.

The default CPU/GPU acceptance now runs each backend in a separate subprocess.
Worker stdout/stderr are streamed and saved as `cpu-worker.log`/`cuda-worker.log`;
GPAW logs and numerical results live under `cpu/` and `cuda/`. A failed worker
writes `failure.json` with its return code and signal and prevents acceptance,
even if it wrote intermediate results before crashing. This isolates calculator
lifetimes; it is not claimed to repair a native FD backend defect.


### Confirmed FD localized-function force crash

A real V100 trace located a crash in `gpaw/lfc.py:derivative`: its native C routine
expects NumPy arrays but received a CuPy object. The Python fix stages derivative
inputs and outputs through host memory, including normalized and complex-valued
cases. The FD compensation-force accumulator now uses the same array backend
as its inputs, avoiding a subsequent NumPy/CuPy mix.
These changes affect only `gpaw/lfc.py` and `gpaw/new/fd/pot_calc.py`; CUDA kernels
and `_gpaw.so` do not need rebuilding. Install those files explicitly:

```bash
/path/to/cuda-build/venv/bin/python \
  "$PROJECT/tools/deepdft/update_python_adapters.py" --include-fd-force-fix --dry-run
/path/to/cuda-build/venv/bin/python \
  "$PROJECT/tools/deepdft/update_python_adapters.py" --include-fd-force-fix
```

The opt-in flag extends the updater's fixed allowlist by exactly those two core
Python files. Both are backed up and native-extension hashes remain checked.
Without the flag, core files are not changed. Rerun real CPU/GPU acceptance on
bulk, slab and finite families. A Python fix plus array emulation is not proof
that every FD GPU operation is supported. FD GPU relaxation is available with `--experimental-fd-gpu`. The former
unconditional block is removed; run the complete real-device force checks before
using relaxed references for scientific training.


## Training in the native GPAW CUDA environment

A native GPAW/CuPy build does not by itself install PyTorch. Training uses CUDA
PyTorch independently of GPAW's PW/FD execution backend. Either activate the
existing rootless ML environment (`.deepdft-build/user-v100/activate.sh`) in a
fresh shell, or add the adapter's training dependency to the native CUDA venv:

```bash
/path/to/gpaw-cuda-build/venv/bin/python \
  "$PROJECT/tools/deepdft/enable_training.py" --profile v100
```

This defaults to PyTorch 2.7.1+cu126 and checks Volta support;
`--torch-runtime cu118` selects the earlier profile. It updates only Python
adapters, initializes the pinned DeepDFT submodule, and registers its path in the
installed adapter package. It does not put the GPAW checkout on sys.path or
rebuild GPAW. The checkout/submodule must remain available when training. Model
checkpoints retain source fingerprints. Set GPAW_DEEPDFT_SOURCE explicitly to
select a different initialized source directory; incompatible checkpoint source
fingerprints are still rejected.

The final check requires an allocated V100 and exercises forward, backward and
Adam parameter updates. `--profile cpu` explicitly selects CPU PyTorch;
`--no-install` checks/configures existing dependencies; `--dry-run` only prints
the destination and intended install. Run with the desired venv's Python and
outside active calculation jobs. Native extension hashes are checked unchanged.
This installs the dependencies used by the density adapter, not all optional
utilities in upstream DeepDFT's dataset module (ASAP3/LZ4 are in the full rootless
ML profile). Native CUDA Toolkit and PyTorch's bundled CUDA runtime are
separate components; a newer system nvcc is not a substitute for CUDA PyTorch.

## References changed after splitting

A split records exact cube hashes. If cube/sidecar references were intentionally
recalculated or replaced, create a new split in a fresh directory and start a new
training output. Do not edit stored checksums or resume an old checkpoint against
a different dataset. If replacement was unexpected, restore the intended data
before training. `tools/deepdft/audit_splits.py /path/to/splits.json` checks the
split, sidecar and actual cube hashes without changing files, and lists the
reference directories used by that split. A new split still rejects inconsistent
physical reference protocols and preserves parent-group separation.


If an installed adapter still reports `FD CUDA relaxation is disabled`, it
contains the older temporary guard. Update the checkout and run
`update_python_adapters.py --include-fd-force-fix` with the destination venv
Python. This updates both the adapter and the two Python force fixes, preserving
the native extension. The existing experimental flag then permits BFGS and emits
an `experimental_fd_relaxation` progress event. No report-file authorization is
required, but the force-validation tests must still be run and reviewed.
