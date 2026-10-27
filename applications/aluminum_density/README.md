# Aluminum reference-density and PaiNN workflow

This source-checkout application prepares neutral aluminum density datasets,
trains a scalar-density PaiNN adapter, and evaluates or exports density cubes.
The reusable implementation lives in `gpaw/density_learning`; case-specific
geometry generation, reference settings and launchers live here. All paths
containing datasets, logs, checkpoints and predictions must be outside the
source checkout. The workflow creates no commits, PRs or upstream notifications.

## Scientific conventions

- Start from the standard scalar-relativistic frozen-core **Al PAW/PBE setup
  with 3 valence electrons**. The application checks the actual PAW valence and
  records the setup fingerprint, GPAW version and resolved k points.
- The selected target is GPAW's **smooth PAW pseudo density without compensation
  charges**, in electrons/angstrom^3 internally. The current GPAW ASE
  `get_pseudo_density()` default adds compensation charges and can be signed;
  this wrapper uses the public `Densities` API with that addition disabled.
- Smooth PAW density is a representation-dependent field, including GPAW's
  pseudo-core convention. Its integral need not equal `3 * number_of_atoms`.
  The measured integral and augmentation difference are recorded. No silent
  normalization to a valence or nuclear electron count is performed.
- Cube values use **electrons/Bohr^3**. Cube headers contain nuclear atomic
  numbers and ionic coordinates in Bohr. Thus one cube already stores both
  the structure and the scalar field; a fictitious second ionic-density field
  is unnecessary. A required JSON sidecar restores PBC, units and provenance.
- `density_kind="valence"` exports reconstructed all-electron valence density;
  `"all-electron"` additionally includes frozen core electrons. Keep these
  targets in separate datasets. They require independent near-nucleus grid
  convergence and are not interchangeable with pseudo density.
  Valence-only reconstruction is not forcibly normalized by this GPAW API.
  Refine its grid until the electron-integral check passes; factors up to 32
  are available and guarded by `--max-density-voxels` (100 million by default).
  Full all-electron count correction alone does not prove near-core convergence.

## 1. Environment

Use the tested rootless build from the repository root:

```bash
bash tools/deepdft/build_user_linux.sh --mode cpu
source .deepdft-build/user-cpu/activate.sh
```

For an allocated V100, use `--mode v100` and its corresponding activation file.
See `doc/deepdft-linux.md` for the NVIDIA driver, PyTorch cu118, and environment
isolation requirements. The density pipeline uses existing optional ML
dependencies; importing `gpaw.density_learning` itself does not import PyTorch
or initialize MPI.

The commands below use an example workspace. Replace it with a writable
persistent path outside Git:

```bash
WORK=/scratch/your-account/al-density
mkdir -p "$WORK"
```

## 2. Start with an inexpensive software smoke run

```bash
python -m applications.aluminum_density.generate \
  --tier smoke --variants 1 --vacuum 3 --output "$WORK/smoke-geometries"

python -m gpaw.density_learning.cli reference \
  --input "$WORK/smoke-geometries/structures.extxyz" \
  --settings applications/aluminum_density/smoke.toml \
  --output "$WORK/smoke-references"

python -m gpaw.density_learning.cli split \
  --references "$WORK/smoke-references" --output "$WORK/smoke-split"

bash applications/aluminum_density/train_cpu.sh \
  --splits "$WORK/smoke-split/splits.json" --output "$WORK/smoke-model" \
  --steps 20 --probes 64 --hidden 16 --interactions 2 --cutoff 3 --validate-every 5

bash applications/aluminum_density/evaluate.sh \
  --splits "$WORK/smoke-split/splits.json" \
  --checkpoint "$WORK/smoke-model/best.pt" --output "$WORK/smoke-evaluation"
```

This tiny bank contains bulk, a thin layer and a finite dimer. Its coarse grid,
small box, broad occupations and tiny training budget test execution only.
Neither the reference settings nor the resulting model are production physics.
A few optimization steps and a decreasing validation loss do not establish
scientific accuracy.

## 3. Build a candidate reference bank

```bash
python -m applications.aluminum_density.generate \
  --tier pilot --variants 2 --seed 0 --vacuum 12 --output "$WORK/pilot-geometries"
```

The current pilot has 52 candidates and 24 parent groups: bulk supercells,
(100)/(110)/(111) slabs with multiple thicknesses, fcc-derived spheres,
icosahedral/decahedral/faceted particles, surface pits, periodic cavities,
small strains and displacement variants. The largest candidate has 243 atoms.
These are **unrelaxed candidates**, not predicted equilibrium structures.
Random displacement is not an ionic temperature or a canonical MD ensemble.

The ambient bulk lattice is fcc, but finite structures need not retain that
motif. Compare several motifs and relax selected parents using the same DFT
protocol. BFGS relaxes ionic positions at fixed cell; establish the bulk lattice
constant separately with an EOS/lattice scan. The 4.05 angstrom generator value
is a starting value, not a calculated equilibrium constant.
The reference command supports `--relax-steps N --fmax 0.02`; an
unconverged requested relaxation is not exported. The initial model is
nonmagnetic as specified by `initial_moment=0`. Small isolated clusters can have
competing spin states, including even-electron clusters. Their magnetic ground state is
not established by this starting protocol. Different spin/XC/occupation
protocols are rejected in one dataset unless a future model explicitly
conditions on those state variables.

Inspect the resource plan before launching DFT:

```bash
python -m gpaw.density_learning.cli reference \
  --input "$WORK/pilot-geometries/structures.extxyz" \
  --settings applications/aluminum_density/reference.toml \
  --output "$WORK/pilot-references" --plan-only
```

The plan estimates FD-grid and wavefunction storage, not peak RAM. Workspaces,
projectors, k-point/band distribution and MPI replication can increase memory.
Allocate resources according to representative measured jobs. Standard
Kohn-Sham labeling of very large inference geometries is not attempted by
default: the reference atom budget is 256 and must be raised explicitly.

## 4. Establish reference convergence

`reference.toml` proposes the following **starting** configuration:

| Setting | Starting value and required checks |
|---|---|
| XC / PAW | PBE / standard 3-valence-electron Al; fingerprint recorded |
| Basis | Real-space FD, h=0.16 angstrom; test 0.18/0.16/0.14/0.12 |
| Alternative basis | PW 500 eV for fully periodic cross-checks; test 400/500/600 |
| Reciprocal sampling | ASE k-point density 4 on periodic axes; test 3/4/5/6 |
| Isolated objects | Gamma only, open FD boundaries |
| Occupations | Fermi-Dirac 0.01 eV; test 0.02/0.01/0.005/0.002 consistently |
| Empty bands | 130% of occupied count; verify band/occupation convergence |
| Mixing | Pulay beta=0.05, history 5, weight 50 |
| SCF criteria | Energy 1e-6, density 1e-5, eigenstates 1e-8 in GPAW conventions |
| Finite boundaries | Initial monopole/dipole correction; test higher moments and box size |
| Slabs | xy periodic, z vacuum, dipole correction; converge slab thickness/vacuum |
| Vacuum | Initial 12 angstrom per exposed side; compare 16/20 and larger where needed |
| Ionic relaxation | Start fmax=0.02 eV/angstrom, verify a tighter criterion |

GPAW interprets Fermi-Dirac width as k_B*T. A numerical occupation width is not
zero temperature; it must be distinguished from a chosen physical electronic
temperature. Do not merge different widths into an unconditioned density target.
The same caveat applies to charge and spin. PBEsol/LDA can be separate
sensitivity studies for lattice/surface properties; they are not silently mixed
with PBE labels. LCAO is useful for screening, not the default reference-density
basis. Ionic minimization with another potential also needs DFT rechecking.

Converge energy **and density**. Example fixed-geometry sweep, with the last
value serving as the comparison reference:

```bash
python -m gpaw.density_learning.cli converge \
  --input "$WORK/pilot-geometries/structures.extxyz" --index 0 \
  --settings applications/aluminum_density/reference.toml \
  --parameter grid_spacing --values 0.18 0.16 0.14 \
  --output "$WORK/convergence-grid"
```

The tool compares grids using explicit unit conventions and interpolation with
axis-specific periodic/open boundary padding. It reports energy/atom, density
MAE/RMSE, integrated relative L1 and electron-integral changes; it never accepts
convergence automatically. Repeat on bulk, a surface and a finite structure.
The comparison itself has interpolation error. Vacuum changes require a common
physical region of interest rather than comparing fractional-cell profiles.

Production calculation, on a shared filesystem:

```bash
GPAW_RANKS=4 bash applications/aluminum_density/reference.sh \
  --input "$WORK/pilot-geometries/structures.extxyz" \
  --output "$WORK/pilot-references" --start 0 --stop 4
```

Use nonoverlapping `--start`/`--stop` ranges for separate jobs. Every MPI rank in
one job evaluates the same references; only rank zero writes cubes/sidecars.
`--resume-completed` checks settings, input geometry, cube checksum and requested
relaxation before reusing a result. Outputs are not overwritten. Review and
expand the calculation range after initial jobs succeed.

## 5. Split and train PaiNN

```bash
python -m gpaw.density_learning.cli split \
  --references "$WORK/pilot-references" --output "$WORK/pilot-split" --seed 0

bash applications/aluminum_density/train_v100.sh \
  --splits "$WORK/pilot-split/splits.json" --output "$WORK/painn-seed0" \
  --steps 20000 --probes 1024 --cutoff 5 --interactions 3 --hidden 64 \
  --learning-rate 0.001 --validate-every 100 --seed 0
```

Use one allocated V100 per process. Preserve scheduler `CUDA_VISIBLE_DEVICES`;
this launcher checks Volta capability and `sm_70` and uses the first visible
GPU. It does not implement six-GPU DDP. Multiple allocated GPUs can run
independent seeds/folds in separate output directories under the site's
scheduler. GPU runtime/training must be checked on the target server.

CPU fallback:

```bash
DENSITY_THREADS=16 bash applications/aluminum_density/train_cpu.sh \
  --splits "$WORK/pilot-split/splits.json" --output "$WORK/painn-cpu" \
  --steps 20000 --probes 1024 --cutoff 5 --interactions 3 --hidden 64
```

Training uses one structure per microbatch, disk-cached reference arrays, cached
atom graphs, and atom-to-probe KD-tree queries. It never creates probe-probe
edges. Half the default training probes are sampled near geometrically
undercoordinated atoms and half uniformly; this is a deliberately biased
training objective, not an unbiased full-grid MSE estimator. Validation probes
are fixed and uniform. `--surface-fraction 0` gives uniform training.

Variants of one parent share a split. Equivalent periodic fcc supercells also
share one parent group. Families with enough independent groups are stratified;
not every small family can appear in every partition. Inspect `splits.json` and
add explicit withheld-size/topology experiments rather than relying only on a
random split. Rotated copies are not independent samples. PaiNN supplies
rotation-equivariant features; a scalar density is tested by jointly rotating
atoms, cell and probes without voxel interpolation.

The adapter adds softplus positivity and a smooth cosine gate vanishing at the
atom-probe cutoff. This removes a nonzero vacuum background but truncates tails;
validate omitted-tail errors. No electron-count renormalization is applied.
Training checks finite losses/gradients, clips gradient norm at 10, and saves
atomic `last.pt`/`best.pt` checkpoints. To resume, specify `--resume .../last.pt`
and a larger total step count with the same dataset, seed and sampling settings.

## 6. Held-out validation

```bash
DENSITY_DEVICE=cuda bash applications/aluminum_density/evaluate.sh \
  --splits "$WORK/pilot-split/splits.json" --checkpoint "$WORK/painn-seed0/best.pt" \
  --output "$WORK/test-metrics" --partition test
```

Evaluation visits the **entire held-out grid** in bounded probe batches. Reports
include MAE, RMSE, integrated relative L1, predicted/reference grid integrals,
near-atom, low-density, far-region and undercoordination-region errors, planar
profiles, and radial profiles with explicit centers and sample counts. The
radial center is the center of mass for a finite object and cell center for a
periodic one; it is a geometric diagnostic, not an automatically inferred
physical interface. A single aggregate metric can hide structured errors.

Use validation for model selection and test only for reporting. Six smoke
training steps on a few small structures do not create a predictive production
model. In particular, successful execution does not establish size extrapolation.

## 7. Large-structure preparation and tiled inference

Generate an inference geometry without DFT:

```bash
python -m applications.aluminum_density.generate \
  --shape sphere --radius 12 --vacuum 12 --output "$WORK/inference-geometry"
# Alternative: --shape cavity --radius 12 --ligament 12
```

Radius, vacuum and ligament are in angstrom. The generator accepts much larger
radii; no 500-atom limit is imposed. First inspect the plan:

```bash
bash applications/aluminum_density/predict.sh \
  --input "$WORK/inference-geometry/structure.extxyz" \
  --checkpoint "$WORK/painn-seed0/best.pt" --output "$WORK/prediction" \
  --spacing 0.2 --tile 24 --plan-only
```

A periodic cavity represents an array of cavities. Converge the ligament and
cell dimensions before interpreting it as an approximately isolated defect.
The same issue applies to periodically repeated pits in a slab.

After reviewing disk/grid/halo budgets:

```bash
DENSITY_DEVICE=cuda bash applications/aluminum_density/predict.sh \
  --input "$WORK/inference-geometry/structure.extxyz" \
  --checkpoint "$WORK/painn-seed0/best.pt" --output "$WORK/prediction" \
  --spacing 0.2 --tile 24 --probe-batch 2048
```

The default voxel budget is 100 million. Raise `--max-voxels` explicitly only
after checking the plan and disk space. ASCII cubes are much larger than binary
float grids. The predictor uses a global atom KD-tree, local halo graphs,
per-tile cached atomic representations and a temporary disk-backed output array,
then streams the cube in standard order. Probe batching limits GPU activation
memory; tiling limits atom-graph memory. `--max-halo-atoms` bounds each local
halo. Periodic tiled inference currently requires orthorhombic cells; open
finite cells can be more general.

For K atomic interaction layers and cutoff R, the halo radius is (K+1)*R.
Small-system tests compare tiled predictions against the complete local model,
including periodic images. This is exact locality bookkeeping within numerical
precision, **not evidence of accurate physical extrapolation**. Structures
identical within that range have identical local predictions even if their
global properties differ. Nonlocal or state-dependent targets require an
appropriate model extension and new reference validation.

## Tests and scope

```bash
python -m pytest -q gpaw/test/deepdft/test_density_workflow.py
mpiexec -n 2 gpaw python -m pytest -q \
  gpaw/test/deepdft/test_density_workflow.py::test_aluminum_reference_export
```

Tests cover cube units/coordinates/PBC, incomplete-output protection,
parent-group splitting, SO(3)/translation covariance, halo/full-model parity,
positive vacuum behavior, training/resume, full-grid metrics and real GPAW Al
export on one/two ranks. GPU execution, production DFT convergence, production
model accuracy and a full large high-resolution cube are separate validation
steps. All generated artifacts stay outside the checkout.

Generic methodological references: the GPAW source documentation under
`doc/documentation/basic.rst` and `doc/documentation/poisson.rst`; the PAW
formalism (Blochl, Phys. Rev. B 50, 17953, 1994); PBE (Perdew, Burke and
Ernzerhof, Phys. Rev. Lett. 77, 3865, 1996); the pinned DeepDFT implementation;
and PaiNN (Schutt, Unke and Gastegger, ICML 2021, arXiv:2102.03150).
