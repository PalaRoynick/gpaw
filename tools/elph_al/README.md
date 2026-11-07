# Al electron–phonon PAW debugging

Based on fork master `e3121c92bfe1cfa3d8ac8b0234167a0fe2b87362`.

This branch implements the four **all-electron** PAW contributions in Chaput,
Togo and Tanaka, *Phys. Rev. B* **100**, 174304 (2019), Eqs. (75)–(81), and a
small Al benchmark. It is a research patch: the numerical convergence study
and an independent QE/EPW comparison are still required before production use.
See `VALIDATION.md` for the actual results obtained in this environment.

Reference: https://doi.org/10.1103/PhysRevB.100.174304 . Agapito and Bernardi,
PRB **97**, 235146 (2018), motivates the localized-orbital interpolation step;
it does not replace the PAW overlap and transformation corrections.

## What was wrong

* `DisplacementRunner` displaces an atom in ASE's **center** reference cell.
  The previous projector derivative used the primitive atom index in the
  **corner** cell. Local-potential and projector perturbations were therefore
  centered on different atoms.
* The Bloch reconstruction phases were reversed. A direct grid-integral
  test at k=q=(1/3,0,0) now checks both signs against GPAW's actual basis.
* The legacy direct LCAO potential-matrix API returns a transposed operator.
  The cache uses the faster cell-block construction with physical Bloch phases
  and Hermitian completion. A direct complex-k grid integral checks it.
  Projector overlaps are `<p|basis>`; their original conjugation convention is
  retained. A grid-integral check prevented incorrectly changing that convention.
* The 25.7 checkout selected the equilibrium atomic Hamiltonian outside the
  spin loop. Current fork master already fixed this; the patch preserves it.
* The old implementation differentiated the pseudo-Hamiltonian. It omitted
  the energy-dependent overlap correction in Eq. (78) and the PAW
  transformation derivative in Eq. (79).
* Eager cache loading was bypassed during ket preprocessing. Gathered
  coefficients were not broadcast before replicated matrix assembly. The old
  25.7 list-index bug was already fixed in current master.

The supplied 2023 example has additional, independent issues:

* Displacement metadata says **3×3×3**, while projection metadata and the
  projection script say **7×7×7**, despite both using a 29³ grid. Equal array
  sizes do not imply equal physical grids. The three large projection files
  were not supplied, so their contents cannot be audited.
* The supplied projection basis has four s,p orbitals and cannot represent the
  d-like Gamma-25-prime initial states of Figure 2. `dzp` includes d polarization
  orbitals; its convergence still needs checking.
* `prefactor=False` already contracts ASE's **mass-weighted** modes. The manual
  normalization in `draw_gmatrix.py` introduces a second `1/sqrt(M_Al)`, reducing
  amplitudes by about **0.192516**, or a factor of **5.1944**.
* Selecting `[mode=2, band=0, band=2]` does not implement the degeneracy average
  in Eq. (97). A single initial state within a triplet is gauge dependent.
* The line-selection helper tests distance to an infinite line, without a
  segment bound or path ordering. A Monkhorst–Pack mesh is unnecessary for a
  fixed initial Gamma state and an explicitly evaluated final-state path.

ASE 3.29's Frederiksen correction addressed a corner-cell atom. The pinned
ASE development version fixes it. This diagnostic explicitly uses unmodified
central differences with `method='standard'`, followed by symmetry and the ASR.
It also enables `use_mean_minimum_images=True`: this removes spurious imaginary
branches in the 27-atom test. It does not establish numerical convergence.

Run `python tools/elph_al/audit_legacy.py /path/to/gpaw_example_eph` for a
read-only metadata report. `pyallen` is not required by this benchmark, and
its transport routines are deliberately outside this patch's scope.

## Formulas, conventions and units

With `B_Mi = <projector_i|basis_M>`, `dB` its **nuclear** derivative, and absolute
band energies `ei`, `ef` in Hartree, the stored kernels are

```
V = <basis|dV_local|basis>
D = sum_atoms B^* dD B^T
P0 = dB^* D B^T + B^* D dB^T
Qleft = dB^* Q B^T
Qright = Qleft^dagger       (also swaps the two real-space cell indices)
Rkernel = B^* R B^T,     R = -setup.nabla_iiv

gV = V
gD = D
gP = P0 - ei Qleft - ef Qright
gR = -(ef - ei) Rkernel
g = gV + gD + gP + gR
```

`setup.dO_ii` supplies Q. `setup.nabla_iiv` is the **spatial** partial-wave
integral, so its negative is R in Eq. (73). The two energies in gP belong to
opposite sides exactly as in Eq. (78). Do not replace them by an average, the
Fermi energy, or Fermi-level shifted eigenvalues.

Axes are `(spin, q, initial k, mode, final band, initial band)`. The pair API
returns `(mode, final band, initial band)`. Fourier phases follow GPAW's LCAO
convention and ASE's centered reference-cell ordering.

Raw spatial derivatives use Hartree/Bohr. ASE modes contain
`e_atom / sqrt(M_atom / amu)`. Consequently, `prefactor=False` returns
**eV/Angstrom/sqrt(amu)**, not an unweighted spatial derivative. With
`prefactor=True`, the final result is in **eV**:

```
g_eV = g_raw * Bohr / sqrt(2 * amu/me * phonon_energy_eV/Hartree)
```

Mass must not be included again. Non-positive phonon energies away from Gamma
translations raise an error, rather than being silently discarded. The three
Gamma translations are set to zero before normalization by convention;
this enforced zero is not an acoustic-sum-rule validation.

Only after coherently adding complex V,D,P,R is the plotted scalar formed:

```
abs_g = sqrt(sum_{mode,final,initial} abs(g)**2
             / (n_modes * n_final * n_initial))
```

## CPU build

On Debian/Ubuntu, install a C compiler, Python development headers, BLAS/LAPACK,
libxc, FFTW and MPI, for example:

```
sudo apt-get install build-essential python3-dev python3-venv \
    libblas-dev liblapack-dev libxc-dev libfftw3-dev libopenmpi-dev openmpi-bin
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=77.0.3' numpy pybind11
python -m pip install -r tools/elph_al/requirements.txt
export GPAW_CONFIG="$PWD/tools/elph_al/siteconfig_cpu.py"
python -m pip install --no-build-isolation -e .
python -m pip install -r tools/elph_al/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m gpaw info
python -m pytest -q gpaw/test/elph
```

The current fork uses ASE APIs newer than released ASE 3.29. The requirements
pin tested ASE commit `a8742a70481542f3c7030c3549d980c3330b095a` (3.30.0b1).
Do not rely on its currently too-permissive `ase>=3.29.0` package requirement.

`gpaw-data` is a declared dependency and supplies standard setups and bases.
For alternate PAW datasets, set `GPAW_SETUP_PATH` and record their checksums.
Current master uses C++ sources and needs an MPI C++ wrapper (`mpicxx`).
The older 25.7 C checkout needed `mpicc`; do not copy that choice to this branch.
Adjust include/library paths in the template for your cluster. Never mix
system Open MPI with a Conda MPICH runtime in the same process.

## NVIDIA build and execution

Install an NVIDIA driver and a compatible CUDA toolkit, plus the CPU libraries
above. Use a **separate environment and clean build tree** for CPU and CUDA:

```
python3 -m venv .venv-cuda
. .venv-cuda/bin/activate
python -m pip install --upgrade pip 'setuptools>=77.0.3' numpy pybind11
python -m pip install -r tools/elph_al/requirements.txt
export CUDA_HOME=/usr/local/cuda
export GPAW_CUDA_ARCH=80  # V100=70, A100=80, RTX 30=86, H100=90
export GPAW_CONFIG="$PWD/tools/elph_al/siteconfig_cuda.py"
python -m pip install cupy-cuda12x  # choose the wheel matching your toolkit
python -m pip install --no-build-isolation -e .
python -m pip install -r tools/elph_al/requirements.txt
python -c 'import cupy; cupy.show_config(); print(cupy.cuda.runtime.getDeviceCount())'
```

The template disables CUDA-aware MPI by default. Enable it only with an MPI
installation and transport tested for GPU buffers. CUDA versions dropping
support for an architecture cannot build it; use an appropriate toolkit.

**GPU acceleration applies to plane-wave displacement SCFs through New GPAW.**
LCAO projection and the NumPy matrix assembly remain on CPU. The displacement
writer explicitly moves gathered potential and on-site arrays to host memory
for ASE JSON serialization. This is not an end-to-end GPU elph implementation.
A compiled extension or CPU emulation does not establish NVIDIA correctness.

```
python tools/elph_al/run.py --profile smoke --output runs/cuda-smoke \
    --gpu-displacements --stage displacements
# CPU projection and bands can use the CUDA-enabled build:
python tools/elph_al/run.py --profile smoke --output runs/cuda-smoke \
    --gpu-displacements --stage project
python tools/elph_al/run.py --profile smoke --output runs/cuda-smoke \
    --gpu-displacements --stage bands
python tools/elph_al/run.py --profile smoke --output runs/cuda-smoke \
    --gpu-displacements --stage matrix
```

Use identical settings in each stage: the manifest rejects changed settings.
Compare CPU/GPU force and potential derivatives, phonons, and V,D,P,R curves
before using GPU results. No NVIDIA device is attached to the current session;
see `VALIDATION.md` for what was actually built/tested.

## Minimal full calculation

From the repository root, with the environment activated:

```
python tools/elph_al/run.py --profile smoke --output runs/al-smoke
python tools/elph_al/plot.py runs/al-smoke \
    --reference tools/elph_al/reference/chaput-figure2.csv
```

The smoke profile uses PBE, a=4.04 Angstrom, PW(200 eV), h<=0.20 Angstrom,
a 3³ **primitive** supercell (27 atoms), 2³ shifted supercell k points, a matched 6³
shifted primitive electronic mesh, dzp, and ±0.01 Angstrom displacements.
The actual primitive grid is matched to the saved displacement grid when its
dimensions are divisible by the repeats; inspect the log for actual spacing.
It is an execution/debugging test, not a convergence claim.

For MPI, export `GPAW_MPI_BACKEND=cgpaw` to select the compiled MPI backend
(or install/configure mpi4py consistently with your MPI). Set writable
`MPLCONFIGDIR` and `XDG_CACHE_HOME` on restricted hosts.

For MPI, projection requires `parallel={'domain':1,'band':1}`; only k-point
parallelism is supported. Run `displacements`, `project`, and `bands` separately
with `mpiexec -n N python ... --stage STAGE`, then run `matrix` in serial.
The all-in-one MPI invocation stops before matrix postprocessing and prints the
serial command requirement. The public `bloch_matrix` API also supports MPI.

Caches permit resuming an interrupted stage. A zero-byte JSON file is a lock,
not a completed result. After confirming there is no live writer, remove only
its stale lock before resuming. Do not reuse a directory with changed physical
settings. Calculated PAW kernels occupy roughly **five times** the old cache
space. Large `g(q,k)` arrays grow additionally as Nq*Nk*Nmodes*Nbands².

The legacy finite-displacement data can be inspected with

```
python tools/elph_al/run.py --profile archive \
    --fd-cache /path/to/gpaw_example_eph/elph --output runs/al-archive
```

This uses the archived a=4.079793 Angstrom, 3³ primitive cell, 29³ saved grid,
and a new dzp projection. It is a diagnostic with old PAW data whose exact
setup compatibility cannot be certified, not a newly converged calculation.

## Figure 2 protocol and convergence

The path is **Gamma–X–W–K–Gamma–L–U–X**. The initial state is the Gamma t2g
triplet (xy,yz,xz), identified from d-orbital character and checked for splitting.
The final band is the lowest band, and the phonon the highest-frequency mode.
On W–K, the phonon starts from the polarization parallel to (1,1,0) at K
and the electron from the lowest K band; both are followed backwards to
implement the Sigma_1 / (1,1,0) continuation described in Sec. V.A. Exact
endpoints are duplicated to display the one-sided branch limits. The output
stores selected band/mode indices, all complex components and all band/phonon
energies so that branch tracking can be inspected rather than trusted blindly.

Degenerate final bands use a 5 meV tolerance; phonons use 10 micro-eV. The
initial triplet must split by less than 50 meV to pass the smoke guard. Tighten
these thresholds with numerical convergence. Near crossings, inspect orbital
character and mode continuity: overlap tracking is not a space-group-irrep
classifier. Compare the saved selected bands/modes with Figure 1.

The paper used 600 eV, a=4.04 Angstrom and **conventional cubic** supercells:
2³/3³/4³ conventional repeats (32/108/256 atoms), sampled with 6³/4³/3³ k grids.
This driver uses diagonal **primitive rhombohedral** repeats, as required by
GPAW's current Supercell interface. A 4³ primitive cell is 64 atoms and is
**not** the paper's 4³ conventional cell. `--profile paper` requests its cutoff
and lattice parameter but **does not reproduce its supercell geometry**.
Agreement must be established by convergence of the primitive sequence, or by
extending GPAW to general supercell matrices before claiming matched-cell
reproduction. The separate `conventional.py` / `pw_reference.py` route below
already supplies the exact cubic-cell geometry for Al without band folding. Do not relabel primitive repeat counts as the paper's counts.

Start a convergence series (new directory for each setting):

```
python tools/elph_al/run.py --profile paper --repeat 3 --k-super 6 \
    --output runs/al-p3
python tools/elph_al/run.py --profile paper --repeat 4 --k-super 5 \
    --output runs/al-p4
python tools/elph_al/run.py --profile paper --repeat 5 --k-super 4 \
    --output runs/al-p5
```

Also vary cutoff (400/600/800 eV), grid spacing (0.20/0.16/0.12 Angstrom),
primitive k mesh, displacement (0.005/0.01/0.02 Angstrom), projection k mesh,
and orbital basis (dzp/tzp and generated higher-quality bases). The wavefunction
basis error cannot be fixed by increasing the displacement cutoff alone.
Report the actual atom count and PAW setup identity for every curve.

`plot.py` overlays the published V,P,total curves and writes error metrics to
`comparison.json`. These metrics quantify a diagnostic discrepancy; they are
not a pass/fail agreement test. The reference was extracted from the vector
polylines in the supplied PDF. It is **plotted data**, not author raw data;
its line-width uncertainty is approximately 1.1 meV. Regenerate it with
`pip install pymupdf` and

```
python tools/elph_al/extract_reference.py PhysRevB.100.174304.pdf \
    tools/elph_al/reference/chaput-figure2.csv
```

The paper/conventional profiles use Gamma-centered meshes. The inexpensive
smoke profile uses a shifted mesh known to execute with positive phonons, but
a shifted even mesh in a primitive fcc cell breaks full cubic sampling symmetry.
Its saved K phonon ordering differs from the paper. Gamma centering alone is
not a convergence fix: `--gamma-supercell` at 2³ supercell k points produced
imaginary branches and was rejected. Increase the k mesh for either choice.
Always inspect the saved modes against Figure 1.

The V,D,P,R split is PAW-dataset dependent. A different PAW code/dataset need
not reproduce the individual terms exactly even when the converged coherent
total agrees. The expected cancellation between V and P is a useful check;
small D and R in this particular paper are not grounds to omit them generally.

## API and cache migration

Version-2 projection entries are dictionaries of V,D,P,Q,R kernels. Version-1
projection caches cannot recover missing Q/R or fix misplaced projectors;
regenerate them. Explicit `paw_corrections=False` permits legacy diagnostic
reading only. `Supercell.load_supercell_matrix()` returns V+D+P0, the
pseudo-Hamiltonian derivative, for callers needing that quantity.

```
elph = ElectronPhononMatrix(atoms, 'supercell', 'elph')
parts = elph.bloch_matrix(calc, k_qc=qpoints, return_components=True)
# parts['total'], parts['V'], parts['D'], parts['P'], parts['R']

# Fixed initial k and arbitrary final path, without a dense k×q grid:
parts = elph.bloch_matrix_pair(C_initial, C_final,
    eps_initial_Hartree, eps_final_Hartree, k_initial, q)
```

Always use absolute energies consistent with the equilibrium PAW Hamiltonian.
Without `return_components=True`, `bloch_matrix` returns the usual array.
With it, saving writes `gsqklnn-components.npz`. The per-pair route can stream
kernels from disk and prioritizes diagnostic transparency over throughput;
it is not optimized for production transport grids. No transport-property
calculation or integration of `pyallen` is included.


## Independent PW and rigid-translation checks

To separate interpolation/basis error from PAW assembly, an Al-only serial PW
reference computes all four terms directly from plane-wave wavefunctions,
projector integrals and their analytic derivatives. It uses no LCAO basis:

```
python tools/elph_al/pw_reference.py runs/al-smoke --output runs/al-pw
python tools/elph_al/plot.py runs/al-pw \
    --reference tools/elph_al/reference/chaput-figure2.csv
python tools/elph_al/check_translation.py runs/al-pw/path.gpw \
    --output runs/al-pw/translation-check.json
```

The PW check supports diagonal primitive cells and the conventional Al cubes
generated below. It requires a cached grid divisible by the supercell repeats. It deliberately refuses the archived 29³/3³ grid.
The primitive PW grid and k mesh are matched to the displacement calculation;
the cubic route explicitly unfolds the supercell k mesh and uses an integer
real-space grid mapping.
The full PW Hamiltonian is diagonalized to avoid false convergence of a
partially included highest degenerate unoccupied manifold. The tool records
its PAW setup fingerprint. It remains a diagnostic, not a scalable general PW
elph implementation or a replacement for QE/EPW.

The rigid-translation check verifies, for arbitrary pairs of bands at Gamma,

```
g_uniform[m,n] = (epsilon_m-epsilon_n) * <psi_m|grad_r|psi_n>_all_electron
```

The momentum contains both the smooth-grid part and `setup.nabla_iiv`.
This simultaneously checks local and nonlocal derivatives, the signs of Q/R,
energy placement, the nuclear sign of projector derivatives and units. It uses
unnormalized Cartesian matrix elements; forcing three acoustic modes to zero
would hide errors. The measured residual is reported in `VALIDATION.md`.

The supplied data and low-cost smoke runs do not yet establish agreement with
the paper. A small total error alongside larger component errors is insufficient:
inspect branch selection, cell/k/grid/basis convergence and PAW dataset effects.


## Conventional-cell reference jobs for your hardware

The paper's 2/3/4 conventional repeats and 6/4/3 supercell k meshes all unfold
to **6912** primitive electronic k points. The primitive 3³ × 2³ smoke run has
only **216**. The reference density is 32 times larger before symmetry
reduction, so these are intended as separate resource-intensive jobs:

```
export GPAW_MPI_BACKEND=cgpaw
mpiexec -n 32 python tools/elph_al/conventional.py \
    --repeat 2 --k-super 6 --ecut 600 --h .16 --output runs/al-c2
python tools/elph_al/pw_reference.py runs/al-c2 --output runs/al-c2-pw
python tools/elph_al/plot.py runs/al-c2-pw \
    --reference tools/elph_al/reference/chaput-figure2.csv
python tools/elph_al/check_translation.py runs/al-c2-pw/path.gpw \
    --output runs/al-c2-pw/translation-check.json
```

Repeat with `(repeat,k-super)=(3,4)` and `(4,3)`. This yields 32, 108, 256 atoms.
Use `--gpu-displacements` on an allocated, validated NVIDIA host for the SCFs;
PW postprocessing remains serial CPU. The run uses Gamma-centered meshes;
the paper specifies mesh sizes but its exact offset/input files were not
provided. The cubic grid is rounded upward to ensure integer primitive-grid
mapping. Inspect the actual printed grid spacing.

Only conventional atom 0 is displaced, in the center reference cell. Since Al
is a monatomic fcc Bravais lattice, its forces on every atom recover all three
primitive phonon branches; this avoids artificial conventional-cell band
folding. The dynamical matrix averages equally short images and imposes the
acoustic sum rule. This shortcut is specific to monatomic fcc Al and is not a
general multi-sublattice phonon solver. Geometry, reciprocal unfolding and
analytic nearest-neighbor spring-model tests are in `test_conventional.py`:

```
python -m pytest -q gpaw/test/elph tools/elph_al/test_conventional.py
```

Each displacement gets a fresh calculator: point-group symmetry can then be
used without reusing the x-displacement little group for a y/z displacement.
Logs are `scf-eq.txt`, `scf-0x-.txt`, etc. Resuming preserves completed cache
entries; remove stale zero-byte locks only after confirming no writer exists.

The selected bands/modes are plotted in `selected-bands.png`. In particular,
`K_highest_mode_longitudinal_overlap` reports whether the highest K phonon
matches the paper's longitudinal state. If this is below 0.8, the plot and
comparison report a branch-order mismatch: the curve is a diagnostic and must
not be called a reproduction of Figure 2. Dense k and grid convergence should
resolve such ordering before quantitative comparison.


## SLURM templates for the available node

The CPU template requests **24 MPI tasks on one node** and 96 GiB RAM, within
the reported 24 available cores and approximately 630 GB host memory. The
GPU template requests **one GPU**, eight CPU cores and 96 GiB host RAM. Both
default to the 32-atom reference; the 108/256-atom variants are selected by
`ELPH_REPEAT=3/4`. These jobs have not been submitted to your cluster.

Build separate CUDA environments for the two GPU architectures, from separate
clean source checkouts/worktrees:

```
# V100 environment: CUDA 12.x supporting Volta
export GPAW_CUDA_ARCH=70
# A100 environment instead:
export GPAW_CUDA_ARCH=80
# Optionally point nvcc at a toolkit-compatible host compiler:
export GPAW_CUDA_CCBIN=/path/to/g++
```

The CUDA configuration is `siteconfig_cuda.py`. Do not install CPU, V100 and
A100 editable builds into the same source tree: their extension modules would
overwrite one another. Activate the appropriate environment for each job.
No GPU result is validated merely by selecting an architecture or passing
CPU emulation. Start with the 32-atom job on A100, then repeat it on V100 and
compare components/forces/phonons. Larger cells need GPU memory sizing.

The GPU template first runs `check_backend_gauge.py --gpu`: a small physical
Old CPU versus New GPU comparison of eigenvalues, local potentials, on-site
Hamiltonians and forces. It stops before the dense job if these differ beyond
the specified tolerances. The same pilot without `--gpu` passed for the two
CPU backends here; real GPU execution remains to be checked on your node.

Submit from the corresponding source checkout root (replace partition and
absolute environment/scratch paths):

```
sbatch --partition=CPU_PARTITION \
  --export=ALL,GPAW_VENV=/absolute/cpu-venv,ELPH_RUN_ROOT=/scratch/al-elph \
  tools/elph_al/slurm/reference_cpu.slurm
sbatch --partition=A100_PARTITION \
  --export=ALL,GPAW_VENV=/absolute/a100-venv,ELPH_RUN_ROOT=/scratch/al-elph-a100 \
  tools/elph_al/slurm/reference_gpu.slurm
sbatch --partition=V100_PARTITION \
  --export=ALL,GPAW_VENV=/absolute/v100-venv,ELPH_RUN_ROOT=/scratch/al-elph-v100 \
  tools/elph_al/slurm/reference_gpu.slurm
```

No partition/account/module names were invented. Use the cluster's MPI and
math-library paths in the build configuration. Return `summary.json`,
`comparison.json`, `translation-check.json`, `couplings.csv`,
`matrix-elements.npz`, selected-band plots and the SCF/SLURM logs for review.
Keep the large raw displacement caches on your filesystem. Include the
`backend-check.json` pilot result when returning GPU data.
