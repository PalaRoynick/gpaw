# Validation record: Al PAW electron–phonon patch

Date: 2026-11-07. Base: `PalaRoynick/gpaw` master
`e3121c92bfe1cfa3d8ac8b0234167a0fe2b87362` (GPAW 26.7.1b1).

**Status: operator fixes are tested; quantitative reproduction of Figure 2 is
not established.** The saved low-cost curves are diagnostic and explicitly
flag a phonon-branch ordering mismatch. Do not use them as converged reference
or transport data.

## What was actually verified

* CPU extension built with MPI, FFTW and ScaLAPACK.
* CUDA extension built for V100 (`sm_70` plus PTX), using CUDA 12.6 and a
  toolkit-compatible GCC host compiler. Its CUDA entry points load.
* The CUDA-enabled wheel also ran a New GPAW **CPU** Al calculation and saved
  NumPy potential/on-site arrays: energy -3.5470713118166897 eV,
  potential shape `(1,9,9,9)`. This is a host-serialization/build check.
* **No NVIDIA device is attached here. No actual GPU calculation, CPU/GPU
  equivalence, or cluster SLURM submission is claimed.**
* `pytest gpaw/test/elph tools/elph_al/test_conventional.py`: **16 passed**.
  The environment emitted one unrelated anyio/pytest deprecation warning.
* Two-rank MPI projection test: **passed**. The current fork requires explicit
  `GPAW_MPI_BACKEND=cgpaw` for this launch configuration.
* Changed Python files pass flake8. Both SLURM scripts pass `bash -n`.
* An independent Old CPU versus New CPU pilot agrees to 1.15e-8 eV in
  eigenvalues, 8.68e-10 Hartree in local potential, 2.70e-10 Hartree in
  on-site Hamiltonians and 4.05e-15 eV/Angstrom in forces. This checks that
  the CPU backends use compatible energy gauges; it is not a GPU check.
  See [backend pilot data](results/backend-check.json).

The CPU test environment used Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1,
libxc 5.2.3, gpaw-data 1.2.1 and ASE 3.30.0b1 at
`a8742a70481542f3c7030c3549d980c3330b095a`. Released ASE 3.29 lacks APIs used
by current fork master, despite its package metadata allowing that version.
The available executor has two CPU cores by quota and 8 GiB RAM.

See [CPU configuration report](results/cpu-build-info.log),
[pytest output](results/pytest.log), [MPI output](results/mpi.log), and
[CUDA build / CPU host check](results/cuda-build-cpu-check.log).

## Independent physics checks

These checks use physical invariants or separately evaluated integrals, rather
than fitting a matrix element to the implementation being tested:

| Check | Result / tolerance |
|---|---|
| Al analytic projector derivative against finite displacement of its center, with the basis fixed | passed, absolute tolerance 2e-6 |
| Centered atom, unequal spin Hamiltonians, complex projectors against a frozen-operator finite difference | passed |
| Local V kernel at a general complex k point against direct real-space integration | passed, 1e-12 absolute |
| Bloch transform at k=q=(1/3,0,0), including rectangular band sets, against independent grid deployment | passed, 1e-12 absolute |
| Energy gauge H→H+cS, epsilon→epsilon+c | passed, 1e-13 absolute |
| Degenerate-subspace unitary rotation, Eq. (97) | passed, 1e-14 absolute |
| Mass prefactor against independent SI conversion | passed |
| Hermitian reciprocity of the Li q/-q matrix elements | passed |
| Cubic conventional cells: reciprocal unfolding for (2,6), (3,4), (4,3) | 6912 primitive k points in each case |
| Conventional-cell dynamical matrix against an analytic fcc nearest-neighbor spring model | passed |

The direct PW check also verifies rigid-translation covariance for every pair
of 16 Gamma states, **without** forcing acoustic matrix elements to zero:

```
g_uniform[m,n] = (epsilon_m-epsilon_n) * <psi_m|grad_r|psi_n>_AE
```

The momentum includes the grid term and the PAW `nabla_iiv` correction.
Measured on the saved PW diagnostic:

* PAW norm/orthogonality residual: **8.22e-15**.
* Maximum translation residual: **2.84e-15 Hartree/Bohr**.
* Relative translation residual: **1.02e-14**.

This is a stringent combined check of V, D, P, Q and R, their energy factors,
projector-derivative sign, and units. It does not establish material/k/grid
convergence. See [machine-readable result](results/pw-smoke/translation-check.json).

## Audit of the supplied 2023 artifacts

The supplied `elph` metadata describes 3³ primitive repeats, 27 atoms, a 29³
potential grid, delta=0.01 Angstrom. The supplied projection metadata and log
instead describe 7³ repeats and 343 atoms on the same 29³ grid. The three
large projection entries were not provided. A numerical audit of those exact
old matrix files is therefore impossible, and the supplied stages cannot be
combined into a consistent reproduction.

The maximum force response identifies displaced atom 13, confirming the
centered reference cell. The original projection basis has only four s,p
functions. It cannot represent the Gamma t2g triplet required by the paper.
Manual normalization introduces an extra `1/sqrt(M_Al)` factor, approximately
0.192516. A single selected initial band also fails the degeneracy average.
The original log and the current default Al setup record the same fingerprint
`0292cae29f5d6237e50f6abdd43a7bdd`; the old cache itself does not bind that
fingerprint, so the log alone is not a cache-integrity guarantee.

See [read-only metadata audit](results/legacy-audit.json). The 29³/3³ grid
cannot be mapped directly onto an integer primitive PW grid; the direct-PW
reference deliberately refuses it.

## Executed fresh 27-atom diagnostic

The positive-phonon derivative cache was generated on the initial 25.7
checkout, then read through the documented v1 adapter. Bands, matrix assembly
and direct-PW diagnostics were rerun on current 26.7.1b1. The separate negative
Gamma control regenerated its derivatives on current 26.7.1b1.

The inexpensive positive-phonon diagnostic uses PBE, a=4.04 Angstrom,
3³ primitive repeats (27 atoms), PW(200 eV), a saved 45³ supercell grid,
a shifted 2³ supercell k mesh, delta=0.01 Angstrom, and mean-minimum-image
phonons. The LCAO route uses dzp and all its bands; the primitive grid is
matched to the saved supercell grid. The PW route uses the same derivative
cache, a matched primitive grid/k mesh and a fully diagonalized PW Hamiltonian.

| Quantity | LCAO | Direct PW |
|---|---:|---:|
| Initial Gamma t2g energy (eV, code gauge) | about 20.3407 | 19.8058 |
| Maximum RMS-averaged total coupling (eV) | 0.11896 | 0.13541 |
| RMS discrepancy from digitized Figure 2 total (eV) | 0.03292 | 0.03933 |
| RMS discrepancy of V (eV) | 0.10376 | 0.08399 |
| RMS discrepancy of P (eV) | 0.10623 | 0.08998 |

**These are discrepancy diagnostics, not accuracy scores.** The highest K
phonon has only 0.0928 overlap with the paper's longitudinal direction. W–K is
tracked from the prescribed longitudinal state, but the other segments use
the specified highest-frequency branch; their ordering is not yet the paper's.
This is reported prominently on the plots. The reference contains published
vector polylines, not original author numerical arrays; nominal half-linewidth
uncertainty is about 1.1 meV. Endpoints/branch jumps are excluded from RMS metrics.

The individual V/P/R split is also PAW-dataset dependent. Agreement of each
term across GPAW and VASP is not guaranteed even after convergence. The
coherent total and the state-selection protocol still require verification.

[LCAO curves](results/lcao-smoke/couplings.png),
[LCAO selected states](results/lcao-smoke/selected-bands.png),
[PW curves](results/pw-smoke/couplings.png),
[PW selected states](results/pw-smoke/selected-bands.png).
The corresponding CSV, complex NPZ, settings and comparison JSON files are in
those directories. The large raw displacement/projection caches are excluded.

## Negative convergence control

The same 27-atom geometry with a Gamma-centered 2³ supercell k mesh is also
underconverged. Even with mean-minimum-image phonons it has **15 imaginary
sampled modes away from Gamma**, with minimum energy **-24.36 meV**.
Both LCAO and direct-PW normalized matrix evaluation stop with an explicit
non-positive-frequency error. No normalized curve was fabricated from this
case. Gamma centering preserves the cubic sampling group but cannot replace
k-point convergence.

See [negative-control phonons](results/gamma-underconverged/phonons.png) and
[its data](results/gamma-underconverged/summary.json).

A separate 8-atom execution test also showed spurious imaginary branches and
was not retained as a quantitative benchmark. On the positive 27-atom shifted
mesh, disabling mean-minimum-image interpolation introduces ten imaginary
sampled modes; enabling it removes them.

## Dense reference and NVIDIA work remaining

The paper used cubic conventional supercells with (repeat,k-super) equal to
(2,6), (3,4), (4,3), i.e. 32/108/256 atoms and **6912 unfolded primitive k
points** in every case. The cheap 27-atom × 2³ mesh has only **216**, a factor
of 32 fewer before symmetry reduction. A 4³ primitive cell is 64 atoms and
must not be labeled the paper's 4³ conventional cell.

`conventional.py` supplies those exact cell sizes, with one displaced atom
sufficient for the monatomic fcc Bravais lattice. `pw_reference.py` unfolds
its electronic mesh, maps the cubic grid into the primitive PW representation,
and computes V,D,P,R directly without LCAO projection. The geometry and force
transform are tested analytically; the dense DFT reference jobs have **not**
been executed here.

The available user hardware is V100/A100, 24 currently available CPU cores on
one node, approximately 630 GB host RAM, and SLURM. Ready templates request
24 CPU tasks / 96 GiB or one GPU / eight CPU cores / 96 GiB. They default to
the 32-atom case. Partition, account, module and environment names remain
site-specific and have not been invented. No remote job has been submitted.

The next required evidence is a dense 32-atom result, its cell/k/cutoff/grid
convergence, the rigid-translation check, correct Figure 1 state selection,
and CPU versus real A100/V100 agreement. See the exact commands and submission
examples in [README.md](README.md). Transport properties and pyallen integration
remain outside this change.
