# Reference validation before density training

The existing workflow is a density-learning baseline. Its smoke settings,
20-step model and successful large-system execution do not establish predictive
accuracy, electrostatic accuracy, or convergence of the production references.

## PAW density and electrostatics

The exported smooth pseudo density excludes compensation charges. Its integral
need not equal three electrons per Al atom. Do not rescale it to that number or
combine it with point ions of charge +3 to construct a Poisson source. Neither
operation recovers the missing PAW augmentation moments. The sidecar explicitly
marks `electrostatic_source_ready: false`, including reconstructed density exports:
those exports also lack a matched ionic source and an electrostatic validation.

Use GPAW's self-consistent electrostatic potential as the first reference.
`calc.get_electrostatic_potential()` returns the PAW smooth electrostatic
potential energy of an electron in eV, not the full effective Kohn-Sham potential
(which includes exchange-correlation). In vacuum outside augmentation regions,
phi in volts is numerically minus this electron potential energy in eV;
E in V/angstrom is its positive spatial gradient in eV/angstrom. Define the
potential gauge explicitly and compare fields away from artificial boundaries.
Use the actual potential grid: it need not match the exported density grid.
Potential export and electrostatic training losses are not implemented here yet.

Before adding a new electrostatic source representation, verify its charge,
dipole, exterior potential and exterior field against GPAW. All-electron density
alone is not a shortcut: nuclear singularities require a consistent representation
and a converged integration scheme. Smooth PAW source reconstruction requires
setup-dependent compensation coefficients and density matrices.

## Boundary conditions

- Isolated finite systems: use FD with open boundaries, converge vacuum and
  moment corrections independently. `multipoles=4` corrects the monopole and
  three dipole components, not all angular momenta through l=4. Test 4, 9 and 16
  moments plus larger boxes. Extended-grid and multicenter corrections are
  available in GPAW but are not exposed by this wrapper.
- Slabs: periodic x/y and open z, with a dipole correction for asymmetric slabs.
  Converge slab thickness, both vacuum regions and lateral repeat distance.
  Dipole correction removes an artificial planar field; it does not remove
  lateral periodic replicas or provide arbitrary isolated boundary conditions.
- Bulk: use periodic boundaries and k-point convergence. A printed cell dipole
  is not an isolated object's dipole or evidence of a physical exterior field.
- Charged systems need a separately validated boundary protocol. A periodic
  neutralizing background does not describe an isolated charged specimen.

The current wrapper applies dipole correction only to x/y-periodic slabs; it
must not be assumed to cover arbitrary slab orientations or wire geometries.

## Numerical and physical validation matrix

PBE and the three-valence-electron Al PAW setup are a starting point. Record its
fingerprint; compare selected cases to a suitable alternative functional and/or
independent implementation. FD h=0.16 angstrom and PW 500 eV are candidates, not
universal accuracy guarantees. Cross-check FD/PW on periodic bulk and surfaces
using consistent physical boundary conditions.

Independently vary grid spacing (for example 0.20, 0.16, 0.12 angstrom), k-point
sampling, smearing (0.10, 0.05, 0.02, 0.01 eV), empty bands, SCF tolerances,
vacuum and slab thickness. Include final energy, forces, density differences,
work function, potential plateau, residual vacuum field and surface dipole in
the acceptance criteria. Set tolerances from the intended observable. Energy
convergence alone does not establish electric-field convergence. Do not mix
reference protocols within a training partition without an explicit model.

At 300 K, kBT is about 0.0259 eV; 0.10 eV corresponds to about 1160 K if treated
as an electronic temperature. Numerical smearing may aid SCF, but zero-temperature
energy extrapolation does not undo density smearing. Test density and potential
as width decreases. Bulk and slabs require Brillouin-zone integration; an isolated
cluster uses Gamma and discrete occupations. Surface states are part of the slab
spectrum, not a separate occupation prescription.

## Geometry and spin

Generate multiple isomer families, relax representative structures, then include
controlled perturbations around those minima. A BFGS minimum is local. Keep
variants, parents and related relaxed structures in the same dataset group.
Use constraints for slab layers intended to represent a bulk substrate; ASE
constraints in the input are retained by the wrapper. Bulk lattice optimization
is a separate task: the wrapper relaxes positions at fixed cell.

For example, after selecting allocated CPU resources and activating the build:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GPAW_RANKS=8 bash applications/aluminum_density/reference.sh \
  --input "$WORK/selected-geometries.extxyz" \
  --output "$WORK/relaxed-references" --relax-steps 200 --fmax 0.02 \
  --max-wavefunction-gib 16
```

The budget is total estimated wavefunction storage, not per rank and not total
RAM. Choose it below the allocation with room for solver workspaces. Jobs above
it fail before initializing GPAW. Nonconverged relaxations are not published.
Final coordinates are already in the cube and JSON. The initial geometry hash
and relaxation request are retained. Zero requested steps means unrelaxed labels.

Bulk Al is ordinarily nonmagnetic; small clusters require explicit spin checks.
Compare several initial moments at the same geometry and numerical settings,
then relevant relaxed minima. `initial_moment` enables spin polarization and
sets an initial total moment distributed over atoms; it does not constrain the
final spin sector. The final moment is recorded, but ground-state spin remains
unverified. Even electron count does not prove a singlet. Electrostatic Poisson
uses the spin-summed charge; spin changes that charge indirectly through SCF.

## Execution diagnosis

`cores: 1`, `GPAW_MPI_BACKEND: serial` and `GPAW_USE_GPUS: False` describe a
serial CPU run. Linking MPI is not an MPI launch. Check the activated environment:

```bash
mpiexec -n 2 gpaw python -c \
  'from gpaw.mpi import world; print("rank", world.rank, "size", world.size)'
```

Expect size 2 and both ranks. Use the same activated MPI stack for the reference
launcher. `OpenMP: False` means compiled GPAW OpenMP kernels are unavailable;
setting OMP_NUM_THREADS cannot enable them. Start with MPI and one BLAS thread per
rank. Threaded BLAS can be benchmarked within the CPU allocation, avoiding rank
and thread oversubscription.

The V100 environment enables PyTorch training, not GPU execution of FD GPAW.
Use `train_v100.sh` for one allocated V100 per model; the workflow has no DDP.
GPAW GPU ground-state support in this revision is PW-only and requires its own
GPU-enabled build and compatible CuPy/MPI stack. Do not switch boundary conditions
or enable GPAW_USE_GPUS simply to make an FD reference run on a GPU.

Different Python-source and extension Git hashes do not alone prove an ABI
failure. Check changes to C sources/build settings, rebuild when affected, and
run serial/MPI numerical checks. The workflow commit changes Python and docs;
a prior extension hash can therefore be expected. A completed SCF establishes
execution, not convergence with respect to discretization.

## Grid storage scaling

Convert nm to angstrom before planning: a 30 nm box with 0.2 angstrom spacing
has 1500^3 = 3.375 billion points, not 150^3. One float64 scalar field alone
requires 27 GB (decimal); FFT workspaces and complex arrays require more.
Tiled prediction reduces peak working memory but does not reduce the output size
or create a global charge response. Exterior probe points, adaptive grids and
validated multipole/Green-function methods can avoid a uniform fine vacuum grid.

## Extending the model

Keep local PaiNN as a baseline. SO(3) equivariance does not supply long-range
charge response. A possible next model combines a local density representation
with a globally constrained charge/multipole response and a differentiable
Poisson solve. Validate each component before combining them.

A Poisson residual loss alone is insufficient: solving Poisson exactly for a
wrong charge gives zero residual. Supervise gauge-aligned exterior potentials,
fields and charge/multipole observables with appropriate units and weights.
QEq is a model hypothesis, not a proof of metallic response. Require a stable
quadratic form on the fixed-charge subspace, appropriate boundary Green functions,
size-transfer and polarizability tests. Avoid double-counting Coulomb interactions
by adding a second Poisson feedback to a QEq energy that already contains them.
Global rescaling of an arbitrary local density does not repair polarization.
An atom-centered density basis is optional; its completeness must be checked.

Experimental comparisons can use clean facet-resolved work functions,
photoemission, Kelvin-probe contact-potential differences, cluster ionization
energies/electron affinities, polarizability and structural measurements.
Account for oxide, adsorbates, support, temperature, charge state and instrumental
averaging; none supplies a unique full ground-state density map by itself.

## Sources available in this checkout

- [GPAW advanced Poisson solvers](../../doc/documentation/poisson.rst), including
  Castro, Rubio and Stott, *Can. J. Phys.* 81, 1151 (2003).
- [GPAW slab dipole correction](../../doc/tutorialsexercises/electrostatics/dipole_correction/dipole.rst),
  including Bengtsson, *Phys. Rev. B* 59, 12301 (1999).
- [Surface work-function example](../../doc/tutorialsexercises/basics/surface/work_function.py).
- [GPAW GPU support and build constraints](../../doc/documentation/gpu.rst).
- [PAW density definitions](../../gpaw/densities.py) and
  [electrostatic potential reconstruction](../../gpaw/electrostatic_potential.py).

## CPU utilization and observable progress

The GPAW header's `cores` value counts MPI ranks, not BLAS threads. For example,
three ranks with four BLAS threads still report `cores: 3`. `OpenMP: False`
means GPAW's own kernels lack OpenMP parallelism; BLAS may still use threads.
SMT siblings, thread migration, helper threads and unrelated jobs can make htop
CPU bars misleading. Use per-process thread activity and affinity together with
`lscpu -e=CPU,CORE,SOCKET,NODE` to distinguish logical CPUs from physical cores.
Do not infer a twofold oversubscription from the number of changing CPU bars.

The reference CLI emits flushed JSON events to stdout: runtime facts for every
rank, initialization start, each completed SCF step, SCF convergence, export start
and completion. Elapsed seconds include initialization and any relaxation steps;
the SCF iteration number restarts for each geometry during relaxation. Runtime
facts include PID, hostname, allowed logical CPUs and thread environment values.
If optional `threadpoolctl` is installed, loaded pool capacities are also reported;
these capacities do not measure how many threads are actively executing.

The per-sample GPAW log is line buffered when CLI progress is enabled. GPAW
already flushes completed SCF rows. There is no periodic heartbeat while inside
a long initialization, eigensolver call or MPI collective, so a quiet log alone
cannot distinguish useful work from a stalled process. Observe CPU time, process
state and memory alongside the logs. Python's unbuffered stdout setting does not
control arbitrary files opened by a calculator.

Benchmark the same fixed geometry and settings in separate output directories,
without relaxation or resumed references, comparing 3x4 and 12x1 on the same
12 allocated physical cores. Record total wall time, initialization time, SCF
iterations, seconds per iteration and peak memory. Repeat to assess variability;
SCF reductions may change iteration counts slightly. A successful multi-rank
launch establishes MPI connectivity, not speedup.
