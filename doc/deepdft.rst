DeepDFT development branch
==========================

This branch prepares an optional integration of the
`DeepDFT electron-density models <https://github.com/peterbjorgensen/DeepDFT>`_
with GPAW. It does not yet provide a DeepDFT-backed GPAW calculator, an SCF
initialization interface, or validated ML energies, forces, stress, or dynamics.
Normal GPAW installation does not require the submodule or ML dependencies.

For automated Linux source builds, CPU validation and an optional NVIDIA V100
training profile, see the `detailed Linux guide <deepdft-linux.md>`_.

Obtain the dependency
---------------------

Work in the personal repository and development branch::

    git clone https://github.com/PalaRoynick/gpaw.git
    cd gpaw
    git checkout DeepDFT
    git submodule update --init --recursive

Alternatively, clone the branch and its dependency together::

    git clone --branch DeepDFT --recurse-submodules https://github.com/PalaRoynick/gpaw.git

The submodule at ``third_party/DeepDFT`` is pinned to commit
``a6bab4deb5cf05d9b46ae397b72253d04ea3c694``. Use
``git submodule update --init --recursive`` after switching branches or pulling
updates. Do not use ``git submodule update --remote`` for reproducible setup.
Checking out a branch alone does not download its submodules.

The external repository includes pretrained checkpoints (about 129 MiB) under
its own version control. This repository tracks only the submodule commit,
not copies of those artifacts. Inspect the dependency's LICENSE before
redistributing its code or models. Do not commit generated datasets or models.

Install
-------

Prefer the pinned automated scripts in the `Linux guide <deepdft-linux.md>`_.
The commands below are a manual, unpinned CPU-only alternative.

Use Python 3.11 or newer and the normal GPAW C++ compiler, BLAS, LibXC and PAW
data prerequisites described in :ref:`installation`. For parallel GPAW builds,
install MPI development headers and use a compatible MPI C++ compiler in
``siteconfig.py`` (for example, ``compiler = 'mpicxx'``).

In an activated virtual environment, from the GPAW checkout root::

    python -m pip install 'setuptools>=77.0.3' numpy 'pybind11>=3.0.1'
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install --no-build-isolation -e '.[deepdft,devel]'

The CPU-only PyTorch install avoids downloading CUDA dependencies for these
checks. GPU use requires a separately selected compatible PyTorch installation.
ASAP3 may compile from source and needs a C++ compiler and, if detected, MPI
development headers. The optional ``deepdft`` extra includes ``lz4``, which the
external data loader imports but omits from its requirements file.

DeepDFT is a collection of scripts with absolute imports, not an installable
Python package. The smoke tests add its directory to the import path locally;
GPAW does not add it globally or package the external source into its wheel.
To run the original CLI, use its working directory, for example::

    cd third_party/DeepDFT
    python predict_with_model.py --help

Validation
----------

Run from the GPAW checkout root, in the same environment::

    gpaw test
    mpiexec --bind-to none -n 2 gpaw test
    python -m pytest -q gpaw/test/deepdft/test_smoke.py

The MPI command assumes OpenMPI; adapt launcher options to the installed MPI
implementation. Verify that the parallel calculation reports two cores.
The old ``gpaw -P`` shortcut is deprecated in current GPAW.

The optional tests are skipped if the submodule or ML dependencies are absent;
an integration validation must report two passed tests, not skips. They check:

* A small periodic H2 finite-difference GPAW calculation and its spin-summed
  all-electron density, including the integrated count of two electrons.
* Export to Gaussian cube with explicit e/angstrom^3 to e/Bohr^3 conversion,
  then import through DeepDFT and comparison against the GPAW density and
  atomic positions within cube serialization precision.
* CPU inference on 32 sampled probes with each supplied QM9 SchNet and PaiNN
  checkpoint, with finite output and deterministic evaluation.

The QM9 models use nonperiodic molecular graphs and were trained with VASP
references. These checks deliberately make no numerical accuracy claim against
the periodic GPAW calculation. Density representation, boundary conditions and
reference-data compatibility must be established before training or physics
benchmarks; finite predictions alone do not validate the physics.

Initial integration runtime: Python 3.12.14, GPAW 26.7.1b1, ASE 3.29.0, NumPy 2.3.5,
SciPy 1.17.0, PyTorch 2.14.1+cpu, ASAP3 3.13.11, lz4 4.4.5, LibXC 5.2.3,
and OpenMPI 5.0.7. These initial checks covered CPU inference only. The later
pinned Linux profile also tests CPU forward/backward/optimizer execution;
see the Linux guide for its versions and separate V100 validation commands.
GPU execution, MPI inference and full GPAW regression suites have not been
validated in this development environment.
