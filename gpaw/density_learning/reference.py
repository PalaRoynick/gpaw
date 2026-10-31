"""Reusable GPAW reference calculator and provenance-rich cube export."""

from dataclasses import asdict, dataclass, field
from time import monotonic

import numpy as np
from ase.optimize import BFGS

from gpaw.density_learning.io import (
    CubeWriter,
    external_directory,
    file_hash,
    geometry_hash,
    save_json,
)


@dataclass(frozen=True)
class ReferenceSettings:
    mode: str = "fd"
    grid_spacing: float = 0.16
    cutoff_ev: float = 500.0
    xc: str = "PBE"
    kpoint_density: float = 4.0
    smearing_ev: float = 0.02
    density_kind: str = "pseudo"
    grid_refinement: int = 1
    extra_bands_percent: int = 30
    charge: float = 0.0
    initial_moment: float = 0.0
    maxiter: int = 250
    energy_tolerance: float = 1e-6
    density_tolerance: float = 1e-5
    eigenstate_tolerance: float = 1e-8
    multipoles: int = 4
    dipole_correction: bool = True
    expected_valence: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in {"fd", "pw"}:
            raise ValueError("Use fd or pw; LCAO is not a reference-density default")
        if self.density_kind not in {"pseudo", "valence", "all-electron"}:
            raise ValueError("Unknown density representation")
        if self.grid_refinement not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("Density refinement must be a power of two from 1 to 32")
        if (
            min(
                self.grid_spacing, self.cutoff_ev, self.smearing_ev, self.kpoint_density
            )
            <= 0
        ):
            raise ValueError(
                "Grid, cutoff, smearing and k-point density must be positive"
            )
        if self.extra_bands_percent < 0 or self.maxiter < 1:
            raise ValueError("Invalid empty-band or SCF iteration budget")


class DensityCalculator:
    """Collective MPI calculation; only rank zero writes files.

    All ranks must call calculate with identical inputs and the same output.
    Pseudo-density integrals are measured, not forced to a valence count.
    """

    def __init__(
        self,
        settings=ReferenceSettings(),
        communicator=None,
        max_atoms=256,
        max_density_voxels=100_000_000,
        max_wavefunction_gib=None,
        gpu=None,
        device="auto",
        experimental_fd_gpu=False,
    ):
        self.settings = settings
        self.device = ("cuda" if gpu else "cpu") if gpu is not None else device
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Reference device must be auto, cpu or cuda")
        self.experimental_fd_gpu = experimental_fd_gpu
        self.gpu = False
        if self.device == "cuda" and settings.mode == "fd" and not experimental_fd_gpu:
            raise ValueError("FD CUDA requires --experimental-fd-gpu (experimental_fd_gpu=True)")
        self.communicator = communicator
        self.max_atoms = max_atoms
        self.max_density_voxels = max_density_voxels
        if max_wavefunction_gib is not None and (
            not np.isfinite(max_wavefunction_gib) or max_wavefunction_gib <= 0
        ):
            raise ValueError("Wavefunction storage budget must be positive and finite")
        self.max_wavefunction_gib = max_wavefunction_gib

    def plan(self, atoms):
        from ase.calculators.calculator import kpts2sizeandoffsets

        s = self.settings
        mesh = (
            kpts2sizeandoffsets(density=s.kpoint_density, gamma=True, atoms=atoms)[0]
            if atoms.pbc.any()
            else np.ones(3, int)
        )
        shape = np.maximum(
            4, np.ceil(atoms.cell.lengths() / s.grid_spacing / 4).astype(int) * 4
        )
        known = all(
            symbol in s.expected_valence for symbol in atoms.get_chemical_symbols()
        )
        valence = (
            sum(s.expected_valence[z] for z in atoms.get_chemical_symbols()) - s.charge
            if known
            else None
        )
        bands = (
            int(
                np.ceil(
                    (valence + abs(s.initial_moment))
                    / 2
                    * (1 + s.extra_bands_percent / 100)
                )
            )
            if known
            else None
        )
        wavefunctions = (
            16
            * int(np.prod(shape))
            * int(np.prod(mesh))
            * bands
            * (2 if s.initial_moment else 1)
            if known and s.mode == "fd"
            else None
        )
        density_voxels = int(np.prod(shape * s.grid_refinement))
        return {
            "natoms": len(atoms),
            "mode": s.mode,
            "execution_device": self.device,
            "execution_device_requested": self.device,
            "kpoint_mesh": list(map(int, mesh)),
            "conservative_fd_grid_shape": shape.tolist(),
            "estimated_bands_per_spin": bands,
            "wavefunction_storage_gib_total": (
                wavefunctions / 2**30 if wavefunctions is not None else None
            ),
            "density_voxel_budget_satisfied": density_voxels <= self.max_density_voxels,
            "wavefunction_budget_gib": self.max_wavefunction_gib,
            "wavefunction_budget_satisfied": (
                wavefunctions / 2**30 <= self.max_wavefunction_gib
                if wavefunctions is not None and self.max_wavefunction_gib is not None
                else None
            ),
            "atom_budget_satisfied": len(atoms) <= self.max_atoms,
            "conservative_density_voxels": density_voxels,
            "estimated_cube_bytes": density_voxels * 22,
            "warning": "Storage estimate excludes workspaces, projectors and MPI replication; not a RAM guarantee",
        }

    def calculate(
        self,
        atoms,
        output,
        sample_id,
        group_id,
        family="unspecified",
        relax_steps=0,
        fmax=0.02,
        progress=None,
    ):
        from gpaw import GPAW, PW, FermiDirac, Mixer, __version__
        from gpaw.mpi import world, broadcast

        comm = self.communicator if self.communicator is not None else world
        s = self.settings
        from gpaw.density_learning.devices import select_reference_device
        execution_device, device_reason = select_reference_device(
            self.device, s.mode, comm.size, self.experimental_fd_gpu)
        self.gpu = execution_device == "cuda"
        if self.gpu and s.mode == "fd" and relax_steps:
            raise RuntimeError(
                "FD CUDA relaxation is disabled pending force-backend validation; "
                "localized-function derivatives may enter CPU-only C routines. "
                "Use --device cpu for FD relaxation."
            )
        if not sample_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in sample_id
        ):
            raise ValueError("Use a simple ASCII sample identifier")
        if not len(atoms) or len(atoms) > self.max_atoms:
            raise ValueError(
                "Reference atom budget exceeded; raise max_atoms explicitly if justified"
            )
        plan = self.plan(atoms)
        if s.mode == "fd" and not plan["density_voxel_budget_satisfied"]:
            raise MemoryError("Planned density export exceeds the voxel budget before SCF")
        if self.max_wavefunction_gib is not None:
            if plan["wavefunction_storage_gib_total"] is None:
                raise ValueError(
                    "Wavefunction budget requires FD mode and expected_valence for all species"
                )
            if not plan["wavefunction_budget_satisfied"]:
                raise MemoryError(
                    "Planned wavefunction storage exceeds the budget before SCF; "
                    "this budget excludes workspaces and is not a total RAM limit"
                )
        output = external_directory(output)
        cube_path = output / (sample_id + ".cube")
        if cube_path.exists() or cube_path.with_suffix(".json").exists():
            raise FileExistsError(cube_path)
        atoms = atoms.copy()
        input_hash = geometry_hash(atoms)
        if s.initial_moment:
            atoms.set_initial_magnetic_moments(
                [s.initial_moment / len(atoms)] * len(atoms)
            )
        kwargs = dict(
            mode="fd" if s.mode == "fd" else PW(s.cutoff_ev),
            xc=s.xc,
            occupations=FermiDirac(s.smearing_ev),
            kpts={"density": s.kpoint_density, "gamma": True}
            if atoms.pbc.any()
            else (1, 1, 1),
            symmetry="off",
            spinpol=bool(s.initial_moment),
            charge=s.charge,
            nbands=f"{100 + s.extra_bands_percent}%",
            maxiter=s.maxiter,
            mixer=Mixer(0.05, 5, 50),
            convergence={
                "energy": s.energy_tolerance,
                "density": s.density_tolerance,
                "eigenstates": s.eigenstate_tolerance,
            },
            communicator=comm,
            parallel={"gpu": self.gpu},
            txt=str(output / (sample_id + ".log")),
        )
        if s.mode == "fd":
            kwargs["h"] = s.grid_spacing
            if not atoms.pbc.any() and s.multipoles:
                kwargs["poissonsolver"] = {
                    "name": "MomentCorrectionPoissonSolver",
                    "poissonsolver": {"name": "fast"},
                    "moment_corrections": s.multipoles,
                }
            elif list(atoms.pbc) == [True, True, False] and s.dipole_correction:
                kwargs["poissonsolver"] = {"dipolelayer": "xy"}
        elif not atoms.pbc.all():
            raise ValueError(
                "Use FD for open boundaries in this initial reference wrapper"
            )
        if self.gpu and s.mode == "fd":
            from gpaw.density_learning.poisson import HostStagedPoisson
            kwargs["poissonsolver"] = HostStagedPoisson(kwargs.get("poissonsolver", {}))
        started = monotonic()

        def report(event, **details):
            if progress is not None and comm.rank == 0:
                progress({"event": event, "sample_id": sample_id,
                          "elapsed_seconds": monotonic() - started, **details})

        report("initialization_started", mpi_ranks=comm.size,
               execution_device=execution_device, device_reason=device_reason,
               log_file=str(output / (sample_id + ".log")))
        atoms.calc = GPAW(**kwargs)
        if progress is not None and not getattr(atoms.calc, "old", False):
            # Flush initialization lines too; GPAW already flushes SCF rows.
            if comm.rank == 0:
                atoms.calc.log.fd.reconfigure(line_buffering=True)
            atoms.calc.hooks["scf_step"] = lambda ctx: report(
                "scf_step", iteration=int(ctx.niter))
            atoms.calc.hooks["converged"] = lambda: report("scf_converged")
        relaxed = False
        if relax_steps:
            relaxed = BFGS(atoms, logfile=str(output / (sample_id + "-relax.log"))).run(
                fmax=fmax, steps=relax_steps
            )
            if not relaxed:
                raise RuntimeError(
                    "DFT relaxation did not reach fmax; reference was not exported"
                )
        energy = float(atoms.get_potential_energy())
        calc = atoms.calc
        if not getattr(calc, "old", False):
            actual = int(np.prod(calc.dft.density.nt_sR.desc.size * s.grid_refinement))
            if actual > self.max_density_voxels:
                raise MemoryError(
                    "Density export exceeds the voxel budget; inspect the plan and raise it explicitly"
                )
        report("density_export_started")
        options = dict(gridrefinement=s.grid_refinement, broadcast=False)
        if s.density_kind == "pseudo":
            # Current GPAW's ASE get_pseudo_density adds compensation charges.
            # The public Densities API makes the smooth target explicit.
            if not getattr(calc, "old", False):
                field = calc.dft.densities().pseudo_densities(
                    grid_refinement=s.grid_refinement, add_compensation_charges=False
                )
                field = field.to_xp(np).gather(broadcast=False)
                density = None if field is None else field.data.sum(axis=0)
            else:
                density = calc.get_pseudo_density(**options)
        else:
            density = calc.get_all_electron_density(
                skip_core=s.density_kind == "valence", **options
            )
        final_magnetic_moment = float(calc.get_magnetic_moment())
        metadata = None
        error = None
        wrote_cube = False
        if comm.rank == 0:
            try:
                if not np.isfinite(density).all():
                    raise ValueError("Nonfinite reference density")
                integral = float(density.sum() * atoms.get_volume() / density.size)
                valence = float(calc.get_number_of_electrons())
                expected = (
                    float(atoms.numbers.sum()) - s.charge
                    if s.density_kind == "all-electron"
                    else valence
                )
                if s.density_kind != "pseudo" and abs(integral - expected) > max(
                    1e-4, expected * 1e-5
                ):
                    raise ValueError(
                        f"Reconstructed density integrates to {integral:.9g}, expected {expected:.9g}; "
                        "increase grid_refinement and establish grid convergence"
                    )
                with CubeWriter(cube_path, atoms, density.shape) as writer:
                    for slab in density:
                        writer.write(slab)
                wrote_cube = True
                setups = []
                for setup in calc.setups:
                    data = setup.data
                    expected_valence = s.expected_valence.get(setup.symbol)
                    if expected_valence is not None and setup.Nv != expected_valence:
                        raise ValueError(
                            "Actual PAW valence does not match the requested protocol"
                        )
                    setups.append(
                        {
                            "symbol": setup.symbol,
                            "valence": float(setup.Nv),
                            "fingerprint": str(
                                getattr(data, "fingerprint", "unavailable")
                            ),
                        }
                    )
                metadata = {
                    "schema": "gpaw-density-v1",
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "family": family,
                    "density_kind": s.density_kind,
                    "pseudo_compensation_charges": False,
                    "electrostatic_source_ready": False,
                    "electrostatic_source_note": (
                        "Density-only export: ionic and PAW electrostatic contributions "
                        "are not exported. Do not combine smooth pseudo density with "
                        "point valence ions or normalize it to the valence count."
                    ),
                    "mpi_ranks": comm.size,
                    "execution_device": execution_device,
                    "execution_device_requested": self.device,
                    "device_reason": device_reason,
                    "poisson_execution_device": "cpu" if self.gpu and s.mode == "fd" else execution_device,
                    "experimental_fd_gpu": bool(self.gpu and s.mode == "fd"),
                    "final_magnetic_moment": final_magnetic_moment,
                    "input_geometry_sha256": input_hash,
                    "units": "electron/angstrom^3",
                    "cube_units": "electron/Bohr^3",
                    "cell_angstrom": atoms.cell.tolist(),
                    "pbc": atoms.pbc.tolist(),
                    "positions_angstrom": atoms.positions.tolist(),
                    "numbers": atoms.numbers.tolist(),
                    "shape": list(density.shape),
                    "origin_angstrom": [0.0, 0.0, 0.0],
                    "integral_electrons": integral,
                    "valence_electrons": valence,
                    "pseudo_augmentation_difference": valence - integral
                    if s.density_kind == "pseudo"
                    else None,
                    "energy_ev": energy,
                    "natoms": len(atoms),
                    "gpaw_version": __version__,
                    "kpoints_bz": calc.get_bz_k_points().tolist(),
                    "relaxation": {
                        "requested_steps": relax_steps,
                        "fmax": fmax,
                        "converged": bool(relaxed),
                    },
                    "open_shell_review_required": bool(
                        not atoms.pbc.any()
                        and round(valence) % 2
                        and not s.initial_moment
                    ),
                    "settings": asdict(s),
                    "setups": setups,
                    "relaxed": bool(relaxed),
                    "spin_ground_state_verified": False,
                    "cube_sha256": file_hash(cube_path),
                }
                save_json(cube_path.with_suffix(".json"), metadata)
            except Exception as exception:
                if wrote_cube:
                    cube_path.unlink(missing_ok=True)
                    cube_path.with_suffix(".json").unlink(missing_ok=True)
                error = f"{type(exception).__name__}: {exception}"
        (error,) = broadcast((error,) if comm.rank == 0 else None, comm=comm)
        if error:
            raise RuntimeError(error)
        report("reference_completed")
        return broadcast(metadata, comm=comm)
