r"""Finite difference calculation of changes in potential.

The implementation is based on finite-difference calculations of the atomic
gradients of the effective potential expressed on a real-space grid. The el-ph
couplings are obtained from LCAO representations of the atomic gradients of the
effective potential and the electronic states.

Full PAW couplings use the V, D, P and R decomposition of Chaput,
Togo and Tanaka, Phys. Rev. B 100, 174304 (2019), Eqs. (75)-(79).
The local potential and on-site Hamiltonian derivatives are obtained by
central differences. Projector derivatives are analytic. The P term includes
energy-dependent overlap corrections; R uses the nuclear derivative of the
partial-wave overlap. See tools/elph_al/README.md for conventions and tests.

The projection cache contains energy-independent kernels, not final
all-electron couplings. ElectronPhononMatrix combines them with band energies.

"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.phonons import Displacement

from gpaw.mpi import MPIComm
from gpaw.new.ase_interface import ASECalculator
from gpaw.old.calculator import GPAW as OldGPAW
from gpaw.utilities import pack_hermitian

dr_version = 2
# v1: saves natom, supercell, delta
# v2: also saves indices


class DisplacementRunner(Displacement):
    """Class for calculating the changes in effective potential.

    The derivative of the effective potential wrt atomic displacements is
    obtained from a finite difference approximation to the derivative by doing
    a self-consistent calculation for atomic displacements in the +/-
    directions. These calculations are carried out in the ``run`` member
    function.
    """

    def __init__(self,
                 atoms: Atoms,
                 calc: OldGPAW | ASECalculator,
                 select_atoms: list[str] | list[int] | None = None,
                 supercell: tuple = (1, 1, 1),
                 name: str = 'elph',
                 delta: float = 0.01,
                 calculate_forces: bool = True,
                 comm: MPIComm | None = None) -> None:
        """Initialize with base class args and kwargs.

        Parameters
        ----------
        atoms: Atoms
            The atoms to work on. Primitive cell.
        calc: GPAW
            Calculator for the supercell finite displacement calculation.
        supercell: tuple, list
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Name to use for files (default: 'elph').
        delta: float
            Magnitude of displacements. (default: 0.01 A)
        calculate_forces: bool
            If true, also calculate and store the forces.
        """

        # Init base class and make the center cell in the supercell the
        # reference cell
        super().__init__(atoms, calc=calc, supercell=supercell,
                         select_atoms=select_atoms, name=name, delta=delta,
                         center_refcell=True, comm=comm)
        self.calculate_forces = calculate_forces

    def calculate(self, atoms_N: Atoms, disp):
        return self(atoms_N)

    def __call__(self, atoms_N: Atoms) -> dict:
        """Extract effective potential and projector coefficients."""

        # Do calculation
        atoms_N.get_potential_energy()

        # Calculate forces if desired
        if self.calculate_forces:
            forces = atoms_N.get_forces()
        else:
            forces = None

        # Get calculator
        calc = atoms_N.calc
        if not isinstance(calc, (OldGPAW, ASECalculator)):
            calc = calc.dft  # unwrap DFTD3 wrapper

        # Effective potential (in Hartree) and projector coefficients
        # Note: Need to use coarse grid, because we project onto basis later
        if isinstance(calc, ASECalculator):
            potential = calc.dft.potential
            Vt_sG = potential.vt_sR.gather(broadcast=True).to_xp(np).data
            dH_all_asp = {a: pack_hermitian(dH_ii)
                          for a, dH_ii
                          in potential.dH_asii.to_cpu().gather(
                              broadcast=True).items()}
        else:
            Vt_sG = calc.hamiltonian.vt_sG
            Vt_sG = calc.wfs.gd.collect(Vt_sG, broadcast=True)
            dH_asp = calc.hamiltonian.dH_asp

            setups = calc.wfs.setups
            nspins = calc.wfs.nspins

            dH_all_asp = {}
            for a, setup in enumerate(setups):
                ni = setup.ni
                nii = ni * (ni + 1) // 2
                dH_tmp_sp = np.zeros((nspins, nii))
                if a in dH_asp:
                    dH_tmp_sp[:] = dH_asp[a]
                calc.wfs.gd.comm.sum(dH_tmp_sp)
                dH_all_asp[a] = dH_tmp_sp

        output = {'Vt_sG': Vt_sG, 'dH_all_asp': dH_all_asp}
        if forces is not None:
            output['forces'] = forces
        return output

    def save_info(self) -> None:
        with self.cache.lock('info') as handle:
            if handle is not None:
                info = {'natom': len(self.atoms), 'supercell': self.supercell,
                        'indices': self.indices, 'delta': self.delta,
                        'dr_version': dr_version,
                        'cell': self.atoms.cell.array,
                        'numbers': self.atoms.numbers,
                        'positions': self.atoms.positions}
                handle.save(info)

    def run(self) -> None:
        """Run the calculations for the required displacements."""
        # Save some information about this run
        self.save_info()
        Displacement.run(self)
