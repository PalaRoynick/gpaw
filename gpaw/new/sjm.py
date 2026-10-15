from __future__ import annotations

import numpy as np
from ase.units import Bohr

from gpaw.core import pw_to_grid
from gpaw.extensions import Extension
from gpaw.extensions.jellium import (FixedPotentialJelliumExtension,
                                     JelliumExtension)
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.solvation.extension import Solvation, SolvationExtension


class SJM(Solvation):
    name = 'sjm'

    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions,
                 jelliumregion: dict | None = None,
                 target_potential: float | None = None,  # eV
                 excess_electrons: float = 0.0,
                 tol: float = 0.001):  # eV
        super().__init__(cavity, dielectric, interactions,
                         dipolelayer={'zero_vacuum': True})
        self.jelliumregion = jelliumregion or {}
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.tol = tol
        raise NotImplementedError(
            'SJM is not yet validated in gpaw.new. '
            'Use gpaw.solvation.sjm.SJM instead.')

    def build(self, builder: DFTComponentsBuilder) -> SJMExtension:
        solvation = super().build(builder)
        grid = builder.fine_grid
        relpos_ac = builder.relpos_ac
        h = grid.cell_cv[2, 2] * Bohr
        z1 = relpos_ac[:, 2].max() * h + 3.0  # Å
        z2 = self.jelliumregion.get('top', h - 1.0)  # Å
        mask_r = grid.empty()
        R_rv = grid.xyz()
        mask_r.data[:] = np.logical_and(R_rv[:, :, :, 2] > z1 / Bohr,
                                        R_rv[:, :, :, 2] < z2 / Bohr)
        if self.target_potential is None:
            jellium = JelliumExtension(
                mask_r,
                charge=self.excess_electrons,
                pw=builder.interpolation_desc)
        else:
            jellium = FixedPotentialJelliumExtension(
                mask_r,
                workfunction_target=self.target_potential,
                excess_electrons_guess=self.excess_electrons,
                tolerance=self.tol,
                pw=builder.interpolation_desc)
        return SJMExtension(solvation, jellium)

    def todict(self):
        dct = super().todict()
        dct.pop('dipolelayer', None)
        dct.update(
            jelliumregion=self.jelliumregion,
            target_potential=self.target_potential,
            excess_electrons=self.excess_electrons,
            tol=self.tol)
        return dct


class SJMExtension(Extension):
    name = 'sjm'

    def __init__(self,
                 solvation: SolvationExtension,
                 jellium: JelliumExtension):
        self.solvation = solvation
        self.jellium = jellium
        self.charge = jellium.charge
        self.dielectric = solvation.dielectric

    def create_poisson_solver(self, grid, pw, charge, xp):
        return self.solvation.create_poisson_solver(
            grid, pw, charge=charge, xp=xp)

    def post_scf_convergence(self,
                             ibzwfs,
                             nelectrons,
                             occ_calc,
                             mixer,
                             log) -> bool:
        self.solvation.post_scf_convergence(
            ibzwfs, nelectrons, occ_calc, mixer, log)
        converged = self.jellium.post_scf_convergence(
            ibzwfs, nelectrons, occ_calc, mixer, log)
        self.charge = self.jellium.charge
        return converged

    def move_atoms(self, relpos_ac):
        raise NotImplementedError(
            'SJM in gpaw.new does not yet support moving atoms.')

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.jellium.update1(nt_r)

    def update1pw(self, nt_g):
        nt_r = pw_to_grid(nt_g, self.jellium.mask_r.desc)
        self.solvation.update1(nt_r)
        self.jellium.update1pw(nt_g)

    def update2(self, nt_r, vHt_r, vt_sr) -> None:
        self.solvation.update2(nt_r, vHt_r, vt_sr)

    def update2pw(self, nt_g, vHt_h, vt_sr) -> None:
        self.solvation.update2pw(nt_g, vHt_h, vt_sr)

    def get_energy_contributions(self) -> dict[str, float]:
        return self.solvation.get_energy_contributions()
