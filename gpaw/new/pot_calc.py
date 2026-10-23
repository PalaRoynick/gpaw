"""
==  ==========
R
r
G
g
h
x   r or h
==  ==========

"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

import numpy as np
from ase.units import Ha

from gpaw.core.arrays import XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UGArray
from gpaw.new import trace, zips
from gpaw.new.energies import DFTEnergies
from gpaw.new.logger import indent
from gpaw.new.potential import Potential
from gpaw.new.xc import Functional
from gpaw.setup import Setup
from gpaw.spinorbit import soc as soc_terms
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.gpu import as_np, cupy as cp
from gpaw.utilities import pack_density, pack_hermitian, unpack_hermitian
from gpaw.mpi import serial_comm


class PotentialCalculator:
    def __init__(self,
                 xc: Functional,
                 poisson_solver,
                 setups: list[Setup],
                 *,
                 grid,
                 wf_grid,  # grid for wave functions
                 wf_atomdist,  # atom-distribution for wave functions
                 kpt_band_comm,  # complementary comm for wave functions
                 relpos_ac: Array2D,
                 extensions: list | None = None,
                 soc: bool = False):
        self.poisson_solver = poisson_solver
        self.xc = xc
        self.setups = setups
        self.relpos_ac = relpos_ac
        self.soc = soc
        self.extensions: list = extensions or []

        self.distributed_density = grid.comm.size != wf_grid.comm.size
        self.domain_complementary_comm = (
            serial_comm if self.distributed_density else kpt_band_comm)
        self.grid = grid
        self.wf_grid = wf_grid
        self.wf_atomdist = wf_atomdist
        self.kpt_band_comm = kpt_band_comm

    def redist(self, potential: Potential) -> Potential:
        """Redistribute potential.

        If the density is distributed over the entire "world" then
        we redistribute it to self.wf_grid and self.wf_atomdist and the
        complementary communicator self.kpt_band_comm.
        """
        if not self.distributed_density:
            return potential
        return potential.redist(
            self.wf_grid,
            None,
            self.wf_atomdist,
            serial_comm,
            self.kpt_band_comm)

    def __str__(self):
        return (f'{self.poisson_solver}\n'
                f'xc functional:\n{indent(self.xc)}\n')

    @property
    def charge(self):
        return sum(ext.charge for ext in self.extensions)

    def calculate_pseudo_potential(self,
                                   density,
                                   vHt_x: XArray | None
                                   ) -> tuple[dict[str, float],
                                              UGArray,
                                              UGArray,
                                              XArray,
                                              AtomArrays,
                                              float]:
        raise NotImplementedError

    def move(self, relpos_ac, atomdist, wf_atomdist):
        self.wf_atomdist = wf_atomdist
        self.xc.paw_xc_batches = None
        for ext in self.extensions:
            ext.move_atoms(relpos_ac)

        if self.xc.name == 'QNA':
            # Special case for QNA:
            getattr(self.xc.xc, 'set_positions', lambda relpos_ac: None)(
                relpos_ac)

    def calculate_charges(self, vHt_x):
        raise NotImplementedError

    def restrict(self, a_r, a_R=None):
        raise NotImplementedError

    def calculate_without_orbitals(self,
                                   density,
                                   ibzwfs=None,
                                   vHt_x: XArray | None = None,
                                   ) -> tuple[Potential,
                                              DFTEnergies,
                                              AtomArrays]:
        xc = self.xc
        if xc.exx_fraction != 0.0:
            from gpaw.new.xc import create_functional
            self.xc = create_functional('PBE', xc.grid)
        potential, energies, V_al = self.calculate(
            density, ibzwfs, vHt_x)
        if xc.exx_fraction != 0.0:
            self.xc = xc
        return potential, energies, V_al

    @trace
    def calculate(self,
                  density,
                  ibzwfs=None,
                  vHt_x: XArray | None = None,
                  ) -> tuple[Potential, DFTEnergies, AtomArrays]:
        energies, vt_sR, dedtaut_sr, vHt_x, V_aL, e_stress = (
            self.calculate_pseudo_potential(density, vHt_x))

        energies['external'] = 0.0
        for ext in self.extensions:
            energies['external'] += ext.update_potential(vt_sR, density)

        e_kinetic = 0.0
        for spin, (vt_R, nt_R) in enumerate(zips(vt_sR, density.nt_sR)):
            e_kinetic -= vt_R.integrate(nt_R)
            if spin < density.ndensities:
                e_kinetic += vt_R.integrate(density.nct_R)

        if dedtaut_sr is not None:
            dedtaut_sR = self.restrict(dedtaut_sr)
            for dedtaut_R, taut_R in zips(dedtaut_sR,
                                          density.taut_sR):
                e_kinetic -= dedtaut_R.integrate(taut_R)
                e_kinetic += dedtaut_R.integrate(density.tauct_R)
        else:
            dedtaut_sR = None

        energies['kinetic_correction'] = e_kinetic

        dH_asii, corrections = calculate_non_local_potential(
            self.setups,
            density,
            self.xc,
            V_aL,
            self.soc,
            self.extensions)

        for ext in self.extensions:
            dct = ext.get_energy_contributions()
            for name, e in dct.items():
                assert name not in energies
                energies[name] = e

        energies['spinorbit'] = 0.0
        for key, e in corrections.items():
            if False:
                print(f'{key:10} {energies[key] * Ha:15.9f} {e * Ha:15.9f}')
            energies[key] += e

        potential = Potential(vt_sR, dH_asii, dedtaut_sR, vHt_x, e_stress)

        potential = self.redist(potential)

        return (potential,
                DFTEnergies(**energies),
                V_aL)


@trace
def calculate_non_local_potential(setups,
                                  density,
                                  xc,
                                  V_aL,
                                  soc: bool,
                                  extensions
                                  ) -> tuple[AtomArrays,
                                             dict[str, float]]:
    dtype = float if density.ncomponents < 4 else complex
    energy_corrections: DefaultDict[str, float] = defaultdict(float)

    D_asii = density.D_asii
    xc_corr = xc.calculate_paw_corrections(setups, density.D_asii, V_aL)

    if xc_corr is None:
        D_asii = D_asii.to_xp(np)
        V_aL = V_aL.to_xp(np)
    dH_asii = D_asii.layout.new(dtype=dtype).empty(density.ncomponents)

    for a, D_sii in D_asii.items():
        V_L = V_aL[a]
        setup = setups[a]
        dH_sii, corrections = calculate_non_local_potential1(
            setup, xc, D_sii, V_L, soc, extensions, a,
            xc_correction=xc_corr[a] if xc_corr is not None else None)
        dH_asii[a][:] = dH_sii
        for key, e in corrections.items():
            energy_corrections[key] += e

    # Sum over domain:
    names = ['kinetic_correction', 'coulomb', 'zero', 'xc', 'external',
             'spinorbit']
    energies = np.array([energy_corrections[name] for name in names])
    density.D_asii.layout.atomdist.comm.sum(energies)

    return (dH_asii.to_xp(density.D_asii.layout.xp) if xc_corr is None
            else dH_asii, dict(zips(names, energies)))


def calculate_non_local_potential1(setup: Setup,
                                   xc: Functional,
                                   D_sii: Array3D,
                                   V_L: Array1D,
                                   soc: bool,
                                   extensions,
                                   atom_index,
                                   xc_correction=None
                                   ) -> tuple[Array3D, dict[str, float]]:
    if xc_correction is None:
        # Legacy xc calculation
        ncomponents = len(D_sii)
        ndensities = 2 if ncomponents == 2 else 1

        D_sp = np.array([pack_density(D_ii.real) for D_ii in as_np(D_sii)])

        D_p = D_sp[:ndensities].sum(0)

        dH_p = (setup.K_p + setup.M_p +
                setup.MB_p + 2.0 * setup.M_pp @ D_p +
                setup.Delta_pL @ V_L)
        e_kinetic = setup.K_p @ D_p + setup.Kc
        e_zero = setup.MB + setup.MB_p @ D_p
        e_coulomb = setup.M + D_p @ (setup.M_p + setup.M_pp @ D_p)
        e_soc = 0.0
        dH_sp = np.zeros_like(
            D_sp, dtype=float if ncomponents < 4 else complex)

        if soc:
            dHsoc_sii = soc_terms(setup, xc.xc, D_sp)
            e_soc += (D_sii[1:4] * dHsoc_sii).sum().real
            dH_sp[1:4] = pack_hermitian(dHsoc_sii)

        e_xc = xc.calculate_paw_correction(
            setup, D_sp, dH_sp, a=atom_index)
        dH_sp[:ndensities] += dH_p
        dH_sii = unpack_hermitian(dH_sp)
    else:
        e_xc = xc_correction['e_xc']
        e_kinetic = xc_correction['e_kin']
        e_zero = xc_correction['e_0']
        e_coulomb = xc_correction['e_cou']
        e_soc = xc_correction['e_soc']
        dH_sp = xc_correction['gradient_sp']
        dH_sii = xc_correction['gradient_sii']

    if setup.hubbard_u is not None:
        eU, dHU_sii = setup.hubbard_u.calculate(setup, as_np(D_sii))
        e_xc += eU
        if isinstance(dH_sii, np.ndarray):
            dH_sii += dHU_sii
        else:
            dH_sii += cp.asarray(dHU_sii)

    for extension in extensions:
        e_xc += extension.update_non_local_hamiltonian(
            as_np(D_sii), setup, atom_index, as_np(dH_sii))

    e_kinetic -= as_np((D_sii * dH_sii).sum().real)

    return dH_sii, {'kinetic_correction': e_kinetic,
                    'coulomb': e_coulomb,
                    'zero': e_zero,
                    'xc': e_xc,
                    'external': 0.0,  # e_external,
                    'spinorbit': e_soc}
