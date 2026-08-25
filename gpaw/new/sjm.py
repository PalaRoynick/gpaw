from __future__ import annotations

import numpy as np
from ase.units import Bohr

from gpaw.core import PWArray, PWDesc, UGArray
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.extensions import Extension
from gpaw.extensions.jellium import (FixedPotentialJelliumExtension,
                                     JelliumExtension)
from gpaw.new.poisson import PoissonSolverWrapper
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.new.solvation import Solvation, SolvationExtension
from gpaw.mpi import broadcast_float


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
                 dipolelayer: bool = True,
                 backwards_compatible: bool = True,  # Only applies if fd mode
                 tol: float = 0.01):  # eV
        super().__init__(cavity, dielectric, interactions)
        self.jelliumregion = jelliumregion or {}
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.dipolelayer = dipolelayer
        self.backwards_compatible = backwards_compatible
        self.tol = tol

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
        return SJMExtension(solvation, jellium, dipolelayer=self.dipolelayer,
                            backwards_compatible=self.backwards_compatible)

    def todict(self):
        dct = super().todict()
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
                 jellium: JelliumExtension,
                 dipolelayer: bool = True,
                 backwards_compatible: bool = True):
        self.solvation = solvation
        self.jellium = jellium
        self.charge = jellium.charge
        self.excess_electrons = jellium.charge
        self.dielectric = solvation.dielectric
        self.dipolelayer = dipolelayer
        self.backwards_compatible = backwards_compatible

    def create_poisson_solver(self, grid, pw, charge, xp, zero_vacuum=False):
        if isinstance(pw, PWDesc):
            if self.solvation.psolver in [None, 'FDsolver']:
                from gpaw.new.pw.poisson import FDPWsolver
                return FDPWsolver(
                    pw, grid, self.dielectric, dipolelayer=self.dipolelayer,
                    zero_vacuum=zero_vacuum,
                    backwards_compatible=self.backwards_compatible)
            else:
                from gpaw.new.pw.poisson import ConjugateGradientPoissonSolver
                return ConjugateGradientPoissonSolver(
                    pw, grid, self.dielectric, zero_vacuum=zero_vacuum)
            # from gpaw.new.sjm import SJMPWPoissonSolver
            # return SJMPWPoissonSolver(pw, environment.dielectric, grid)

        ps = self.solvation.create_poisson_solver(
            grid, pw, charge=charge, xp=xp).solver
        return SJMPoissonSolver(ps, self.solvation.dielectric,
                                self.dipolelayer, False,
                                self.backwards_compatible)

    def post_scf_convergence(self,
                             ibzwfs,
                             nelectrons,
                             occ_calc,
                             mixer,
                             log) -> bool:
        converged = self.jellium.post_scf_convergence(
            ibzwfs, nelectrons, occ_calc, mixer, log)
        self.excess_electrons = self.jellium.charge
        self.charge = self.jellium.charge

        return converged

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.jellium.update1(nt_r)

    def update1pw(self, nt_g):
        grid = self.jellium.mask_r.desc
        nt_r = grid.empty()
        nt_r.scatter_from(nt_g.ifft(grid=grid.new(comm=None))
                          if nt_g is not None else None)
        self.solvation.update1(nt_r)
        self.jellium.update1pw(nt_g)

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        return self.solvation.update2(nt_r, vHt_r, vt_sr)

    def todict(self):
        dct = super().todict()
        dct.update(
            jelliumregion=self.jelliumregion,
            target_potential=self.target_potential,
            excess_electrons=self.excess_electrons,
            tol=self.tol)
        print('todict SJMext:', dct)
        return dct


class SJMPoissonSolver(PoissonSolverWrapper):
    def __init__(self, solver, dielectric,
                 dipolelayer: bool = True,
                 pw: bool = True,
                 backwards_compatible: bool = True):
        super().__init__(solver)
        self.dipolelayer = dipolelayer
        self.backwards_compatible = backwards_compatible
        self.pw = pw

    def solve(self,
              vHt_r,
              rhot_r) -> float:
        self.solver.solve(vHt_r.data, rhot_r.data)
        correction = 0.0
        if self.dipolelayer:
            eps_r = vHt_r.desc.from_data(self.solver.dielectric.eps_gradeps[0])
            eps0_r = eps_r.gather()
            vHt0_r = vHt_r.gather()
            if eps0_r is not None:
                v1, v2 = vHt0_r.data[:, :, [-10, -2]].mean(axis=(0, 1))
                if self.backwards_compatible:
                    if self.pw:
                        saw_tooth = pw_modified_saw_tooth(eps0_r)
                    else:
                        saw_tooth = fd_modified_saw_tooth(eps0_r)

                    s1, s2 = saw_tooth[[-10, -2]]
                    vHt0_r.data -= (v2 - v1) / (s2 - s1) *\
                        saw_tooth[np.newaxis, np.newaxis]
                else:
                    saw_tooth = modified_3d_saw_tooth(eps0_r)
                    s1, s2 = saw_tooth[:, :, [-10, -2]].mean(axis=(0, 1))
                    vHt0_r.data -= (v2 - v1) / (s2 - s1) * saw_tooth

                correction = (v2 - v1) / (s2 - s1) / 2
                vHt0_r.data -= vHt0_r.data[:, :, -1].mean()
            vHt_r.scatter_from(vHt0_r)
            broadcast_float(correction, vHt_r.desc.comm)

        # TODO: We need to return self.correction which is should be the
        # half the difference of the two workfunctions, this does not work
        # yet, but is only an issue for the text output.
        return correction


def modified_3d_saw_tooth(eps_r, idisc=None) -> np.ndarray:
    """Create a modified saw tooth potential in 3D.
    The saw tooth potential is created by integrating 1/eps_r
    along the z direction.

    Parameters:
    eps_r : GDArray
        Relative permittivity in real space
    idisc : int, optional
        Upper index of the discontinuity in the saw tooth potential.
    Returns:
    saw_tooth_r : np.ndarray
        Saw tooth potential in real space where the slope is scaled by
        1/eps_r
    """

    if idisc is None:
        idisc = eps_r.data.mean(axis=(0, 1)).size // 8

    # Create the sawtooth with a discontinuity at idisc
    a_r = 1.0 / np.roll(eps_r.data, -idisc, axis=2)
    saw_tooth_r = np.add.accumulate(a_r, axis=2)
    saw_tooth_r /= saw_tooth_r.mean(axis=(0, 1))[-1] -\
        saw_tooth_r.mean(axis=(0, 1))[0]
    saw_tooth_r = np.roll(saw_tooth_r, idisc, axis=2)

    # Fuse the discontinuity by adding an error function ramp
    from scipy.special import erf
    ramp = np.zeros_like(saw_tooth_r)
    w = 2.0  # erf spans -0.995 to 0.995 at w=2
    er_start = 0
    erf_vals = erf(np.linspace(-w, w, idisc - er_start))

    delta = (saw_tooth_r[:, :, idisc] - saw_tooth_r[:, :, er_start]) / 2
    ramp[:, :, er_start:idisc] = erf_vals[None, None, :] * \
        delta[:, :, None] + delta[:, :, None]
    saw_tooth_r += ramp
    return saw_tooth_r


def pw_modified_saw_tooth(eps_r, idisc=None) -> np.ndarray:
    if idisc is None:
        idisc = eps_r.data.mean(axis=(0, 1)).size // 8
    a_z = 1.0 / np.roll(eps_r.data.mean(axis=(0, 1)), -idisc)
    saw_tooth_z = np.add.accumulate(a_z)
    saw_tooth_z = np.roll(saw_tooth_z, idisc)

    # Reference to the mean value to no add a net potenial
    # I think this is not really needed
    # saw_tooth_z -= saw_tooth_z.mean()

    # Fuse the discontinuity by adding an error function ramp
    from scipy.special import erf
    ramp = np.zeros_like(saw_tooth_z)

    # erf spans -0.995 to 0.995 at w=2 smaller values makes it steeper
    # larger values create discontinuities at the edges
    w = 2.0
    ramp[:idisc] = erf(np.linspace(-w, w, idisc)) * \
        (saw_tooth_z[idisc] - saw_tooth_z[0]) / 2 + \
        (saw_tooth_z[idisc] - saw_tooth_z[0]) / 2
    saw_tooth_z += ramp
    return saw_tooth_z


def fd_modified_saw_tooth(eps_r: UGArray) -> np.ndarray:
    a_z = 1.0 / eps_r.data.mean(axis=(0, 1))
    saw_tooth_z = np.add.accumulate(a_z)
    saw_tooth_z -= 0.5 * a_z  # +0.5 from z=0.0 ???
    return saw_tooth_z


# BELOW IS the TRASHBIN AND NEVER USED
class SJMPWPoissonSolver(PWPoissonSolver):
    def __init__(self, pw, dielectric):
        super().__init__(pw)
        self.dielectric = dielectric
        self.saw_tooth_g = saw_tooth(pw, 0.25)

    def solve(self, vHt_g, rhot_g):
        energy = super().solve(vHt_g, rhot_g)
        dipole = rhot_g.moment()[2]
        slope = 4 * np.pi * dipole / rhot_g.desc.volume
        vHt_g.data += slope * self.saw_tooth_g.data
        # Shift potential so that it is zero above the slab:
        shift = 0.5 * slope * rhot_g.desc.cell_cv[2, 2]
        v0 = vHt_g.boundary_value(2)
        vHt_g.data[0] -= shift + v0
        return energy


def saw_tooth_sympy():
    """Fourier-transform."""
    from sympy import Symbol, integrate, sin, var
    z = var('z')
    G = Symbol('G', positive=True)
    b = Symbol('b', positive=True)
    m = integrate(sin(G * z) * z, (z, 0, b))
    print(m)  # -b*cos(G*b)/G + sin(G*b)/G**2


def saw_tooth(pw: PWDesc, width: float = 0.5) -> PWArray:
    """Saw-tooth in reciprocal space with a slope of 1."""
    assert np.allclose(pw.cell_cv[:2, 2], 0.0)
    assert np.allclose(pw.cell_cv[2, :2], 0.0)

    m0_g, m1_g = pw.indices_cG[:2, pw.ng1:pw.ng2] == 0
    mask_g = m0_g & m1_g
    Gz_i = pw.G_plus_k_Gv[mask_g, 2]
    if pw.comm.rank == 0.0:
        assert Gz_i[0] == 0.0
        Gz_i[0] = 1.0
    L = pw.cell_cv[2, 2]
    b = L / 2
    st_i = -(np.sin(b * Gz_i) / Gz_i -
             b * np.cos(b * Gz_i)) / Gz_i * (2j / L)
    if pw.comm.rank == 0.0:
        st_i[0] = 0.0

    # Make the saw-tooth more smooth (fold with Gaussian):
    alpha = width**-2
    st_i *= np.exp(-Gz_i**2 / (4 * alpha))

    # Shift by half the cell height:
    st_i *= np.exp(1j * Gz_i * b)

    st_g = pw.zeros()
    st_g.data[mask_g] = st_i
    return st_g
