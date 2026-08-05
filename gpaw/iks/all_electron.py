from __future__ import annotations

import sys
from math import log, pi, sqrt
from typing import TYPE_CHECKING

import numpy as np
from ase.data import atomic_names

from gpaw.atom.all_electron import shoot, calculate_hartree, get_r2dvdr
from gpaw.atom.configurations import configurations
from gpaw.atom.radialgd import AERadialGridDescriptor
from gpaw.xc import XC

if TYPE_CHECKING:
    from gpaw.setup import Setup


ALPHA = 1 / 137.036


def calculate_xc(rgd, xc, n_g):
    vxc_g = np.zeros(rgd.N)
    Exc = xc.calculate_spherical(
        rgd, n_g.reshape((1, -1)), vxc_g.reshape((1, -1)))
    return vxc_g, Exc


def calculate_potentials(rgd, xc, n_g, Z):
    vHr_g = calculate_hartree(rgd, n_g, Z)
    vxc_g, Exc = calculate_xc(rgd, xc, n_g)
    vr_g = vHr_g + vxc_g * rgd.r_g
    return vr_g, vHr_g, vxc_g, Exc


def calculate_density(f_j, u_jg, r_g):
    n_g = np.dot(f_j, np.where(abs(u_jg) < 1e-160, 0, u_jg)**2) / (4 * pi)
    n_g[1:] /= r_g[1:]**2
    n_g[0] = n_g[1]
    return n_g


def get_configuration_from_setupdata(data):
    #   (atomic number, [(n, l, occ, energy), ...])
    default_config = configurations[data.symbol][1]
    configs = []
    fcore = 0
    for n, ll, f, e in default_config:
        fcore += f
        if fcore > data.Nc:
            break
        configs.append((n, ll, float(f), e))

    # valence
    for n, ll, f, e in zip(data.n_j, data.l_j,
                           data.f_j, data.eps_j):
        configs.append((n, ll, f, e))

    return configs


def calculate_density_from_density_matrix(D_ii, ae: AEKS, setup):
    """Return the spherical AE density represented by the bound PAW blocks.

    The frozen-core occupations come from the atomic calculation.  The
    valence occupations and coherences come from the bound-state blocks of
    the PAW density matrix.  Unbound/scattering projector blocks have no
    corresponding orbital in ``AllElectron`` and are therefore not included.
    """
    ni = sum(2 * l + 1 for l in setup.l_j)
    assert D_ii.shape == (ni, ni)
    n_g = ae.rgd.zeros()
    remaining_core = float(setup.Nc)
    core_indices = []
    for j, (f, u_g) in enumerate(zip(ae.f_j, ae.u_j)):
        if remaining_core <= 0.0:
            break
        core_occupation = float(f)
        n_g += core_occupation * u_g**2
        remaining_core -= core_occupation
        core_indices.append(j)
    else:
        raise ValueError("Unmatched number of electrons!")

    # Locate every bound setup state in the AE atomic configuration
    ae_j = {(n, l): j for j, (n, l) in
            enumerate(zip(ae.n_j, ae.l_j))}
    bound_states = []
    i = 0
    for n, l in zip(setup.n_j, setup.l_j):
        if n > 0:
            j = ae_j[(n, l)]
            bound_states.append((l, i, j))
        i += 2 * l + 1

    # n(r) = sum_{jj'm} D_{jm,j'm} u_j(r)u_j'(r)/(4*pi*r**2).
    valence_occupations = []
    valence_indices = []
    for l1, i1, j1 in bound_states:
        nm = 2 * l1 + 1
        for l2, i2, j2 in bound_states:
            if l1 == l2:
                weight = np.trace(D_ii[i1:i1 + nm, i2:i2 + nm])
                n_g += weight * ae.u_j[j1] * ae.u_j[j2]
                valence_occupations.append(weight)
                valence_indices.append((i1, i2))

    n_g /= 4 * pi
    n_g[1:] /= ae.r[1:]**2
    n_g[0] = n_g[1]
    return n_g, core_indices, valence_occupations, bound_states


class AEKS:

    def __init__(self, setup: Setup, xc_setup: str | dict,
                 txt='-', gpernode=150):

        assert setup.data.has_corehole is False

        self.txt = txt
        if txt == '-':
            self.fd = sys.stdout
        else:
            self.fd = open(txt, 'w')

        self.setup = setup
        self.xc_setup = XC(xc_setup)
        self.scalarrel = setup.data.type == 'scalar-relativistic'

        # Get reference state:
        nlfe_j = get_configuration_from_setupdata(setup.data)
        self.n_j = [n for n, l, f, e in nlfe_j]
        self.l_j = [l for n, l, f, e in nlfe_j]
        self.f_j = [f for n, l, f, e in nlfe_j]
        self.e_j = [e for n, l, f, e in nlfe_j]

        maxnodes = max([n - l - 1 for n, l in zip(self.n_j, self.l_j)])
        self.N = (maxnodes + 1) * gpernode
        self.beta = 0.4
        self.rgd = AERadialGridDescriptor(self.beta / self.N, 1.0 / self.N,
                                          self.N)

    def log(self, *args, **kwargs):
        self.fd.write(kwargs.get('sep', ' ').join(
            [str(arg) for arg in args]) + kwargs.get('end', '\n'))

    def initialize_wave_functions(self):
        r = self.r
        dr = self.dr
        # Initialize with Slater function:
        for l, e, u in zip(self.l_j, self.e_j, self.u_j):
            if self.symbol in ['Hf', 'Ta', 'W', 'Re', 'Os',
                               'Ir', 'Pt', 'Au']:
                a = sqrt(-4.0 * e)
            else:
                a = sqrt(-2.0 * e)

            u[:] = r**(1 + l) * np.exp(-a * r)
            norm = np.dot(u**2, dr)
            u *= 1.0 / sqrt(norm)

    def run_scf(self):
        #     beta g
        # r = ------, g = 0, 1, ..., N - 1
        #     N - g
        #
        #        rN
        # g = --------
        #     beta + r

        N = self.N
        self.log(N, 'radial gridpoints.')
        g = np.arange(N, dtype=float)
        self.r = self.rgd.r_g
        self.dr = self.rgd.dr_g
        self.d2gdr2 = self.rgd.d2gdr2()

        # Number of orbitals:
        nj = len(self.n_j)

        # Radial wave functions multiplied by radius:
        self.u_j = np.zeros((nj, self.N))

        # Effective potential multiplied by radius:
        self.vr = np.zeros(N)

        # Electron density:
        self.n = np.zeros(N)

        # Always spinpaired nspins=1
        # self.xc = XC(self.xcname)

        n_j = self.n_j
        l_j = self.l_j
        f_j = self.f_j
        e_j = self.e_j

        Z = self.Z    # nuclear charge
        r = self.r    # radial coordinate
        dr = self.dr  # dr/dg
        n = self.n    # electron density
        vr = self.vr  # effective potential multiplied by r

        vHr = np.zeros(self.N)
        self.vXC = np.zeros(self.N)

        self.initialize_wave_functions()
        n[:] = self.calculate_density()

        bar = '|------------------------------------------------|'

        niter = 0
        nitermax = 117
        qOK = log(1e-10)
        mix = 0.4

        vrold = None

        while True:
            vr[:], vHr[:], self.vXC[:], Exc = calculate_potentials(
                rgd=self.rgd, xc=self.xc, n_g=n, Z=Z)

            # calculate new total Kohn-Sham effective potential and
            # admix with old version

            if niter > 0:
                vr[:] = mix * vr + (1 - mix) * vrold
            vrold = vr.copy()

            # solve Kohn-Sham equation and determine the density change
            self.solve()
            dn = self.calculate_density() - n
            n += dn

            # estimate error from the square of the density change integrated
            q = log(np.sum((r * dn)**2))

            # check if converged and break loop if so
            if q < qOK:
                break

            niter += 1
            if niter > nitermax:
                raise RuntimeError('Did not converge!')

        self.n_scf_iter = niter

        Ekin = 0

        for f, e in zip(f_j, e_j):
            Ekin += f * e

        e_coulomb = 2 * pi * np.dot(n * r * (vHr - Z), dr)
        Ekin += -4 * pi * np.dot(n * vr * r, dr)

        self.vHr = vHr
        self.vr = vr
        self.Ekin = Ekin
        self.e_coulomb = e_coulomb
        self.Exc = Exc

    def calculate_density(self):
        """Return the electron charge density divided by 4 pi."""
        return calculate_density(self.f_j, self.u_j, self.r)

    def solve(self):
        """Solve the Schrodinger equation

        ::

             2
            d u     1  dv  du   u     l(l + 1)
          - --- - ---- -- (-- - -) + [-------- + 2M(v - e)] u = 0,
              2      2 dr  dr   r         2
            dr    2Mc                    r


        where the relativistic mass::

                   1
          M = 1 - --- (v - e)
                    2
                  2c

        and the fine-structure constant alpha = 1/c = 1/137.036
        is set to zero for non-scalar-relativistic calculations.

        On the logaritmic radial grids defined by::

              beta g
          r = ------,  g = 0, 1, ..., N - 1
              N - g

                 rN
          g = --------, r = [0; oo[
              beta + r

        the Schrodinger equation becomes::

           2
          d u      du
          --- c  + -- c  + u c  = 0
            2  2   dg  1      0
          dg

        with the vectors c0, c1, and c2  defined by::

                 2 dg 2
          c  = -r (--)
           2       dr

                  2         2
                 d g  2    r   dg dv
          c  = - --- r  - ---- -- --
           1       2         2 dr dr
                 dr       2Mc

                                    2    r   dv
          c  = l(l + 1) + 2M(v - e)r  + ---- --
           0                               2 dr
                                        2Mc
        """
        r = self.r
        dr = self.dr
        vr = self.vr

        c2 = -(r / dr)**2
        c10 = -self.d2gdr2 * r**2  # first part of c1 vector

        if self.scalarrel:
            assert self.N == self.rgd.N
            self.r2dvdr = get_r2dvdr(self.rgd, vr)
        else:
            self.r2dvdr = None

        # solve for each quantum state separately
        for j, (n, l, e, u) in enumerate(zip(self.n_j, self.l_j,
                                             self.e_j, self.u_j)):
            nodes = n - l - 1  # analytically expected number of nodes
            delta = -0.2 * e
            nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                          self.scalarrel)
            # adjust eigenenergy until u has the correct number of nodes
            while nn != nodes:
                diff = np.sign(nn - nodes)
                while diff == np.sign(nn - nodes):
                    e -= diff * delta
                    nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                                  self.scalarrel)
                delta /= 2

            # adjust eigenenergy until u is smooth at the turning point
            de = 1.0
            while abs(de) > 1e-9:
                norm = np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr)
                u *= 1.0 / sqrt(norm)
                de = 0.5 * A / norm
                x = abs(de / e)
                if x > 0.1:
                    de *= 0.1 / x
                e -= de
                assert e < 0.0
                nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                              self.scalarrel)
            self.e_j[j] = e
            u *= 1.0 / sqrt(np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr))

    def __del__(self):
        self.fd.close()


if __name__ == '__main__':
    a = AEKS('Cu', scalarrel=True)
    a.run()
