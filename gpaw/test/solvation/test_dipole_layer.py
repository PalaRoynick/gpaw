from math import pi

import numpy as np
import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.fd_operators import Gradient
from gpaw.solvation.dipole_layer import GaussianDerivativeDipole
from gpaw.solvation.poisson import SolvationPoissonSolver
from gpaw.solvation.dielectric import Dielectric


@pytest.mark.parametrize('mode', ['fd', 'pw'])
def test_dipole_layer(mode):
    """Asymmetric slab charge in a z-dependent dielectric: with the dipole
    layer, the potential step between the vacuum regions below and above
    the charge must match the 1D result -4 pi int Q(z) / eps(z) dz."""
    grid = UGDesc(cell=np.diag([4.0, 4.0, 26.0]), size=(16, 16, 104),
                  pbc=(True, True, False), zerobc=(False, False, mode == 'fd'))
    z_R = grid._gd.get_grid_point_coordinates()[2]
    rhot_r = grid.from_data(0.05 * (z_R - 11.0) * np.exp(-(z_R - 11.0)**2))
    eps_R = 80.0 - 79.0 * np.exp(-((z_R - 12.0) / 3.0)**4)

    if mode == 'fd':
        dielectric = Dielectric(epsinf=80.0)
        dielectric.set_grid_descriptor(grid._gd)
        dielectric.allocate()
        dielectric.eps_gradeps[0][:] = eps_R
        for v in range(3):
            Gradient(grid._gd, v, 1.0, 3).apply(eps_R - 80.0,
                                                dielectric.eps_gradeps[1 + v])
        ps = SolvationPoissonSolver(None, grid, dielectric, nn=3)
        vHt_r = grid.zeros()
        rhot = rhot_r
    else:
        dielectric = type('MockDielectric', (), {'eps_gradeps': [eps_R]})
        pw = PWDesc(ecut=grid.ekin_max(), cell=grid.cell)
        ps = SolvationPoissonSolver(pw, grid, dielectric, rtol=1e-8,
                                    maxiter=1000)
        vHt_r = pw.zeros()
        rhot = rhot_r.fft(pw=pw)

    ps = GaussianDerivativeDipole(zero_vacuum=True).build(ps)
    ps.solve(vHt_r, rhot)
    if mode == 'pw':
        vHt_r = vHt_r.ifft(grid=grid)
    v_z = vHt_r.data.mean(axis=(0, 1))

    # 1D reference, from z = 7 (above the layer) to z = 22 bohr (vacuum):
    z_z, dz = z_R[0, 0], z_R[0, 0, 1] - z_R[0, 0, 0]
    rho_z, eps_z = rhot_r.data[0, 0], eps_R[0, 0]
    Q_z = (np.cumsum(rho_z) - rho_z / 2) * dz
    i, j = np.searchsorted(z_z, [7.0, 22.0])
    step = -4 * pi * (Q_z / eps_z)[i:j].sum() * dz
    # (Tolerance is the solvers' discretization error at h = 0.25 bohr.)
    assert v_z[j] - v_z[i] == pytest.approx(step, rel=1e-2)
    assert v_z[j] == pytest.approx(0.0, abs=1e-5)  # zero_vacuum
