from types import SimpleNamespace

import numpy as np
import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.solvation.dipole_layer import GaussianDerivativeDipole
from gpaw.solvation.poisson import SolvationPoissonSolver


def f(a, z, z0, w):
    return np.exp(-((z - z0) / w)**2) / a**2 / np.pi**0.5 / w


def test_cg_poisson_solver_zeroes_the_vacuum():
    """``zero_vacuum`` puts the potential at zero above the slab.

    The companion test in ``gpaw/test/solvation/test_pw_poisson.py`` checks
    that ``dipolelayer`` flattens the vacuum; this one checks the other half,
    that ``zero_vacuum`` then pins it to zero.  The cell here is deliberately
    short, so the vacuum is thin and still feels the charge -- do not add a
    slope assertion, which needs the roomier cell used over there.
    """
    a = 1.0
    L = 10.0
    grid = UGDesc.from_cell_and_grid_spacing(cell=[a, a, L], grid_spacing=0.15)
    z = grid.xyz()[0, 0, :, 2]
    c = 0.05
    rhot = f(a, z, 4.0, 1.0) * (1 + c)  # electrons
    rhot -= f(a, z, 4.0, 0.5)  # nuclei
    rhot -= f(a, z, 6.0, 1.0) * c  # jellium
    eps = 1.0 + f(a, z, 5.0, 2.0) * 20  # dielectric
    rhot_r = grid.zeros()
    rhot_r.data[:] = rhot
    eps_r = grid.zeros()
    eps_r.data[:] = eps

    # The solver only needs `eps_gradeps[0]` from the Dielectric object.
    dielectric = SimpleNamespace(eps_gradeps=[eps_r.data])

    pw = PWDesc(ecut=grid.ekin_max(), cell=grid.cell)
    ps = GaussianDerivativeDipole(zero_vacuum=True).build(
        SolvationPoissonSolver(pw, grid, dielectric))
    vt_g = pw.zeros()
    ps.solve(vt_g, rhot_r.fft(pw=pw))
    vt_z = vt_g.ifft(grid=grid).data.mean(axis=(0, 1))

    assert vt_z[-1] == pytest.approx(0.0, abs=1e-12)
