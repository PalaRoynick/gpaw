from math import pi

import numpy as np
import pytest

from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.old.grid_descriptor import GridDescriptor
from gpaw.xc import XC


@pytest.mark.ci
@pytest.mark.parametrize('xc_name', ['LDA', 'PBE'])
@pytest.mark.parametrize('nspins', [1, 2])
@pytest.mark.parametrize('radial_i', [0, 21, -1])
def test_xc_XC2_radial(xc_name, nspins, radial_i):
    rgd = EquidistantRadialGridDescriptor(0.01, 100)

    xc = XC(xc_name)
    n = rgd.zeros(nspins)
    v = rgd.zeros(nspins)
    n[:] = np.exp(-rgd.r_g**2)
    n[-1] *= 2
    E = xc.calculate_spherical(rgd, n, v)
    i = radial_i
    x = v[-1, i] * np.clip(rgd.dv_g[i], 1e-40, None)
    n[-1, i] += 1e-6
    Ep = xc.calculate_spherical(rgd, n, v)
    n[-1, i] -= 2e-6
    Em = xc.calculate_spherical(rgd, n, v)
    x2 = (Ep - Em) / 2e-6
    print(xc_name, nspins, E, x, x2, x - x2)
    assert x == pytest.approx(x2, abs=1e-9)
    n[-1, i] += 1e-6
    if nspins == 1:
        ns = rgd.empty(2)
        ns[:] = n / 2
        Es = xc.calculate_spherical(rgd, ns, 0 * ns)
        assert E == pytest.approx(Es, abs=1e-13)


@pytest.mark.ci
@pytest.mark.parametrize('xc_name', ['LDA', 'PBE'])
@pytest.mark.parametrize('nspins', [1, 2])
def test_xc_XC2_grid(xc_name, nspins):
    N = 20
    a = 1.0
    gd = GridDescriptor((N, N, N), (a, a, a))

    xc = XC(xc_name)
    n = gd.empty(nspins)
    n.fill(0.03)
    z = np.arange(gd.beg_c[2], gd.end_c[2]) * a / N
    n[:] += 0.01 * np.sin(2 * pi * z / a)
    if nspins == 2:
        n[1] += 0.01 * np.cos(2 * pi * z / a)
    n /= nspins

    v = 0.0 * n
    E = xc.calculate(gd, n, v)

    here = (gd.beg_c[0] <= 1 < gd.end_c[0] and
            gd.beg_c[1] <= 2 < gd.end_c[1] and
            gd.beg_c[2] <= 3 < gd.end_c[2])
    if here:
        x = v[-1, 1, 2, 3] * gd.dv
        n[-1, 1, 2, 3] += 0.000001
    Ep = xc.calculate(gd, n, v)
    if here:
        n[-1, 1, 2, 3] -= 0.000002
    Em = xc.calculate(gd, n, v)
    x2 = (Ep - Em) / 0.000002
    if here:
        print(xc_name, nspins, E, x, x2, x - x2)
        assert x == pytest.approx(x2, abs=1e-11)
        n[-1, 1, 2, 3] += 0.000001

    if nspins == 1:
        ns = gd.empty(2)
        ns[:] = n / 2
        Es = xc.calculate(gd, ns, 0 * ns)
        assert E == pytest.approx(Es, abs=1e-13)
