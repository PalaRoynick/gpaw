import numpy as np
# import pytest

from gpaw.core import PWDesc
from gpaw.utilities.spread import pseudo_spread_integral, spread_and_center


def test_spread():
    a = 6.5
    pw = PWDesc(cell=[a, a, 11.0, 90, 90, 90], ecut=25)

    # Gussian:
    alpha = 4.0
    s = (0, 3.0, lambda r: np.exp(-alpha * r**2))
    gauss_integral = np.pi / 2 / alpha**1.5
    basis = pw.atom_centered_functions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]])
    coefs = basis.layout.empty(1)
    if 0 in coefs:
        c = (np.sqrt(2) * np.sqrt(np.pi) / (16 * alpha**(3 / 2)))**-0.5
        print(coefs[0])
        coefs[0] = c
    psi_nG = pw.zeros(1)
    basis.add_to(psi_nG, coefs)
    print(psi_nG.norm2())
    print(gauss_integral)
    if 1:
        from sympy import exp, integrate, oo, var, sqrt, pi
        a0, r = var('a, r')
        c = 16 * sqrt(a0**3) / sqrt(2) / sqrt(pi)
        print(integrate(c * exp(-2 * a0 * r**2) * r**2, (r, 0, oo)))
        print(integrate(c * exp(-2 * a0 * r**2) * r**4, (r, 0, oo)))

    # assert f1.integrate() == pytest.approx(gauss_integral)
    psi_nR = psi_nG.ifft(grid=pw.minimal_uniform_grid())
    print(psi_nR.norm2())
    z, G = pseudo_spread_integral(psi_nR)
    print(z, G)
    s, c_v = spread_and_center(z[0], G)
    assert s == (0.75 / alpha)**0.5


if __name__ == '__main__':
    test_spread()
