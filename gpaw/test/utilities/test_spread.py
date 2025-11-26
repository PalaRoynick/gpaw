import numpy as np
# import pytest

from gpaw.core import PWDesc
from gpaw.utilities.spread import pseudo_spread_integral


def test_spread():
    a = 6.5
    pw = PWDesc(cell=[a, a, 11.0, 90, 90, 120], ecut=25)

    # Gussian:
    alpha = 4.0
    s = (0, 3.0, lambda r: np.exp(-alpha * r**2))
    gauss_integral = np.pi / 2 / alpha**1.5
    basis = pw.atom_centered_functions(
        [[s]],
        positions=[[0.4, 0.5, 0.6]])
    coefs = basis.layout.empty(1)
    if 0 in coefs:
        print(coefs[0])
        coefs[0] = 1
    psi_nG = pw.zeros(1)
    basis.add_to(psi_nG, coefs)
    print(psi_nG.integrate())
    print(psi_nG.integrate(psi_nG))
    print(psi_nG.norm2())
    print(gauss_integral)
    if 0:
        from sympy import exp, integrate, oo, var
        a0, r = var('a, r')
        integrate(exp(-a0 * r**2) * r**2, (r, 0, oo))

    # assert f1.integrate() == pytest.approx(gauss_integral)
    psi_nR = psi_nG.ifft(grid=pw.minimal_uniform_grid())
    print(psi_nR.integrate())
    print(psi_nR.integrate(psi_nR))
    print(psi_nR.norm2())
    z, G = pseudo_spread_integral(psi_nR)
    print(z, G)


if __name__ == '__main__':
    test_spread()
