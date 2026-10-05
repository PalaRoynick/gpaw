import pytest

from scipy.special import factorial
import numpy as np

from gpaw.atom.radialgd import AERadialGridDescriptor


def test_derivatives():
    N = 900
    a = 1 / 2250
    b = 1 / N

    rgd = AERadialGridDescriptor(a, b, N)
    r_g = rgd.r_g

    Z_eff = 10
    A = np.sqrt((2 * Z_eff / 4)**3 * 1 / (2 * 4 * factorial(6))) * Z_eff**2
    Exp = np.exp(- Z_eff * r_g / 4)

    f_g = A * Exp * (3 / 2 * r_g**2
                     - 1 / 8 * Z_eff * r_g**3)

    # Analytic derivatives
    dfdr_g = A * Exp * (3 * r_g - 3 / 4 * Z_eff * r_g**2
                        + 1 / 32 * Z_eff**2 * r_g**3)
    d2fdr2_g = A * Exp * (3 - 9 / 4 * Z_eff * r_g
                          + 9 / 32 * Z_eff**2 * r_g**2
                          - 1 / 128 * Z_eff**3 * r_g**3)
    d3fdr3_g = A * Exp * (- 3 * Z_eff + 9 / 8 * Z_eff**2 * r_g
                          - 3 / 32 * Z_eff**3 * r_g**2
                          + 1 / 512 * Z_eff**4 * r_g**3)

    # Numerical derivatives
    num_dfdr_g, num_d2fdr2_g, num_d3fdr3_g = rgd.calculate_derivatives(f_g)

    # Assert small absolute difference
    assert num_dfdr_g == pytest.approx(dfdr_g, abs=1.e-6)
    assert num_d2fdr2_g == pytest.approx(d2fdr2_g, abs=1.e-6)
    assert num_d3fdr3_g == pytest.approx(d3fdr3_g, abs=1.e-5)
