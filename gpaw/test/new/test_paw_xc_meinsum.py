import numpy as np
import pytest

from gpaw.new.paw_xc_meinsum import calculate_paw_corrections
from gpaw.setup import create_setup
from gpaw.xc import XC


@pytest.mark.gpu
def test_collective_float32_gpu_pbe_matches_gpaw():
    cp = pytest.importorskip('cupy')
    xc = XC('PBE')
    setup = create_setup('Mo', xc)
    setups = [setup, setup]
    xcc = setup.xc_correction
    D_asp = {}
    for a, occupation in enumerate([0.9, 1.1]):
        D_sp = np.zeros((1, xcc.nii))
        D_sp[0, 0] = occupation
        D_asp[a] = D_sp

    e_xc_a, dH_asp = calculate_paw_corrections(setups, D_asp, cp)

    for a, D_sp in D_asp.items():
        expected = np.zeros_like(D_sp)
        energy = xc.calculate_paw_correction(setup, D_sp, expected)
        np.testing.assert_allclose(e_xc_a[a], energy, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            dH_asp[a], expected, rtol=3e-4, atol=3e-5)
