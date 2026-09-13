import numpy as np
import pytest

from gpaw.new.paw_xc_meinsum import calculate_paw_corrections
from gpaw.setup import create_setup
from gpaw.xc import XC


@pytest.mark.gpu
@pytest.mark.parametrize('xc_name, polarized', [
    ('LDA', False),
    ('LDA', True),
    ('PBE', False),
    ('PBE', True),
])
def test_collective_float32_gpu_matches_gpaw(xc_name, polarized):
    cp = pytest.importorskip('cupy')
    xc = XC(xc_name)
    setup = create_setup('Mo', xc)
    setups = [setup, setup]
    xcc = setup.xc_correction
    D_asp = {}
    for a, occupation in enumerate([0.9, 1.1]):
        nspins = 2 if polarized else 1
        D_sp = np.zeros((nspins, xcc.nii))
        if polarized:
            D_sp[0, 0] = 0.65 * occupation
            D_sp[1, 0] = 0.35 * occupation
        else:
            D_sp[0, 0] = occupation
        D_asp[a] = D_sp

    e_xc_a, dH_asp = calculate_paw_corrections(
        setups, D_asp, cp, xc_name)

    for a, D_sp in D_asp.items():
        expected = np.zeros_like(D_sp)
        energy = xc.calculate_paw_correction(setup, D_sp, expected)
        np.testing.assert_allclose(
            e_xc_a[a], energy, rtol=8e-4, atol=8e-4)
        np.testing.assert_allclose(
            dH_asp[a], expected, rtol=5e-3, atol=8e-4)


@pytest.mark.gpu
def test_mixed_mo_s_layout_groups_match_gpaw():
    """Different setup dimensions execute as concurrent shape batches."""
    cp = pytest.importorskip('cupy')
    xc = XC('PBE')
    setups = [create_setup(symbol, xc) for symbol in ['Mo', 'S', 'Mo', 'S']]
    D_asp = {}
    for a, setup in enumerate(setups):
        D_sp = np.zeros((1, setup.xc_correction.nii))
        D_sp[0, 0] = 0.85 + 0.1 * a
        D_asp[a] = D_sp

    e_xc_a, dH_asp = calculate_paw_corrections(
        setups, D_asp, cp, 'PBE')

    for a, (setup, D_sp) in enumerate(zip(
            setups, D_asp.values(), strict=True)):
        expected = np.zeros_like(D_sp)
        energy = xc.calculate_paw_correction(setup, D_sp, expected)
        np.testing.assert_allclose(
            e_xc_a[a], energy, rtol=8e-4, atol=8e-4)
        np.testing.assert_allclose(
            dH_asp[a], expected, rtol=5e-3, atol=8e-4)
