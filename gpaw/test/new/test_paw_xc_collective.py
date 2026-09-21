import numpy as np

from gpaw.core import UGDesc
from gpaw.new.xc import create_functional
from gpaw.setup import create_setup
from gpaw.xc import XC


def test_collective_paw_xc_matches_gpaw_atom_loop():
    xc = XC('PBE')
    setups = [create_setup(symbol, xc) for symbol in ['N', 'O']]
    functional = create_functional(
        xc, UGDesc(cell=[1, 1, 1], size=[9, 9, 9]))
    D_asp = {}
    for a, setup in enumerate(setups):
        D_sp = np.zeros((1, setup.xc_correction.nii))
        D_sp[0, 0] = 0.8 + 0.2 * a
        D_asp[a] = D_sp

    e_xc_a, dH_asp = functional.calculate_paw_corrections(setups, D_asp)

    for a, D_sp in D_asp.items():
        expected = np.zeros_like(D_sp)
        energy = functional.calculate_paw_correction(
            setups[a], D_sp, expected, a=a)
        np.testing.assert_allclose(e_xc_a[a], energy, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            dH_asp[a], expected, rtol=0.0, atol=0.0)


def test_collective_paw_xc_accepts_empty_rank():
    functional = create_functional(
        XC('LDA'), UGDesc(cell=[1, 1, 1], size=[9, 9, 9]))
    assert functional.calculate_paw_corrections([], {}) == ({}, {})
