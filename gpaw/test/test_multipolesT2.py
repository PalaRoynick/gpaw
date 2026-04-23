# test_h2o_multipoles_and_polar_tensors.py

import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms
from gpaw import GPAW
from gpaw.eigensolvers import RMMDIIS
from gpaw.multipoles import MomTensor, PolarTensor


# --- Common geometry ---------------------------------------------------------

H2O_SYMBOLS = ['O', 'H', 'H']
H2O_POSITIONS = [
    [0.00000000,  0.00000000,  0.06664447],
    [0.76804750,  0.00000000, -0.52889132],
    [-0.76804750, 0.00000000, -0.52889132],
]
PBC = [False, False, False]


# --- Helper builders ---------------------------------------------------------

def build_h2o_atoms():
    mol = Atoms(
        symbols=H2O_SYMBOLS,
        positions=H2O_POSITIONS,
        pbc=PBC,
    )
    mol.center(vacuum=3.5)
    return mol


def build_water_moments():
    """H2O multipole moments regression (coarse grid, loose conv.)."""
    gpaw_args = dict(
        mode = "fd",
        convergence={'eigenstates': 1e-1, 'density': 1e-1},
        h=0.25,
        xc="LDA",
        eigensolver=RMMDIIS(niter=5),
    )

    molecule = build_h2o_atoms()
    calc = GPAW(**gpaw_args)
    molecule.calc = calc

    mom = MomTensor(molecule)
    # (Q0, dipole, quadrupole, octupole, hexadecapole)
    return mom.get_multipole_moments()


def build_polar_tensors():
    """H2O polar tensors regression (slightly finer grid)."""
    gpaw_args = dict(
        mode = "fd",
        convergence={'eigenstates': 1e-1, 'density': 1e-1},
        h=0.25,
        xc='LDA',
        eigensolver=RMMDIIS(niter=5),
    )

    molecule = build_h2o_atoms()
    calc = GPAW(**gpaw_args)
    molecule.calc = calc

    pol = PolarTensor(molecule)
    dd = pol.calc_dd_pol()
    dq = pol.calc_dq_pol()
    qq = pol.calc_qq_pol()
    return dd, dq, qq


# --- Tests: multipole regression ---------------------------------------------

def test_h2o_multipoles_match_reference():
    monopole, dipole, quad, octu, hexa = build_water_moments()

    ref_monopole = np.float64(-15.50151075773448)

    ref_dipole = np.array([-1.16492027e-15, -7.65626707e-16,  7.50255009e-01])

    ref_quad = np.array([[-1.94145179e+00, -6.99132217e-16, -3.16952455e-15],
    [ 0.00000000e+00,  1.81825232e+00, -1.72764724e-16],
    [ 0.00000000e+00,  0.00000000e+00,  1.23199469e-01]])

    ref_octu = np.array([[[ 9.25181174e-15,  4.86866373e-16,  3.35161513e+00],
    [ 0.00000000e+00, -1.03193776e-14,  1.08874847e-15],
    [ 0.00000000e+00,  0.00000000e+00,  9.49498294e-17]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00, -1.29159733e-15, -1.39081634e+00],
    [ 0.00000000e+00,  0.00000000e+00,  1.78062584e-15]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00, -1.96079879e+00]]])

    ref_hexa = np.array([[[[ 7.79082964e-01, -1.44616649e-15, -1.10588899e-14],
    [ 0.00000000e+00,  3.62955561e+00,  8.69817793e-16],
    [ 0.00000000e+00,  0.00000000e+00, -4.40863857e+00]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  1.48197378e-15,  5.37108641e-15],
    [ 0.00000000e+00,  0.00000000e+00, -6.29495442e-16]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  1.02753424e-14]]],


    [[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00, -4.57706638e+00, -4.19494650e-16],
    [ 0.00000000e+00,  0.00000000e+00,  9.47510765e-01]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00, -3.72044421e-16]]],


    [[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  3.46112781e+00]]]])  



    # tolerances can be loosened if GPAW versions / PAW data change
    assert np.isclose(monopole, ref_monopole, rtol=1e-3, atol=1e-3)
    assert_allclose(dipole, ref_dipole, rtol=1e-3, atol=1e-3)
    assert_allclose(quad, ref_quad, rtol=1e-3, atol=1e-3)
    assert_allclose(octu, ref_octu, rtol=1e-3, atol=1e-3)
    assert_allclose(hexa, ref_hexa, rtol=1e-3, atol=1e-3)


dd, dq, qq = build_polar_tensors()

# --- Tests: polar tensor regression ------------------------------------------

def test_h2o_polar_tensors_match_reference():
    dd, dq, qq = build_polar_tensors()

    dd_ref = np.array([[10.8208, -0.0000, -0.0000],
                        [ 0.0000, 10.3436, -0.0000],
                        [ 0.1049,  0.0483, 10.8592]])

    dq_ref = np.array([[[ 0.0000, -0.0000, -7.3308],
    [-0.0000,  0.0000, -0.0000],
    [-7.3308, -0.0000, -0.0000]],

    [[ 0.0000, -0.0000, -0.0000],
    [-0.0000,  0.0000, -2.7941],
    [-0.0000, -2.7941, -0.0000]],

    [[-1.0246, -0.0000, -0.0000],
    [-0.0000,  4.0665, -0.0000],
    [-0.0000, -0.0000, -3.0419]]])

    qq_ref = np.array([[[[16.9221,  0.0250,  0.0908],
    [ 0.0250, -8.8468,  0.3473],
    [ 0.0908,  0.3473, -8.0753]],

    [[ 0.0250, 13.2810,  0.0000],
    [13.2810,  0.0319,  0.0000],
    [ 0.0000,  0.0000, -0.0569]],

    [[ 0.0908,  0.0000, 15.5516],
    [ 0.0000, -0.1113,  0.0000],
    [15.5516,  0.0000,  0.0204]]],


    [[[ 0.0250, 13.2810,  0.0000],
    [13.2810,  0.0319,  0.0000],
    [ 0.0000,  0.0000, -0.0569]],

    [[-8.8468,  0.0319, -0.1113],
    [ 0.0319, 17.5703, -0.3110],
    [-0.1113, -0.3110, -8.6227]],

    [[ 0.3473,  0.0000,  0.0000],
    [ 0.0000, -0.3110, 13.4284],
    [ 0.0000, 13.4284, -0.0363]]],


    [[[ 0.0908,  0.0000, 15.5516],
    [ 0.0000, -0.1113,  0.0000],
    [15.5516,  0.0000,  0.0204]],

    [[ 0.3473,  0.0000,  0.0000],
    [ 0.0000, -0.3110, 13.4284],
    [ 0.0000, 13.4284, -0.0363]],

    [[-8.0753, -0.0569,  0.0204],
    [-0.0569, -8.6227, -0.0363],
    [ 0.0204, -0.0363, 16.6980]]]])
    
    assert_allclose(dd, dd_ref, rtol=1e-4, atol=1e-4)
    assert_allclose(dq, dq_ref, rtol=1e-4, atol=1e-4)
    assert_allclose(qq, qq_ref, rtol=1e-4, atol=1e-4)