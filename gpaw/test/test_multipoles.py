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
    mol.center(vacuum=7.0)
    return mol


def build_water_moments():
    """H2O multipole moments regression (coarse grid, loose conv.)."""
    gpaw_args = dict(
        mode = "fd",
        convergence={'eigenstates': 1e-1, 'density': 1e-1},
        h=0.5,
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
        h=0.3,
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

    ref_monopole = np.float64(135.91706143979988)
    ref_dipole = np.array([0.22976428, -0.20996236, 2.5809135])
    ref_quad = np.array([
        [14.12904773,  7.78827662, -1.07217344],
        [0.        , -6.39935584, -0.06805148],
        [0.        ,  0.        , -7.7296919 ],
    ])
    ref_octu = np.array([
        [[  7.23584227,   2.59161954,  43.97702213],
         [  0.        ,  -3.89820241,   0.18714712],
         [  0.        ,   0.        ,  -3.33763986]],
        [[  0.        ,   0.        ,   0.        ],
         [  0.        ,  -3.01115286, -11.84194785],
         [  0.        ,   0.        ,   0.41953332]],
        [[  0.        ,   0.        ,   0.        ],
         [  0.        ,   0.        ,   0.        ],
         [  0.        ,   0.        , -32.13507428]],
    ])
    ref_hexa = np.array([
        [[[-163.11281379,   51.78598043,    4.48353565],
          [  0.        ,  125.64110306,  -37.08498495],
          [  0.        ,    0.        ,   37.47171073]],
         [[  0.        ,   0.        ,   0.        ],
          [  0.        ,  -7.94072558,  -14.28076034],
          [  0.        ,   0.        ,  -43.84525485]],
         [[  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,    9.79722469]]],
        [[[  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ]],
         [[  0.        ,   0.        ,   0.        ],
          [  0.        ,-146.22697787,   15.13850845],
          [  0.        ,   0.        ,   20.58587481]],
         [[  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   21.9464765 ]]],
        [[[  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ]],
         [[  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ]],
         [[  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,   0.        ],
          [  0.        ,   0.        ,  -58.05758554]]],
    ])

    # tolerances can be loosened if GPAW versions / PAW data change
    assert np.isclose(monopole, ref_monopole, rtol=1e-3, atol=1e-3)
    assert_allclose(dipole, ref_dipole, rtol=1e-3, atol=1e-3)
    assert_allclose(quad, ref_quad, rtol=1e-3, atol=1e-3)
    assert_allclose(octu, ref_octu, rtol=1e-3, atol=1e-3)
    assert_allclose(hexa, ref_hexa, rtol=1e-3, atol=1e-3)


# --- Tests: polar tensor regression ------------------------------------------

def test_h2o_polar_tensors_match_reference():
    dd, dq, qq = build_polar_tensors()

    dd_ref = np.array([
        [10.9254, -0.0,    -0.0   ],
        [ 0.0,    11.0911,  0.0   ],
        [-0.0156,  0.0530, 10.5321],
    ])

    dq_ref = np.array([
        [[-0.0,    -0.0,    -7.4668],
         [-0.0,     0.0,     0.0   ],
         [-7.4668,  0.0,    -0.0   ]],
        [[-0.0,    -0.0,    -0.0   ],
         [-0.0,    -0.0,    -3.3350],
         [-0.0,    -3.3350,  0.0   ]],
        [[-0.7018, -0.0,    -0.0   ],
         [-0.0,     3.9856, -0.0   ],
         [-0.0,    -0.0,    -3.2839]],
    ])

    qq_ref = np.array([
        [
            [
                [ 1.82726e+01, -1.20000e-03, -6.48000e-02],
                [-1.20000e-03, -1.05145e+01,  1.74000e-01],
                [-6.48000e-02,  1.74000e-01, -7.75810e+00],
            ],
            [
                [-1.20000e-03,  1.49734e+01,  0.00000e+00],
                [ 1.49734e+01,  1.01900e-01,  0.00000e+00],
                [ 0.00000e+00,  0.00000e+00, -1.00700e-01],
            ],
            [
                [-6.48000e-02,  0.00000e+00,  1.59787e+01],
                [ 0.00000e+00,  1.64000e-02, -0.00000e+00],
                [ 1.59787e+01, -0.00000e+00,  4.83000e-02],
            ],
        ],
        [
            [
                [-1.20000e-03,  1.49734e+01,  0.00000e+00],
                [ 1.49734e+01,  1.01900e-01,  0.00000e+00],
                [ 0.00000e+00,  0.00000e+00, -1.00700e-01],
            ],
            [
                [-1.05145e+01,  1.01900e-01,  1.64000e-02],
                [ 1.01900e-01,  2.09692e+01, -5.49300e-01],
                [ 1.64000e-02, -5.49300e-01, -1.02965e+01],
            ],
            [
                [ 1.74000e-01,  0.00000e+00, -0.00000e+00],
                [ 0.00000e+00, -5.49300e-01,  1.47030e+01],
                [-0.00000e+00,  1.47030e+01,  3.75300e-01],
            ],
        ],
        [
            [
                [-6.48000e-02,  0.00000e+00,  1.59787e+01],
                [ 0.00000e+00,  1.64000e-02, -0.00000e+00],
                [ 1.59787e+01, -0.00000e+00,  4.83000e-02],
            ],
            [
                [ 1.74000e-01,  0.00000e+00, -0.00000e+00],
                [ 0.00000e+00, -5.49300e-01,  1.47030e+01],
                [-0.00000e+00,  1.47030e+01,  3.75300e-01],
            ],
            [
                [-7.75810e+00, -1.00700e-01,  4.83000e-02],
                [-1.00700e-01, -1.02965e+01,  3.75300e-01],
                [ 4.83000e-02,  3.75300e-01,  1.80546e+01],
            ],
        ],
    ])

    assert_allclose(dd, dd_ref, rtol=1e-4, atol=1e-4)
    assert_allclose(dq, dq_ref, rtol=1e-4, atol=1e-4)
    assert_allclose(qq, qq_ref, rtol=1e-4, atol=1e-4)