from gpaw import GPAW
import pytest
from gpaw import LCAO
import numpy as np
from ase import Atoms
import os


def h2o_maker(vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):

    d = 0.9575
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d + eps, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)], **kwargs)
    H2O.center(vacuum=vacuum)
    return H2O


@pytest.mark.parametrize('legacy_gpaw', [False])
@pytest.mark.parametrize("max_step", [0.25, 0.05, 0.005])
@pytest.mark.parametrize("searchdir_algo", ['l-bfgs', 'l-bfgs-p', 'LSR1P'])
def test_h2o(legacy_gpaw, searchdir_algo, max_step):

    """
    Test for search direction algorithms and max step sizes
    """

    atm = h2o_maker(vacuum=5.0)

    calc = GPAW(
             legacy_gpaw=legacy_gpaw,
             mode=LCAO(),
             basis='dzp',
             occupations={'name': 'fixed-uniform'},
             eigensolver={'name': 'etdm',
                          'representation': 'full',
                          'matrix_exp': 'egdecomp',
                          'searchdir_algo': searchdir_algo,
                          'linesearch_algo': {'name': 'max-step',
                                              'max_step': max_step}},
             mixer={'backend': 'no-mixing'},
             spinpol=True,
             nbands='nao',
             symmetry='off',
             maxiter=100,
             txt=None
                )
    atm.calc = calc
    e = atm.get_potential_energy()
    f = atm.get_forces()
    target_pot_en = -13.643156256566218
    abs_force_tol, abs_en_tol = 1e-2, 1.0e-4
    f2 = np.array([[-1.11463, -1.23723, 0.0],
                   [1.35791, 0.00827, 0.0],
                   [-0.34423, 1.33207, 0.0]])
    assert e == pytest.approx(target_pot_en, abs=abs_en_tol)
    assert f == pytest.approx(f2, abs=abs_force_tol)
    atm.calc = calc
    e = atm.get_potential_energy()
    f = atm.get_forces()
    target_pot_en = -13.643156256566218
    abs_force_tol, abs_en_tol = 1e-2, 1.0e-4
    f2 = np.array([[-1.11463, -1.23723, 0.0],
                   [1.35791, 0.00827, 0.0],
                   [-0.34423, 1.33207, 0.0]])
    assert e == pytest.approx(target_pot_en, abs=abs_en_tol)
    assert f == pytest.approx(f2, abs=abs_force_tol)