import pytest

from gpaw import GPAW
from ase import Atoms
import numpy as np
from gpaw import LCAO


def h2o_maker(vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):

    d = 0.9575
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d + eps, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)], **kwargs)
    H2O.center(vacuum=vacuum)
    return H2O


@pytest.mark.do
@pytest.mark.parametrize("max_step", [0.25, 0.05, 0.005])
def test_etdm_iters_old_vs_new(in_tmp_dir, gpw_files, max_step):
    """
    Test that old and new take roughly the same amount of iterations
    to converge
    """

    atm = h2o_maker(vacuum=5.0)

    calc_old = GPAW(
        legacy_gpaw=True,
        mode=LCAO(),
        basis='dzp',
        occupations={'name': 'fixed-uniform'},
        eigensolver={'name': 'etdm',
                     'matrix_exp': 'egdecomp',
                     'searchdir_algo': 'l-bfgs-p',
                     'linesearch_algo': {'name': 'max-step',
                                         'max_step': max_step}},
        convergence={'density': 1e-5},
        mixer={'backend': 'no-mixing'},
        spinpol=True,
        nbands='nao',
        symmetry='off',
        maxiter=100,
        txt='-')

    atm.calc = calc_old
    atm.get_potential_energy()
    niter_old = calc_old.get_number_of_iterations()

    calc_new = GPAW(
        legacy_gpaw=False,
        mode=LCAO(),
        basis='dzp',
        occupations={'name': 'fixed-uniform'},
        eigensolver={'name': 'etdm',
                     'matrix_exp': 'egdecomp',
                     'searchdir_algo': 'l-bfgs-p',
                     'linesearch_algo': {'name': 'max-step',
                                         'max_step': max_step}},
        mixer={'backend': 'no-mixing'},
        convergence={'density': 1e-5},
        spinpol=True,
        nbands='nao',
        symmetry='off',
        maxiter=100,
        txt='-')

    atm.calc = calc_new
    atm.get_potential_energy()
    niter_new = calc_new.get_number_of_iterations()

    assert niter_new == pytest.approx(niter_old, abs=3)
