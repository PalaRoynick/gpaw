from gpaw import GPAW
import pytest
from gpaw import LCAO
import numpy as np
from ase import Atoms


def h2o_maker(vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):
        d = 0.9575
        H2O = Atoms('OH2',
                    positions=[(0, 0, 0),
                               (d + eps, 0, 0),
                               (d * np.cos(t), d * np.sin(t), 0)],
                    **kwargs)
        H2O.center(vacuum=vacuum)
        return H2O


def test_h2o(gpaw_new):

        atm = h2o_maker(vacuum=5.0)

        if gpaw_new:
                atm.calc = GPAW(
                    mode=LCAO(),
                    basis='dzp',
                    occupations={'name': 'fixed-uniform'},
                    eigensolver={'name': 'etdm'},
        #                         'representation': 'full',
        #                         'matrix_exp': 'egdecomp',
        #                         'searchdir_algo': 'l-bfgs'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off',
                )
        else:
                atm.calc = GPAW(
                    mode=LCAO(),
                    basis='dzp',
                    occupations={'name': 'fixed-uniform'},
                    eigensolver={'name': 'etdm',
                                 'representation': 'full',
                                 'matrix_exp': 'egdecomp',
                                 'searchdir_algo': 'l-bfgs',
                                 'linesearch_algo': {'name': 'max-step', 'max_step': 0.25}},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off',
                )

        atm.get_potential_energy()
