from gpaw import GPAW
from gpaw import LCAO
import numpy as np
import pytest
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


atm = h2o_maker(vacuum=5.0)
atm.calc = GPAW(
            mode=LCAO(),
            basis='dzp',
            occupations={'name': 'fixed-uniform'},
            eigensolver='etdm',
            mixer={'backend': 'no-mixing'},
            nbands='nao',
            symmetry='off',
        )

atm.get_potential_energy()