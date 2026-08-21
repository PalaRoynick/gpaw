import functools
from math import sqrt
from pathlib import Path

import numpy as np
from ase import Atom, Atoms

from gpaw import FD, GPAW, LCAO, PW, Davidson, FermiDirac, Mixer
from ase import Atoms
from gpaw.directmin.etdm_lcao import LCAOETDM

import io

import numpy as np
import numpy.testing as npt
import pytest

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.old.logger import GPAWLogger
from gpaw.old.wavefunctions.base import eigenvalue_string
from gpaw.test.sic._utils import MockWorld, extract_lagrange_section, mk_arr_from_str


def h2o_maker(vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):
    d = 0.9575
    H2O = Atoms(
        "OH2",
        positions=[(0, 0, 0), (d + eps, 0, 0), (d * np.cos(t), d * np.sin(t), 0)],
        **kwargs,
    )
    H2O.center(vacuum=vacuum)
    return H2O


def h2o_lcaosic():
    calc = GPAW(
        mode=LCAO(force_complex_dtype=True),
        h=0.22,
        occupations={"name": "fixed-uniform"},
        eigensolver="etdm",
        convergence={"eigenstates": 1e-4},
        mixer={"backend": "no-mixing"},
        nbands="nao",
        symmetry="off",
    )
    return calc


@pytest.mark.sic
def test_lcaosic(in_tmp_dir, gpw_files):
    """
    Test Perdew-Zunger Self-Interaction
    Correction  in LCAO mode using ETDM
    :param in_tmp_dir:
    :return:
    """

    H2O = h2o_maker(vacuum=4.0)
    calc = h2o_lcaosic()
    H2O.calc = calc
    e = H2O.get_potential_energy()
    f = H2O.get_forces()

    # Numerical derivatives need full position-dependent state (nct_G,
    # vbar, etc.) which read() does not provide for density/hamiltonian.

    assert e == pytest.approx(-12.16352, abs=1e-3)
