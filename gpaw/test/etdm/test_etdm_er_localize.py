import numpy as np
import pytest
from ase.build import molecule

from gpaw import GPAW
from gpaw.new.etdm.edmiston_ruedenberg.obj_function import \
    EdmistonRuedenbergUpdateRef as ER
from gpaw.new.etdm.etdm import ETDM
from gpaw.new.etdm.tools import random_a


pytestmark = pytest.mark.usefixtures('module_tmp_path')


@pytest.fixture(scope='module')
def ibzwfs_from_new_gpaw():
    """Create IBZWaveFunctions using new GPAW."""

    atoms = molecule('C6H6')
    atoms.center(vacuum=3.0)

    calc = GPAW(
        mode='pw',
        xc='LDA',
        nbands=15,
        spinpol=False,
        kpts=(1, 1, 1),
        txt='benzene.txt')
    atoms.calc = calc
    atoms.get_potential_energy()

    ibzwfs = calc.dft.ibzwfs

    return ibzwfs


def test_er_localize_reproducibility(ibzwfs_from_new_gpaw):
    """Test that ER localization is reproducible."""
    ibzwfs = ibzwfs_from_new_gpaw
    seed = 42
    _ = er_localize(ibzwfs, gtol=1e-3, seed=seed)
    assert _.energy == pytest.approx(-3.641042186008981, abs=5.e-3)


def er_localize(ibzwfs,
                states='all',
                loct='pseudo-paw',
                gtol=1e-6,
                niter=333,
                seed=None):
    objfunc = ER(ibzwfs, loct, states, representation="full")
    a_init = []
    for a in objfunc.a_vec_u:
        skmat = random_a((a._ndim, a._ndim), ibzwfs.dtype, seed=seed) * 0.01
        skmat -= skmat.T.conj()
        a_init.append(skmat[a.ind_up])
    a_init = np.asarray(a_init)
    ibzwfs.comm.broadcast(a_init, 0)

    etdm = ETDM(objfunc, a_init, niter, gtol, True)
    etdm.optimize()

    return etdm
