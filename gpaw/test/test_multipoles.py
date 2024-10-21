from gpaw.multipoles import PolarTensor, MomTensor
from ase.io import read
from gpaw import GPAW
from gpaw.eigensolvers import RMMDIIS
from ase.parallel import parprint
import numpy as np
from gpaw.test import multipoles
import os


def test_moment_tensors():

    dirname = os.path.dirname(multipoles.__file__)
    dp_ref = np.load(os.path.join(dirname,"dp.npy"))
    qp_ref = np.load(os.path.join(dirname,"qp.npy"))
    op_ref = np.load(os.path.join(dirname,"op.npy"))
    hp_ref = np.load(os.path.join(dirname,"hp.npy"))
    
    exmaple_src_path = os.path.join(dirname,"H2O.xyz")

    gpaw_args = dict(convergence={'eigenstates': 1e-01,
                                'density': 1e-01},
                    h=0.5, xc="LDA",
                    eigensolver=RMMDIIS(niter=5))                                                      


    molecule = read(exmaple_src_path)
    molecule.center(vacuum=7.0)
    calc = GPAW(**gpaw_args)
    molecule.set_calculator(calc)

    mom = MomTensor(molecule).get_multipole_moments()

    assert np.array_equal(mom[0],dp_ref), "dipoles are not consistent"
    assert np.array_equal(mom[1],qp_ref), "qpoles are not consistent"
    assert np.array_equal(mom[2],op_ref), "opoles are not consistent"
    assert np.array_equal(mom[3],hp_ref), "hpoles are not consistent"

def test_polarizability_tensors():

    dirname = os.path.dirname(multipoles.__file__)

    dd_ref = np.load(os.path.join(dirname,"dd.npy"))
    dq_ref = np.load(os.path.join(dirname,"dq.npy"))
    qq_ref = np.load(os.path.join(dirname,"qq.npy"))
    exmaple_src_path = os.path.join(dirname,"H2O.xyz")

    gpaw_args = dict(convergence={'eigenstates': 1e-01,
                                'density': 1e-01},
                    h=0.3, xc="LDA",
                    eigensolver=RMMDIIS(niter=5))                                                   


    molecule = read(exmaple_src_path)
    molecule.center(vacuum=7.0)
    calc = GPAW(**gpaw_args)
    molecule.set_calculator(calc)
    Pol = PolarTensor(molecule)
    dd = Pol.calc_dd_pol()
    dq = Pol.calc_dq_pol()
    qq = Pol.calc_qq_pol()


    assert np.array_equal(dd,dd_ref), "dipole-dipole polarizability tensors are not consistent"
    assert np.array_equal(dq,dq_ref), "dipole-quadrupole polarizability tensors are not consistent"
    assert np.array_equal(qq,qq_ref), "quadrupole-quadrupole polarizability tensors are not consistent"
