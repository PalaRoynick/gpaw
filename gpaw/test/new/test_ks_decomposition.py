"""Test decomposition of KS eigenvalues into <T>, <v_H>, <v_xc>, <v_bar>."""
import numpy as np
import pytest
from ase.build import bulk

from gpaw import GPAW
from gpaw.new.pw.ks_decomposition import (KSDecompositionCalculator,
                                          ks_decomposition)
from gpaw.xc.tools import vxc


def check(calc, d):
    """Check closure and compare xc-term with gpaw.xc.tools.vxc()."""
    nspins = calc.get_number_of_spins()
    nkpts = len(calc.get_ibz_k_points())
    assert d.eps_skn.shape == (nspins, nkpts, d.n2 - d.n1)

    # Eigenvalues:
    for s in range(nspins):
        for k in range(nkpts):
            eps_n = calc.get_eigenvalues(kpt=k, spin=s)[d.n1:d.n2]
            assert d.eps_skn[s, k] == pytest.approx(eps_n, abs=1e-10)

    # eps = <T> + <v_H> + <v_xc> + <v_bar> (all-electron terms):
    assert abs(d.residual_skn).max() < 1e-8

    # Zero-potential terms should (almost) cancel:
    assert abs(d.zero.total).max() < 0.2
    assert abs(d.zero.pseudo).max() > 0.2

    # Kinetic energies are positive:
    assert (d.kinetic.pseudo > 0).all()
    assert (d.kinetic.total > 0).all()

    # Compare with existing implementation of <v_xc>:
    vxc_skn = vxc(calc.gs_adapter(), n1=d.n1, n2=d.n2)
    assert d.xc.total == pytest.approx(vxc_skn, abs=1e-5)

    # Flat dict:
    dct = d.todict()
    assert dct['kinetic_total'] == pytest.approx(d.kinetic.total)
    assert (dct['bands'] == [d.n1, d.n2]).all()


@pytest.mark.parametrize('name', ['si_pw', 'h_pw'])
def test_ks_decomposition(gpw_files, name):
    calc = GPAW(gpw_files[name])
    d = ks_decomposition(calc)
    print(d)
    check(calc, d)

    # Band and k-point subsets:
    kdc = KSDecompositionCalculator(calc.dft)
    nbands = calc.get_number_of_bands()
    n1 = min(1, nbands - 1)
    d2 = kdc.calculate(n1=n1, n2=nbands, kpt_indices=[0])
    assert d2.eps_skn.shape[1:] == (1, nbands - n1)
    assert d2.kinetic.total[:, 0] == pytest.approx(
        d.kinetic.total[:, 0, n1:], abs=1e-10)
    assert d2.hartree.pseudo[:, 0] == pytest.approx(
        d.hartree.pseudo[:, 0, n1:], abs=1e-10)

    # From file name:
    d3 = ks_decomposition(gpw_files[name], n1=0, n2=-n1)
    assert d3.xc.paw == pytest.approx(d.xc.paw[:, :, :nbands - n1],
                                      abs=1e-10)


def test_ks_decomposition_pbe(in_tmp_dir):
    """GGA, k-points and (default) parallelization."""
    atoms = bulk('Si')
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                      xc='PBE',
                      kpts=(2, 2, 2),
                      nbands=8,
                      convergence={'bands': 6},
                      txt='si_pbe.txt')
    atoms.get_potential_energy()
    d = ks_decomposition(atoms.calc, n1=0, n2=6)
    check(atoms.calc, d)

    calc = atoms.calc

    # Without wave functions we can't do anything:
    calc.write('si_pbe_no_wfs.gpw')
    with pytest.raises(ValueError, match='No wave functions'):
        ks_decomposition('si_pbe_no_wfs.gpw')

    # ... but from a file with wave functions we can:
    calc.write('si_pbe.gpw', mode='all')
    d2 = ks_decomposition('si_pbe.gpw', n1=0, n2=6)
    assert d2.hartree.total == pytest.approx(d.hartree.total, abs=1e-8)
    assert d2.kinetic.paw == pytest.approx(d.kinetic.paw, abs=1e-8)
    assert np.allclose(d2.residual_skn, 0.0, atol=1e-8)
