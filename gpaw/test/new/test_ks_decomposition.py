"""Test decomposition of KS eigenvalues into <T>, <v_H>, <v_xc>, <v_bar>."""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.units import Ha

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.new.pw.ks_decomposition import (KSDecompositionCalculator,
                                          coulomb_correction_parts,
                                          ks_decomposition)
from gpaw.new.pw.poisson import ChargedPWPoissonSolver
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

    # Electrostatic terms: valence + core + nuclear = full
    # (pseudo and PAW parts separately):
    parts = d.hartree_valence + d.hartree_core + d.hartree_nuclear
    assert parts.pseudo == pytest.approx(d.hartree.pseudo, abs=1e-9)
    assert parts.paw == pytest.approx(d.hartree.paw, abs=1e-9)
    assert d.hartree_electrons.total == pytest.approx(
        d.hartree_valence.total + d.hartree_core.total, abs=1e-12)
    if not d.compensation_charges:
        # Nuclei are represented by compensation charges only:
        assert abs(d.hartree_nuclear.pseudo).max() == 0.0

    # One-center Coulomb corrections: core + nuclear = M_p
    for setup in calc.dft.setups:
        M_core_p, M_nuc_p = coulomb_correction_parts(setup)
        assert M_core_p + M_nuc_p == pytest.approx(setup.M_p, abs=1e-12)
        if setup.Nc == 0:  # no core electrons
            assert abs(M_core_p).max() == 0.0

    # Flat dict:
    dct = d.todict()
    assert dct['kinetic_total'] == pytest.approx(d.kinetic.total)
    assert dct['hartree_electrons_pseudo'] == pytest.approx(
        d.hartree_electrons.pseudo)
    assert (dct['bands'] == [d.n1, d.n2]).all()

    # symGW features (Hartree units, eigenvalues relative to the VBM):
    assert d.e_vbm == pytest.approx(calc.get_homo_lumo()[0], abs=1e-10)
    x = d.ks4()
    assert x.t_skn == pytest.approx(d.kinetic.total / Ha)
    assert x.vh_skn == pytest.approx(d.hartree_valence.total / Ha)
    assert x.vxc_skn == pytest.approx(d.xc.total / Ha)
    assert (x.eps_skn + x.e_vbm) * Ha == pytest.approx(d.eps_skn, abs=1e-10)
    assert x.x_sknf.shape == d.eps_skn.shape + (4,)
    assert x.x_sknf[..., 3] == pytest.approx(x.eps_skn)
    assert x.todict()['convention'][1] == 'hartree_valence.total'
    if d.compensation_charges:
        with pytest.raises(ValueError, match='compensation_charges=False'):
            d.ks4('pseudo')
    else:
        x = d.ks4('pseudo')
        assert x.t_skn == pytest.approx(d.kinetic.pseudo / Ha)
        assert x.vh_skn == pytest.approx(d.hartree_valence.pseudo / Ha)
        assert x.vxc_skn == pytest.approx(d.xc.pseudo / Ha)
    x = d.ks4({'t': 'kinetic.pseudo',
               'vh': 'hartree_valence.paw',
               'vxc': 'xc.total'}, e_vbm=0.0)
    assert x.eps_skn == pytest.approx(d.eps_skn / Ha)
    assert x.vh_skn == pytest.approx(d.hartree_valence.paw / Ha)
    with pytest.raises(ValueError, match='Unknown convention'):
        d.ks4('nope')
    with pytest.raises(ValueError, match="'t', 'vh' and 'vxc'"):
        d.ks4({'t': 'kinetic.total'})
    with pytest.raises(ValueError, match='Bad specification'):
        d.ks4({'t': 'kinetic.all', 'vh': 'hartree.total', 'vxc': 'xc.paw'})


def compare(d1, d2, abs=1e-10):
    """All terms and parts of two decompositions must agree."""
    assert d1.eps_skn == pytest.approx(d2.eps_skn, abs=abs)
    for name, term in d1.all_terms.items():
        for part in ['pseudo', 'paw']:
            assert getattr(term, part) == pytest.approx(
                getattr(d2.all_terms[name], part), abs=abs), (name, part)


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

    # Band blocks must not change anything:
    for blocksize in [1, 2]:
        d5 = kdc.calculate(n1=n1, n2=nbands, kpt_indices=[0],
                           blocksize=blocksize)
        compare(d2, d5)
    with pytest.raises(ValueError, match='blocksize'):
        kdc.calculate(blocksize=0)

    # From file name:
    d3 = ks_decomposition(gpw_files[name], n1=0, n2=-n1)
    assert d3.xc.paw == pytest.approx(d.xc.paw[:, :, :nbands - n1],
                                      abs=1e-10)

    # Without compensation charges in the pseudo parts: same totals,
    # different split:
    d4 = ks_decomposition(calc, compensation_charges=False)
    check(calc, d4)
    for name, term in d.all_terms.items():
        assert term.total == pytest.approx(
            d4.all_terms[name].total, abs=1e-9), name
    assert d.hartree.pseudo != pytest.approx(d4.hartree.pseudo, abs=0.1)
    if name == 'h_pw':
        # No core electrons:
        assert abs(d.hartree_core.total).max() == 0.0
        assert abs(d4.hartree_core.total).max() == 0.0
    if name == 'si_pw':
        # The VBM (at Gamma) is in the k-point set:
        x = d.ks4()
        assert x.eps_skn[:, :, :4].max() == pytest.approx(0.0, abs=1e-8)
        assert x.eps_skn[:, :, 4:].min() > 0.0


def test_ks_decomposition_spinpol(in_tmp_dir):
    """Non-magnetic Si with and without spin polarization.

    All terms must agree.  In particular the valence/core split of the
    electrostatic term is the only place where the number of spin
    channels enters explicitly (the smooth core density is stored per
    spin channel), so this checks the split independently of the
    closure of the decomposition.
    """
    d = {}
    for spinpol in [False, True]:
        atoms = bulk('Si')
        atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                          kpts=(2, 2, 2),
                          nbands=8,
                          spinpol=spinpol,
                          convergence={'bands': 6, 'density': 1e-6},
                          txt=f'si_spinpol_{spinpol}.txt')
        atoms.get_potential_energy()
        for cc in [True, False]:
            d[spinpol, cc] = ks_decomposition(atoms.calc, n1=0, n2=6,
                                              compensation_charges=cc)
        check(atoms.calc, d[spinpol, True])
    for cc in [True, False]:
        d0 = d[False, cc]
        d1 = d[True, cc]
        assert d1.eps_skn.shape == (2,) + d0.eps_skn.shape[1:]
        for s in range(2):
            for name, term in d0.all_terms.items():
                for part in ['pseudo', 'paw']:
                    assert getattr(term, part)[0] == pytest.approx(
                        getattr(d1.all_terms[name], part)[s],
                        abs=1e-6), (name, part, s)


def test_ks_decomposition_charged(in_tmp_dir):
    """Charged non-periodic system (H2+).

    The pieces of the electrostatic potential (valence, core, nuclear)
    are not neutral and each needs its own Gaussian neutralizing charge
    to add up to the SCF potential.
    """
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.06]],
                  magmoms=[0.5, 0.5])
    atoms.center(vacuum=2.5)
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 250},
                      charge=1,
                      convergence={'density': 1e-6},
                      txt='h2+.txt')
    atoms.get_potential_energy()
    ps = atoms.calc.dft.pot_calc.poisson_solver.poisson_solver
    assert isinstance(ps, ChargedPWPoissonSolver)
    for cc in [True, False]:
        d = ks_decomposition(atoms.calc, compensation_charges=cc)
        print(d)
        check(atoms.calc, d)
        assert abs(d.hartree_core.total).max() == 0.0
        # Electrons and nuclei have opposite signs:
        assert (d.hartree_valence.total > 0).all()
        assert (d.hartree_nuclear.total < 0).all()


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
    compare(d, d2, abs=1e-8)

    # Band, domain and k-point parallelization (when run in parallel):
    for kind in ['band', 'domain', 'kpt']:
        calc3 = GPAW('si_pbe.gpw', parallel={kind: world.size}, txt=None)
        d3 = ks_decomposition(calc3, n1=0, n2=6, blocksize=2)
        compare(d, d3, abs=1e-8)
