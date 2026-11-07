"""Test for elph/gmatrix"""
import numpy as np
import pytest
from ase.build import bulk

from gpaw import GPAW
from gpaw.elph import ElectronPhononMatrix

# from ase.phonons import Phonons


@pytest.mark.serial
@pytest.mark.elph
def test_gmatrix(session_tmp_path, supercell_cache):
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    supercell_cache
    elph = ElectronPhononMatrix(atoms, 'supercell', 'elph',
                                use_mean_minimum_images=True)
    q = [[0, 0, 0], [0.5, 0.5, 0.5]]

    # phonon = Phonons(atoms, name='elph', supercell=(2, 1, 1),
    #                  center_refcell=True)
    # phonon.read()
    # w_ql, u_ql = phonon.band_structure(q, modes=True)
    # print(w_ql)
    # print(u_ql)
    # [[0.00124087  0.00044091  0.00132004  0.0421112   0.04212737  0.04218485]
    # [0.03031018  0.03031948  0.03041029  0.03041035  0.04326759  0.04327498]]

    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts={'size': (2, 2, 2), 'gamma': False},
                symmetry='off',
                txt='li_gs_nosym.txt')
    atoms.calc = calc
    atoms.get_potential_energy()

    parts = elph.bloch_matrix(calc, k_qc=q, savetofile=False,
                              prefactor=False, return_components=True)
    g_sqklnn = parts['total']
    assert g_sqklnn == pytest.approx(sum(parts[key]
                                         for key in ('V', 'D', 'P', 'R')))

    assert g_sqklnn.shape == (1, 2, 8, 6, 2, 2)

    # NOTE: It seems g is Hermitian if q=0 and symmetric otherwise. CHECK THIS!

    # Li has lots of degenerate phonon modes, average/sum those

    # q = 0
    print("q0 checks")

    # accoustic sum rule
    assert np.allclose(g_sqklnn[0, 0, :, 0:3], 0.)

    g_knn = np.sum(g_sqklnn[0, 0, :, 3:6], axis=1)  # modes 4-6
    assert g_knn.shape == (8, 2, 2)
    # Hermitian
    assert np.allclose(g_knn[:, 0, 1], g_knn[:, 1, 0].conj())
    # and check one specific value
    print(g_knn[0])

    # the phonon modes are degenerate. Easier to test sum_nu |g_nu|^2
    g_abs = np.sum(np.abs(g_sqklnn[0, 0, 0, 3:6, 0, 1])**2)
    print(g_abs)
    assert g_abs > 0

    # q = 1
    print("q1 checks")
    g_knn = np.sum(g_sqklnn[0, 1, :, 4:6], axis=1)  # modes 5-6
    assert g_knn.shape == (8, 2, 2)
    # Hermitian, actually symmetric
    assert np.allclose(g_knn[:, 0, 1], g_knn[:, 1, 0])
    # and check one specific value
    print(g_knn[0])

    # the phonon modes are degenerate. Easier to test sum_nu |g_nu|^2
    g_abs = np.sum(np.abs(g_sqklnn[0, 1, 0, 4:6, 0, 1])**2)
    print(g_abs)
    # Hermitian reciprocity for q=-q modulo a reciprocal lattice vector.
    rates = np.sum(abs(g_sqklnn[0, 1])**2, axis=1)
    partner = calc.wfs.kd.find_k_plus_q(np.array(q[1]))
    assert rates == pytest.approx(rates[partner].swapaxes(1, 2),
                                  rel=1e-9, abs=1e-8)

    # Independent SI conversion of mass-weighted eV/Angstrom derivatives.
    from ase import units
    normalized = elph.bloch_matrix(calc, k_qc=q[1:], savetofile=False)
    energy = elph.phonon.band_structure(np.asarray(q[1:]))[0]
    displacement = np.sqrt(units._hbar**2 / (
        2 * units._amu * units._e * energy)) * 1e10
    assert normalized[0, 0] == pytest.approx(
        g_sqklnn[0, 1] * displacement[None, :, None, None], abs=1e-12)
