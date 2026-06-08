import pytest

import numpy as np

from ase import Atoms
from ase.units import Bohr

from gpaw import GPAW, PW, restart
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.new.two_electron_integrals import (TwoElectronIntegrals,)
                                            #write_rank2_tensor,
                                            #write_rank4_tensor)
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.tools import sort_orbitals_according_to_energies


@pytest.mark.twoelectronintegrals
def test_two_electron_integrals(in_tmp_dir):
    d = 1.4 * Bohr
    h2 = Atoms('H2',
               positions=[[-d / 2, 0, 0],
                          [d / 2, 0, 0]])
    h2.center(vacuum=3)

    h_nn_ref = np.array([[-5.87918300e-01,  1.06435691e-16],
                         [ 6.50267395e-17,  4.40608657e-01]])
    h_nn_exc_ref = np.array([[-4.30724874e-01,  3.02534524e-15],
                             [ 3.02684329e-15, -1.90313214e-01]])
    u_tens_ref = np.array([[[[ 6.61175470e-01,  1.40357433e-18],
                             [ 1.40357433e-18,  5.84714519e-01]],
                            [[ 1.40357433e-18,  1.38657665e-01],
                             [ 1.38657665e-01, -3.84098327e-17]]],
                           [[[ 1.40357433e-18,  1.38657665e-01],
                             [ 1.38657665e-01, -3.84098327e-17]],
                            [[ 5.84714519e-01, -3.84098327e-17],
                             [-3.84098327e-17,  5.67241603e-01]]]])
    u_tens_exc_ref = np.array([[[[ 6.61175470e-01, -8.19042632e-16],
                                 [-8.19042632e-16,  4.55686029e-01]],
                                [[-8.19042632e-16,  8.44544662e-02],
                                 [ 8.44544662e-02, -1.48584252e-15]]],
                               [[[-8.19042632e-16,  8.44544662e-02],
                                 [ 8.44544662e-02, -1.48584252e-15]],
                                [[ 4.55686029e-01, -1.48584252e-15],
                                 [-1.48584252e-15,  3.92724987e-01]]]])

    calc = GPAW(mode=PW(300),
                h=0.3,
                legacy_gpaw = False,
                xc={'name': 'EXX', 'backend': 'pw'},
                basis='sz(dzp)',
                eigensolver={'name': 'etdm-fdpw', 'converge_unocc': True},
                mixer={'backend': 'no-mixing'},
                occupations={'name': 'fixed-uniform'},
                symmetry='off',
                nbands=2,
                convergence={'eigenstates': 4.0e-6})
    h2.calc = calc
    h2.get_potential_energy()
    calc.write('h2.gpw', mode='all')

    h2, calc = restart('h2.gpw', txt='-')
    # calc.initialize()
    # calc.set_positions()
    calc.initialize_positions(h2)
    
    #eigensolver = calc._dft.scf_loop.eigensolver

    # do i need to do it here?
    # there is no dm helper in new, what is the equivalent in pw?
    # calc.wfs.eigensolver.initialize_dm_helper(calc.wfs,
    #                                           calc.hamiltonian,
    #                                           calc.density, calc.log)

    Ucalc = TwoElectronIntegrals(calc, hamilton_how=1)
    Ucalc.update_wf_ref()
    Ucalc.obtain_hopping_matrix()
    Ucalc.evaluate_two_e_integrals()
    h_nn = Ucalc.H_nn
    u_tens = Ucalc.TwoeI

    assert h_nn == pytest.approx(h_nn_ref, abs=1.0e-2)
    assert u_tens == pytest.approx(u_tens_ref, abs=1.0e-2)

    calc.set(eigensolver=FDPWETDM(excited_state=True,
                                  converge_unocc=False))
    f_sn = [[0, 1]]
    prepare_mom_calculation(calc, h2, f_sn)

    h2.get_potential_energy()
    calc.write('h2.gpw', mode='all')

    h2, calc = restart('h2.gpw', txt='-')
    calc.initialize()
    calc.set_positions()
    calc.wfs.eigensolver.initialize_dm_helper(calc.wfs,
                                              calc.hamiltonian,
                                              calc.density, calc.log)
    sort_orbitals_according_to_energies(
        calc.hamiltonian, calc.wfs, use_eps=True)

    Ucalc = TwoElectronIntegrals(calc, hamilton_how=1)
    Ucalc.update_wf_ref()
    Ucalc.obtain_hopping_matrix()
    Ucalc.evaluate_two_e_integrals()
    h_nn = Ucalc.H_nn
    u_tens = Ucalc.TwoeI

    assert h_nn == pytest.approx(h_nn_exc_ref, abs=1.0e-2)
    assert u_tens == pytest.approx(u_tens_exc_ref, abs=1.0e-2)
