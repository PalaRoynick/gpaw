from gpaw import GPAW
import pytest
from gpaw import LCAO, PW
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.new.do import DirectOptimizationLcao
from gpaw.new.lcao.dir_opt import DirOptLCAO
from gpaw.new.pwfd.diropt import DirOptPWFD
import numpy as np
from ase import Atoms
from gpaw.dft import DFT
import os
from gpaw.new.etdm.tools import excite
from gpaw.mom import prepare_mom_calculation


#def test_roks_directopt_groundstate_lcao(in_tmp_dir, gpw_files):
#    """ Compare results obtained with UKS and ROKS
#        for a ground state calculation
#    """
#
#    # first the UKS calculation
#    calc = GPAW(gpw_files['h2o_uks_lcao'])
#    atoms = calc.atoms
#    atoms.calc = calc
#
#    e_uks = atoms.get_potential_energy()
#
#    # then the ROKS calculation
#
#    # problem with restart from gpw file so temporarily remake calc
#    #calc = GPAW(gpw_files['h2o_roks_lcao'])
#
#    dft = DFT(atoms,
#              mode={'name': 'lcao'},
#              eigensolver={'name': 'etdm',
#                           'roks': True,
#                           'linesearch_algo': {'name': 'max-step',
#                                               'max_step': 0.05}},
#              mixer={'backend': 'no-mixing'},
#              nbands='nao',
#              basis='dzp',
#              occupations={'name': 'fixed-uniform'},
#              symmetry='off',
#              spinpol=False,
#              converge=False,
#              txt='-')
#    atoms.calc = dft.ase_calculator()
#
#    e_roks = atoms.get_potential_energy()
#
#    assert e_roks == pytest.approx(e_uks, abs=1.e-4)


#def test_roks_directopt_groundstate_T_lcao(in_tmp_dir, gpw_files):
#
#    # which molecule do we test on?
#
#    # first the UKS calculation
#    calc = GPAW(gpw_files['h2o_uks_lcao'])
#    atoms = calc.atoms
#    atoms.calc = calc
#
#    e_uks = atoms.get_potential_energy()
#
#    # then the ROKS calculation
#
#    # problem with restart from gpw file so temporarily remake calc
#    #calc = GPAW(gpw_files['h2o_roks_lcao'])
#
#    dft = DFT(atoms,
#              mode={'name': 'lcao'},
#              eigensolver={'name': 'etdm',
#                           'roks': True,
#                           'linesearch_algo': {'name': 'max-step',
#                                               'max_step': 0.05}},
#              mixer={'backend': 'no-mixing'},
#              nbands='nao',
#              basis='dzp',
#              occupations={'name': 'fixed-uniform'},
#              symmetry='off',
#              spinpol=False,
#              converge=False,
#              txt='-')
#    atoms.calc = dft.ase_calculator()
#
#    e_roks = atoms.get_potential_energy()
#
#    assert e_roks == pytest.approx(e_uks, abs=1.e-4)   


def test_roks_directopt_excitedstate_lcao(in_tmp_dir, gpw_files):

    # first the uks part, MIXED
    #calc = GPAW(gpw_files['h2o_uks_lcao'])
    #atoms = calc.atoms
    #atoms.calc = calc
    #dft = calc._dft
    #f_sn = excite(calc, 0, 0, spin=(0, 0))

    #dft.change(eigensolver={'name': 'etdm', 'excited_state': True,
    #                        'searchdir_algo': 'l-bfgs-p'},
    #           occupations={'name': 'excited', 'numbers': f_sn},
    #           )
    #           
    #atoms.calc = dft.ase_calculator()
    #e_uks_M = atoms.get_potential_energy()

    # then the uks TRIPLET

    calc = GPAW(gpw_files['h2o_uks_lcao'])
    atoms = calc.atoms
    atoms.calc = calc
    dft = calc._dft
    f_sn = excite(calc, 0, 0, spin=(0, 1))

    dft.change(eigensolver={'name': 'etdm', 'excited_state': True,
                            'searchdir_algo': 'l-bfgs-p'},
               occupations={'name': 'excited', 'numbers': f_sn},
               )
               
    atoms.calc = dft.ase_calculator()
    e_uks_T = atoms.get_potential_energy()

    # spin purified energy
    e_uks_pur = 2 * e_uks_M - e_uks_T

    # then roks
    dft = DFT(atoms,
              mode={'name': 'lcao'},
              eigensolver={'name': 'etdm',
                           'roks': True,
                           'linesearch_algo': {'name': 'max-step',
                                               'max_step': 0.05}},
              mixer={'backend': 'no-mixing'},
              nbands='nao',
              basis='dzp',
              occupations={'name': 'fixed-uniform'},
              symmetry='off',
              spinpol=False,
              converge=False,
              txt='-')
    atoms.calc = dft.ase_calculator()
    calc = atoms.calc
    e_gs = atoms.get_potential_energy()
    f_n = excite(calc, 0, 0, spin=(0, 0))
    f_n[0] = f_n[0] / 2
    dft.change(eigensolver={'name': 'etdm', 'excited_state': True},
               occupations={'name': 'excited', 'numbers': f_n})
    atoms.calc = dft.ase_calculator()

    e_ex_roks = atoms.get_potential_energy()

    # non devono essere uguali(?)
    print("Spin purified:", e_uks_pur)
    print("ROKS:", e_ex_roks)
    #assert e_ex_roks == pytest.approx(e_uks_pur, abs=1.e-4)