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


def h2o_maker(vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):

    d = 0.9575
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d + eps, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)], **kwargs)
    H2O.center(vacuum=vacuum)
    return H2O


def analyze_energies(representation, spinpol, matrix_exp, searchdir_algo, pwd):
    import subprocess
    import sys

    print("start")
    new_file = f'{pwd}temp_new_{representation}_{spinpol}_{matrix_exp}_{searchdir_algo}.txt'
    old_file = f'{pwd}temp_old_{representation}_{spinpol}_{matrix_exp}_{searchdir_algo}.txt'
    command = f'/Users/giulia/Desktop/read_test.sh {new_file} {old_file}'
    subprocess.call([command], shell=True)

    file_new = open(f'{new_file}_numbers.txt', 'r')
    file_old = open(f'{old_file}_numbers.txt', 'r')

    new_num = []
    with file_new:
        for line in file_new:
            new_num.append(float(((line.strip('\n')).strip('c'))))
        new_num = np.array(new_num)
    file_new.close()

    old_num = []
    with file_old:
        for line in file_old:
            old_num.append(float(((line.strip('\n')).strip('c'))))
        old_num = np.array(old_num)
    file_old.close()

    output = open(f'{pwd}OUT.txt', 'a')
    sys.stdout = output

    print("################################################################")
    print("---------CALCULATION INFO---------")
    print(representation, spinpol, matrix_exp, searchdir_algo)
    print('------------------------------------')
    print("Gpaw old iterations:", len(old_num))
    print("Gpaw new iterations:", len(new_num))

    if len(new_num) < len(old_num):
        shorter = new_num
        longer = old_num
        longer_name = "old"
    else:
        shorter = old_num
        longer = new_num
        longer_name = "new"

    for i in range(len(shorter)):

        diff = new_num[i] - old_num[i]
        row = [old_num[i], new_num[i], '', diff]

        if diff == 0.:
            row[2] = 'ok'

        print('{: >12} {: >12} {: >8} {: > 3.2e}'.format(*row))

    for j in range(len(shorter), len(longer)):
        if longer_name == "new":
            print('{: >12} {: >12}'.format(' ', longer[j]))
        else:
            print('{: >12} {: >12}'.format(longer[j], ' '))
    print("################################################################")
    print()
    print()
    # os.remove(new_file)
    # os.remove(old_file)
    # os.remove(f'{new_file}_numbers.txt')
    # os.remove(f'{old_file}_numbers.txt')

# ['full', 'u-invar', 'sparse']
# [False, True]
# ['egdecomp', 'pade-approx']
# ['l-bfgs', 'l-bfgs-p', 'LSR1P']

# full, True, egdecomp, lbfgsp
@pytest.mark.parametrize("representation", ['full'])
@pytest.mark.parametrize("matrix_exp", ['egdecomp'])
@pytest.mark.parametrize("searchdir_algo", ['LSR1P'])
def test_h2o(representation, matrix_exp, searchdir_algo, gpw_files):

    """
    temporary test that recalculates from scratch, slow
    """
    atm = h2o_maker(vacuum=5.0)
    pwd = '/Users/giulia/Desktop/reference/'

    legacy_gpaw = False

    if legacy_gpaw:

        calc = GPAW(
             mode=LCAO(),
             basis='dzp',
             occupations={'name': 'fixed-uniform'},
             eigensolver={'name': 'etdm',
                          'representation': representation,
                          'matrix_exp': matrix_exp,
                          'searchdir_algo': searchdir_algo,
                          'linesearch_algo': {'name': 'max-step',
                                              'max_step': 0.05}},
             mixer={'backend': 'no-mixing'},
             spinpol=True,
             legacy_gpaw=legacy_gpaw,
             nbands='nao',
             symmetry='off',
             maxiter=100,
             # txt=f'{pwd}temp_new_{representation}_{spinpol}_{matrix_exp}_{searchdir_algo}.txt'
             txt='-'
        )
        atm.calc = calc
        e = atm.get_potential_energy()
        # calc.calculate(properties=['energy'], system_changes=None)
        f_n = excite(calc, 0, 0, spin=(0, 0))
        # f_n[0] = f_n[0]/2
        prepare_mom_calculation(calc, atm, f_n, use_fixed_occupations=True)
        calc.set(eigensolver={'name': 'etdm-lcao',
                              'excited_state': True})
        e = atm.get_potential_energy()
        f = atm.get_forces()

    else:
        dft = DFT(atm,
                  mode={'name': 'lcao'},
                  eigensolver={'name': 'etdm',
                                  'representation': representation,
                                  'matrix_exp': matrix_exp,
                                  'searchdir_algo': searchdir_algo,
                                  'linesearch_algo': {'name': 'max-step',
                                                      'max_step': 0.05}},
                  spinpol=True,
                  mixer={'backend': 'no-mixing'},
                  nbands='nao',
                  basis='dzp',
                  occupations={'name': 'fixed-uniform'},
                  symmetry='off',
                  maxiter=100,
                  converge=False,
                  convergence={'density': 1e-5},
                  txt='-')

        atm.calc = dft.ase_calculator()
        calc = atm.calc
        e = atm.get_potential_energy()

        for i in range(4,6):
            print(i)
            print(calc._dft.ibzwfs._wfs_u[0].C_nM.data[i,:10])
            print()
        # roks occupations
        f_n = excite(calc, 0, 0, spin=(0, 0))
        # this division is necessary, how to remove it????
        # f_n[0] = f_n[0] / 2
        dft.change(eigensolver={'name': 'etdm', 'excited_state': True},
                   occupations={'name': 'excited', 'numbers': f_n})
        atm.calc = dft.ase_calculator()
        e = atm.get_potential_energy()
        f = atm.get_forces()

    target_pot_en = -13.643156256566218
    abs_force_tol, abs_en_tol = 1e-2, 1.0e-4
    f2 = np.array([[-1.11463, -1.23723, 0.0],
                   [1.35791, 0.00827, 0.0],
                   [-0.34423, 1.33207, 0.0]])
    # assert e == pytest.approx(target_pot_en, abs=abs_en_tol)
    # assert f == pytest.approx(f2, abs=abs_force_tol)

    analyze = False
    if analyze:
        analyze_energies(representation, spinpol, matrix_exp, searchdir_algo, pwd)


