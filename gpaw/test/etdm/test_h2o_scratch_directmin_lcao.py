from gpaw import GPAW
import pytest
from gpaw import LCAO
import numpy as np
from ase import Atoms
import os


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
@pytest.mark.parametrize("spinpol", [True])
@pytest.mark.parametrize("matrix_exp", ['egdecomp'])
@pytest.mark.parametrize("searchdir_algo", ['LSR1P'])
def test_h2o(gpaw_new, representation, spinpol, matrix_exp, searchdir_algo):

    """
    temporary test that recalculates from scratch, slow
    """
    atm = h2o_maker(vacuum=5.0)
    pwd = '/Users/giulia/Desktop/reference/'

    if gpaw_new == 1:
        calc = GPAW(
             mode=LCAO(),
             basis='dzp',
             occupations={'name': 'fixed-uniform'},
             eigensolver={'name': 'etdm',
                          'representation': representation,
                          'matrix_exp': matrix_exp,
                          'searchdir_algo': searchdir_algo,
                          'linesearch_algo': {'name': 'max-step',
                                              'max_step': 0.005}},
             mixer={'backend': 'no-mixing'},
             spinpol=spinpol,
             nbands='nao',
             symmetry='off',
             maxiter=100,
             # txt=f'{pwd}temp_new_{representation}_{spinpol}_{matrix_exp}_{searchdir_algo}.txt'
             txt='-'
        )
        atm.calc = calc
        e = atm.get_potential_energy()
        f = atm.get_forces()
        target_pot_en = -13.643156256566218
        abs_force_tol, abs_en_tol = 1e-2, 1.0e-4
        f2 = np.array([[-1.11463, -1.23723, 0.0],
                       [1.35791, 0.00827, 0.0],
                       [-0.34423, 1.33207, 0.0]])
        assert e == pytest.approx(target_pot_en, abs=abs_en_tol)
        assert f == pytest.approx(f2, abs=abs_force_tol)

    else:
        calc = GPAW(
            mode=LCAO(),
            basis='dzp',
            occupations={'name': 'fixed-uniform'},
            eigensolver={'name': 'etdm',
                         'representation': representation,
                         'matrix_exp': matrix_exp,
                         'searchdir_algo': searchdir_algo,
                         'linesearch_algo': {'name': 'max-step',
                                             'max_step': 0.005}},
            spinpol=spinpol,
            mixer={'backend': 'no-mixing'},
            nbands='nao',
            symmetry='off',
            # txt=f'{pwd}temp_old_{representation}_{spinpol}_{matrix_exp}_{searchdir_algo}.txt'
            text='-'
        )

        atm.calc = calc

        e = atm.get_potential_energy()
        f = atm.get_forces()
        target_pot_en = -13.643156256566218
        abs_force_tol, abs_en_tol = 1e-2, 1.0e-4
        f2 = np.array([[-1.11463, -1.23723, 0.0],
                       [1.35791, 0.00827, 0.0],
                       [-0.34423, 1.33207, 0.0]])
        assert e == pytest.approx(target_pot_en, abs=abs_en_tol)
        assert f == pytest.approx(f2, abs=abs_force_tol)

    analyze = False
    if analyze:
        analyze_energies(representation, spinpol, matrix_exp, searchdir_algo, pwd)


