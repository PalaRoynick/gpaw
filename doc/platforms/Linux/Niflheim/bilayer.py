import sys

from ase.build import mx2
from gpaw.new.ase_interface import GPAW


def gpaw_test(gpu=False):
    atoms = mx2('MoS2', size=(9, 9, 1), vacuum=5.0)
    kpts = (2, 2, 1)
    params = {'convergence': {'density': 1e-06},
              'kpts': kpts,
              'random': True,
              'mode': {'ecut': 800,
                       'name': 'pw',
                       'force_complex_dtype': True},
              'occupations': {'name': 'fermi-dirac',
                              'width': 0.05},
              'mixer': {'method': 'fullspin',
                        'backend': 'pulay'},
              'txt': f'relax_gpu_{gpu}.txt',
              'parallel': {'gpu': gpu},
              'xc': 'LDA'}
    atoms.calc = GPAW(**params)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    return E, F


if __name__ == '__main__':
    gpaw_test(gpu=eval(sys.argv[1]))
