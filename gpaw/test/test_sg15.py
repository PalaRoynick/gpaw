from gpaw.dft import DFT
from ase import Atoms
from ase.data import atomic_numbers
from pathlib import Path

SG15_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
    'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Hg', 'Tl', 'Pb', 'Bi']


def test_all():
    for s in SG15_SYMBOLS:
        a = Atoms(s, cell=[4, 5, 6], pbc=1)
        DFT(
            a,
            xc='PBE',
            setups='sg15',
            # random=True
            mode='pw')


if __name__ == '__main__':
    test_all()
    if 0:
        symbols = set()
        for f in (Path.home() / 'sg15_oncv_upf_2020-02-06/').glob('*_*.upf'):
            symbols.add(f.name.split('_')[0])
        print(sorted(symbols, key=lambda s: atomic_numbers[s]))
