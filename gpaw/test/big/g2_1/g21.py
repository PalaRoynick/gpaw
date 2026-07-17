from ase.build import molecule
from ase.data.g2_1 import atom_names, molecule_names
from ase.optimize.bfgs import BFGS
from gpaw.dft import DFT, PW


def relax(name):
    atoms = molecule(name)
    atoms.cell = [12, 13, 14]
    atoms.center()
    if name == 'SiH2_s3B1d':
        kwargs = dict(
            occupations=dict(name='fermi-dirac',
                             width=0.01,
                             fixmagmom=True))
    else:
        kwargs = {}
    dft = DFT(
        atoms,
        mode=PW(800),
        xc='PBE',
        **kwargs,
        txt=name + '.txt')
    atoms.calc = dft.ase_calculator()
    opt = BFGS(atoms, logfile=name + '.log')
    opt.run(0.01)


if __name__ == '__main__':
    for name in molecule_names + atom_names:
        relax(name)
