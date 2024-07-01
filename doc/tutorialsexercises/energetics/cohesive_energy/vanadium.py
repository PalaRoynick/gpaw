from ase import Atoms
from gpaw import GPAW
from gpaw.mom import prepare_mom_calculation

L = 12
d = 0.1
atom = Atoms('V', magmoms=[0], cell=[L, L, L + d])
atom.center()
atom.positions[0, 2] += 0.1234
atom.calc = GPAW(
    xc='PBE',
    mode={'name': 'pw', 'ecut': 800.0},
    occupations={'name': 'fermi-dirac', 'width': 0.1},
    spinpol=True,
    convergence={'eigenstates': 1e-3, 'density': 1e-3})
e_atom = atom.get_potential_energy()
prepare_mom_calculation(
    atom.calc, atom,
    [[1, 1, 1, 1,
      1,
      0, 1, 1, 1, 1,
      0, 0],
     [1, 1, 1, 1,
      0,
      0, 0, 0, 0, 0,
      0, 0]])
e_atom = atom.get_potential_energy()
