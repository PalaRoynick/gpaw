from ase.io import read, write, Trajectory
from ase.visualize import view
from ase import Atoms
from ase.parallel import parprint
from gpaw import GPAW, PW
import numpy as np

filename = 'md_impact_small.traj'
kpts = (2, 2, 1)
pwcutoff = 350  # eV

original_atoms = read(filename)
cell = original_atoms.cell

# Find the elements
elements = set()
for atom in original_atoms:
    elements.add(atom.symbol)

print(elements)

energies = {}

# We need to sort elements, as the order of the set will vary between MPI tasks! 
for element in sorted(elements):
    atoms = Atoms(element, positions=[[0,0,0]], cell=cell, pbc=True)
    atoms.center()
    atoms.calc = GPAW(xc='PBE',
                      mode={'name': 'pw', 'ecut': pwcutoff},
                      kpts={'size': kpts},
                      txt=f'{element}.txt',
                     )
    energies[element] = atoms.get_potential_energy()

for element, energy in energies.items():
    parprint(f'  {element}: {energy}')

