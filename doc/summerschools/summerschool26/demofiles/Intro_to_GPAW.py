# %% 
# Intro to GPAW

from ase import Atoms
from gpaw import GPAW, PW

# %% 
# Set up a single Hydrogen atom

a = 4.  # Size of unit cell (Angstrom)
c = a / 2
# Hydrogen atom:
atom = Atoms('H',
             positions=[(c, c, c)],
             magmoms=[1],
             cell=(a, a, a))

# %%
calc = GPAW(mode=PW(400), xc='PBE')
atom.calc = calc

# %%
e1 = atom.get_potential_energy()
print('Energy', e1)

# %%
d = 0.74  # Experimental bond length
molecule = Atoms('H2',
                 positions=([c - d / 2, c, c],
                            [c + d / 2, c, c]),
                 cell=(a, a, a))

# %%
molecule.calc = calc
e2 = molecule.get_potential_energy()

# %% 
# Print the results

print('Hydrogen atom energy:     {:6.2f} eV'.format(e1))
print('Hydrogen molecule energy: {:6.2f} eV'.format(e2))
print('Atomization energy:       {:6.2f} eV'.format(2 * e1 - e2))


