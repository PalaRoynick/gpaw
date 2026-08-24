# snippet-TcMF-start
from ase.units import kB  # Boltzmann's constant in eV/K

from gpaw import GPAW

calc_fm = GPAW('CrI3_fm.gpw', txt=None)  # ferromagnetic calculation
calc_afm = GPAW('CrI3_afm.gpw', txt=None)  # anti-ferromagnetic calculation

N = 3
# Energy per magnetic atom:
E_fm = calc_fm.get_potential_energy() / 2
E_afm = calc_afm.get_potential_energy() / 2
dE = E_afm - E_fm

# Define variables S, J and Tc here
# S = ???
# J = ???
# T_c = ???
# snippet-TcMF-end

S = 3 / 2
J = dE / S**2 / 3
T_c = N * J * S * (S + 1) / 3 / kB

# snippet-print-start
print(f'J = {J * 1000:1.3f} meV')
print(f'T_c(MF) = {T_c:1.1f} K')
# snippet-print-end

import numpy as np
assert np.isclose(J * 1000, 2.283, atol=1e-3)
assert np.isclose(T_c, 99.4, atol=1e-1)
