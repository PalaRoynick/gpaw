from ase.units import kB  # Boltzmann's constant in eV/K
from gpaw import GPAW

calc_fm = GPAW('CrI3_fm.gpw', txt=None)  # ferromagnetic calculation
calc_afm = GPAW('CrI3_afm.gpw', txt=None)  # anti-ferromagnetic calculation

N = 3
# Energy per magnetic atom:
E_fm = calc_fm.get_potential_energy() / 2
E_afm = calc_afm.get_potential_energy() / 2
dE = E_afm - E_fm

S = 1.5  # student: S = ???
J = dE / S**2 / 3  # student: J = ???
print(f'J = {J * 1000:1.3f} meV')

T_c = N * J * S * (S + 1) / 3 / kB  # student: T_c = ???
print(f'T_c(MF) = {T_c:1.1f} K')
