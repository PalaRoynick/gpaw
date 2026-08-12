from gpaw.spinorbit import soc_eigenstates
from gpaw import GPAW

calc_fm = GPAW('CrI3_fm.gpw')
e_x = soc_eigenstates(calc_fm, theta=90, phi=0).calculate_band_energy() / 2
e_y = soc_eigenstates(calc_fm, theta=90, phi=90).calculate_band_energy() / 2
e_z = soc_eigenstates(calc_fm, theta=0, phi=0).calculate_band_energy() / 2
de_zx = e_z - e_x
de_zy = e_z - e_y
print(f'dE_zx = {de_zx * 1000:1.3f} meV')
print(f'dE_zy = {de_zy * 1000:1.3f} meV')
S = 1.5  # student: S = ???
A = (de_zx + de_zy) / 2 / S**2  # student: A = ???
print(f'A = {A * 1000:1.3f} meV')
