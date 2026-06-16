from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates

calc = GPAW('VI2_relaxed.gpw')
e_x, e_y, e_z = (
    soc_eigenstates(calc, theta=theta, phi=phi).calculate_band_energy()
    for theta, phi in [(90, 0), (90, 90), (0, 0)])
de_zx = e_z - e_x
de_zy = e_z - e_y
print(f'dE_zx = {de_zx * 1000:1.3f} meV')
print(f'dE_zy = {de_zy * 1000:1.3f} meV')
S = 1.5
A = (de_zx + de_zy) / 2 / S**2
print(f'A = {A * 1000:1.3f} meV')
