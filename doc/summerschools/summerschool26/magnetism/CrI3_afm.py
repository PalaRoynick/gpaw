from ase.io import read
from gpaw import GPAW, PW

m = 3
a = read('CrI3.xyz')
a.set_initial_magnetic_moments([m, -m, 0, 0, 0, 0, 0, 0])

Nk = 4
calc = GPAW(
    mode=PW(400),
    xc='PBE',
    kpts=(Nk, Nk, 1))
a.calc = calc
a.get_potential_energy()
calc.write('CrI3_afm.gpw')
