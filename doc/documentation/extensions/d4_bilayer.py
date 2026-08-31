from gpaw import GPAW
from gpaw.extensions.d4 import D4
from gpaw import PW
from ase.build import mx2

MoS2 = mx2('MoS2', a=3.2)
WSe2 = mx2('WSe2', a=3.2)

# 6.6Å of distance between layers
MoS2.positions[:, 2] += 3.3
WSe2.positions[:, 2] -= 3.3

bilayer = WSe2 + MoS2
bilayer.center(vacuum=6.0, axis=2)

calc = GPAW(mode=PW(400),
            xc='PBE',
            extensions=[D4(xc='PBE')],
            txt='out.txt')

bilayer.calc = calc
energy = bilayer.get_potential_energy()

# Acces the D4 correction energy
d4_correction = bilayer.calc.dft.d4.get_energy()
