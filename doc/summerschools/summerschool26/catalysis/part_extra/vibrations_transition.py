from ase.io import read
from ase.vibrations import Vibrations

from gpaw import GPAW, PW

slab = read('TS.xyz')
slab.calc = GPAW(xc='PBE',
                 mode=PW(800),
                 kpts={'size': (6, 6, 1), 'gamma': True},
                 symmetry={'point_group': False},
                 txt='vib_transition.txt')
slab.get_potential_energy()

vib = Vibrations(slab,
                 indices=(8, 9),
                 nfree=4,
                 delta=0.02,
                 name='vib_transition')
vib.run()
vib.summary(log='vib_transition_summary.log')
for i in range(6):
    vib.write_mode(i)

# ---------------------
#   #    meV     cm^-1
# ---------------------
#   0   72.3i    583.1i
#   1    5.4      43.9
#   2   41.7     336.0
#   3   47.4     382.3
#   4   70.1     565.7
#   5   71.3     575.0
# ---------------------
# Zero-point energy: 0.118 eV
# The imaginary mode is beautifully along the reaction coordinate!
