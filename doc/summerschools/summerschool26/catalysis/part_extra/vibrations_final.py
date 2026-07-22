import numpy as np
from ase.io import read
from ase.vibrations import Vibrations

from gpaw import GPAW, PW

slab = read('tight-2Nads.traj')
slab.calc = GPAW(xc='PBE',
                 mode=PW(500),
                 kpts={'size': (4, 4, 1), 'gamma': True},
                 symmetry={'point_group': False},
                 txt='vib_final.txt')
energy_final = slab.get_potential_energy()
np.savetxt('energy_final.txt', [energy_final])

vib_final = Vibrations(slab,
                       indices=(8, 9),
                       nfree=4,
                       name='vib_final')
vib_final.run()
vib_final.summary(log='vib_final_summary.log')
for i in range(6):
    vib_final.write_mode(i)
