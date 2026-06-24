# Intended values (hidden from student)
structure_file = 'tight-N2Ru-top.traj'
kpts_size = (4, 4, 1)
N2_indices = (8, 9)

# snippet-vibrations-start
import numpy as np
from ase.io import read
from ase.vibrations import Vibrations

from gpaw import GPAW, PW

slab = read(structure_file)
slab.calc = GPAW(xc='PBE',
                 mode=PW(500),
                 kpts={'size': kpts_size, 'gamma': True},
                 symmetry={'point_group': False},
                 txt='vib_initial.txt')
energy_initial = slab.get_potential_energy()
np.savetxt('energy_initial.txt', [energy_initial])

vib_initial = Vibrations(slab,
                         indices=N2_indices,
                         nfree=4,
                         name='vib_initial')
vib_initial.run()
vib_initial.summary(log='vib_initial_summary.log')
for i in range(6):
    vib_initial.write_mode(i)
# snippet-vibrations-end
