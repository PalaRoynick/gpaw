from ase.io import read, write, Trajectory
from ase.parallel import parprint
from gpaw import GPAW, PW
import numpy as np

filename = 'md_impact_small.traj'
framenumber = 55
output = f'dft_frame_{framenumber}'
kpts = (2, 2, 1)
pwcutoff = 350  # eV

atoms = read(filename, framenumber)
old_forces = atoms.get_forces()

atoms.calc = GPAW(xc='PBE',
                  mode={'name': 'pw', 'ecut': pwcutoff},
                  kpts={'size': kpts},
                  convergence={'forces': 0.01},
                  txt='-',  # output + '.txt',
)
forces = atoms.get_forces()
write(output + '.traj', atoms)

old_forces_norm = np.linalg.norm(old_forces, axis=1)
forces_norm = np.linalg.norm(forces, axis=1)
diff_norm = np.linalg.norm(forces - old_forces, axis=1)

parprint(f'Data for configuration number {framenumber} in {filename}:')
parprint(f'Maximal MLIP force: {old_forces_norm.max():.2f} eV/Å')
parprint(f'Maximal DFT force:  {forces_norm.max():.2f} eV/Å')
parprint(f'Maximal difference: {diff_norm.max():.2f} eV/Å')
