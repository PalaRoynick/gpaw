from ase.io import read, write, Trajectory
from ase.parallel import parprint
from gpaw import GPAW, PW
import numpy as np

filename = 'md_impact_small.traj'
framenumbers = np.arange(55, 150, 15)
parprint('Will run DFT on these frames:', framenumbers)

kpts = (2, 2, 1)
pwcutoff = 350  # eV
output = 'dft_impact_small'

calc = GPAW(xc='PBE',
            mode={'name': 'pw', 'ecut': pwcutoff},
            kpts={'size': kpts},
            convergence={'forces': 0.01},
            txt=output + '.txt',
)

traj = Trajectory(output + '.traj', 'w')

for f in framenumbers:
    atoms = read(filename, f)
    old_forces = atoms.get_forces()
    atoms.calc = calc
    forces = atoms.get_forces()
    traj.write(atoms)

    old_forces_norm = np.linalg.norm(old_forces, axis=1)
    forces_norm = np.linalg.norm(forces, axis=1)
    diff_norm = np.linalg.norm(forces - old_forces, axis=1)

    parprint(f'Data for configuration number {f} in {filename}:')
    parprint(f'Maximal MLIP force: {old_forces_norm.max():.2f} eV/Å')
    parprint(f'Maximal DFT force:  {forces_norm.max():.2f} eV/Å')
    parprint(f'Maximal difference: {diff_norm.max():.2f} eV/Å')

parprint('All done!')
traj.close()
