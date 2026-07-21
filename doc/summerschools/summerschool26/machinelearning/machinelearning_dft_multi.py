from ase.io import read, write, Trajectory
from ase.parallel import parprint
from gpaw import GPAW, PW
import numpy as np

filename = 'md_impact_small.traj'
framenumbers = np.arange(55, 170, 10)
parprint('Will run DFT on these frames:', framenumbers)

# snippet-gpaw-start
kpts = (2, 2, 1)
pwcutoff = 350  # eV
output = 'dft_impact_small'

calc = GPAW(xc='PBE',
            mode={'name': 'pw', 'ecut': pwcutoff},
            kpts={'size': kpts},
            convergence={'forces': 0.01},
            txt=output + '.txt',
)
# snippet-gpaw-end

traj = Trajectory(output + '.traj', 'w')

for f in framenumbers:
    # snippet-oldforces-start
    atoms = read(filename, f)
    old_forces = atoms.get_forces()
    # Sanity check:
    zcoords = atoms.positions[:,2]
    if zcoords.min() < 0 or zcoords.max() > atoms.cell[2,2]:
        raise RuntimeError(f'Atom is outside unit cell in {filename} frame {f}.')
    atoms.set_pbc(True)  # Prevents warning from GPAW
    atoms.calc = calc
    forces = atoms.get_forces()
    # snippet-oldforces-end
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
