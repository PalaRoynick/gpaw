# Script for reading previous structure and converging
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.io import read

from gpaw import GPAW, PW


for name in ['N2Ru-top.traj', '2Nads.traj']:
    slab = read(name)

    z = slab.positions[:, 2]
    constraint = FixAtoms(mask=(z < z.min() + 1.0))
    slab.set_constraint(constraint)

    slab.calc = GPAW(xc='PBE',
                     mode=PW(500),
                     txt='tight-' + name[:-4] + 'txt',
                     kpts={'size': (4, 4, 1), 'gamma': True},
                     convergence={'eigenstates': 1e-8})

    dyn = QuasiNewton(slab,
                      trajectory='tight-' + name,
                      maxstep=0.02,
                      logfile='tight-' + name[:-4] + 'log')
    dyn.run(fmax=0.01)
