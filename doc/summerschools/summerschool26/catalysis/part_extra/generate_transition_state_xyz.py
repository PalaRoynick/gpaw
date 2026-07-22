from ase.constraints import FixAtoms
from ase.io import Trajectory, read, write
from ase.mep import NEB
from ase.optimize import FIRE

from gpaw import GPAW, PW

initial = read('N2Ru.traj')
final = read('2Nads.traj')

images = read('neb.traj@-5:-2')
constraint = FixAtoms(indices=list(range(4)))
ts = images[1]
ts.set_constraint(constraint)
calc = GPAW(xc='PBE',
            mode=PW(800),
            convergence={'forces': 0.001},
            txt='ts.txt',
            kpts={'size': (6, 6, 1), 'gamma': True})
ts.calc = calc

neb = NEB(images, k=1.0, climb=True, method='improvedtangent')
qn = FIRE(neb, logfile='neb.log')
traj = Trajectory('ts.traj', 'w', ts)
qn.attach(traj)
qn.run(fmax=0.01)
del ts.info['adsorbate_info']  # xyz-format doesn't support that
write('TS.xyz', ts)
