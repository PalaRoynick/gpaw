from ase import Atoms
from ase.mep import NEB
from ase.optimize import BFGS
from gpaw import GPAW, PW
from ase.constraints import FixAtoms

ccdist = 1.39
a = ccdist * 3**0.5
c = 3.7

initial = Atoms(
    'C6Li',
    positions=[(0, 0, 0),
               (0, ccdist, 0),
               (a, 0, 0),
               (-a, 0, 0),
               (-a / 2, -ccdist / 2, 0),
               (a / 2, -ccdist / 2, 0),
               (0, -ccdist, c / 2)],
    cell=([1.5 * a, -1.5 * ccdist, 0],
          [1.5 * a, 1.5 * ccdist, 0],
          [0, 0, c]),
    pbc=(1, 1, 1))
# snippet-final
final = initial.copy()
a, b = final.cell[:2]
final.positions[6] += a / 3 + b / 3
# snippet-neb1
# These will become the minimum energy path images:
images = [initial]
images += [initial.copy() for i in range(5)]
images += [final]
# snippet-neb2
neb = NEB(images, method='improvedtangent')
neb.interpolate()
# snippet-constraint-gpaw
# Constrain C-atoms:
mask = [atom.symbol == 'C' for atom in initial]
constraint = FixAtoms(mask=mask)
for image in images[0:7]:
    calc = GPAW(
        mode=PW(500),
        kpts=(5, 5, 6),
        xc='LDA',
        txt=None,
        symmetry={'point_group': False})
    image.calc = calc
    image.set_constraint(constraint)

images[3].rattle(stdev=0.05, seed=42)
# snippet-initial-final
images[0].get_potential_energy()
images[0].get_forces()
images[6].get_potential_energy()
images[6].get_forces()
# snippet-optimize
optimizer = BFGS(neb, trajectory='neb.traj', logfile='neb.log')
optimizer.run(fmax=0.10)
