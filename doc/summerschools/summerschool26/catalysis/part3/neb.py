import ase.visualize as viz
viz.view = lambda atoms, repeat=None: None

# snippet-imports-start
import matplotlib.pyplot as plt
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.visualize import view
from ase.mep import NEB
from ase.io import read
# snippet-imports-end

# snippet-setup-surface-start
# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
view(slab)
# snippet-setup-surface-end

# snippet-optimize-start
# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

# Use EMT potential:
slab.calc = EMT()

# Optimise initial state:
qn = BFGS(slab, trajectory='initial.traj')
qn.run(fmax=0.05)
# snippet-optimize-end

# snippet-optimize-finalstate-start
# Optimise final state:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = BFGS(slab, trajectory='final.traj')
qn.run(fmax=0.05)
# snippet-optimize-finalstate-end

# snippet-setupneb-start
initial = read('initial.traj')
final = read('final.traj')

constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])

n_im = 3  # number of images
images = [initial]
for i in range(n_im):
    image = initial.copy()
    image.calc = EMT()
    image.set_constraint(constraint)
    images.append(image)

images.append(final)

neb = NEB(images, method='improvedtangent')
neb.interpolate()
view(images)

# snippet-setupneb-end

# snippet-bfgsneb-start
qn = BFGS(neb, trajectory='neb.traj')
qn.run(fmax=0.05)
# snippet-bfgsneb-end

# snippet-viewfinalimages-start
view(images)
# snippet-viewfinalimages-end

# snippet-nebtools-start
from ase.mep import NEBTools

images = read('neb.traj@-5:')

nebtools = NEBTools(images)

# Get the calculated barrier and the energy change of the reaction.
Ef, dE = nebtools.get_barrier()
print('Barrier:', Ef)
print('Reaction energy:', dE)

# Generate new matplotlib axis - otherwise nebtools plot double.
fig, ax = plt.subplots()
nebtools.plot_band(ax)
# snippet-nebtools-end

# snippet-mpiranks-start
# This code is just for illustration
from gpaw.mpi import world
n_im = 4              # Number of images
n = world.size // n_im      # number of cpu's per image
j = 1 + world.rank // n     # image number on this cpu
assert n_im * n == world.size
# snippet-mpiranks-end

# snippet-gpaw-start
# This code is just for illustration
from gpaw import GPAW, PW
images = [initial]
for i in range(n_im):
    ranks = range(i * n, (i + 1) * n)
    image = initial.copy()
    image.set_constraint(constraint)
    if world.rank in ranks:
        calc = GPAW(mode=PW(350),
                    nbands='130%',
                    xc='PBE',  # student: ...,
                    txt=f'{i}.txt',
                    communicator=world.new_communicator(ranks))
        image.calc = calc
    images.append(image)
images.append(final)
# snippet-gpaw-end

# teacher:
from gpaw.dft import GPAW, PW
from ase.optimize import FIRE
from gpaw.mpi import world

initial = read('N2Ru.traj')
final = read('2Nads.traj')

images = [initial]
N = 4  # No. of images

z = initial.positions[:, 2]
constraint = FixAtoms(mask=(z < z.min() + 1.0))

j = world.rank * N // world.size
n = world.size // N  # number of cpu's per image

for i in range(N):
    ranks = range(i * n, (i + 1) * n)
    image = initial.copy()
    image.set_constraint(constraint)
    if world.rank in ranks:
        calc = GPAW(xc='PBE',
                    mode=PW(350),
                    nbands='130%',
                    communicator=world.new_communicator(ranks),
                    txt=f'{i}.txt',
                    kpts={'size': (4, 4, 1), 'gamma': True})
        image.calc = calc
    images.append(image)
images.append(final)

neb = NEB(images, k=1.0, parallel=True, method='improvedtangent')
neb.interpolate()
qn = FIRE(neb, logfile='neb.log', trajectory='neb.traj')
qn.run(fmax=0.1)
neb.climb = True
qn.run(fmax=0.05)

# teacher:
# Check results:
images[1:5] = read('neb.traj@-5:-1')
energies = [image.get_potential_energy() for image in images]
emax = max(energies)
assert energies[2] == emax
assert abs(emax - energies[0] - 1.32) < 0.02
d = images[2].get_distance(-1, -2)
assert abs(d - 1.777) < 0.004
