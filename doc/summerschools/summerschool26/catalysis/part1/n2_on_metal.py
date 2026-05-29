# snippet-imports-header-start
from ase import Atoms
from gpaw import GPAW, PW
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import hcp0001
from ase.visualize import view
from ase.io import read, write
import time
# snippet-imports-header-end

# teacher
import ase.visualize as viz
viz.view = lambda atoms, repeat=None: None

# snippet-setup-slab-start
a_Ru = 2.704  # PBE value from OQMD.org; expt value is 2.706
slab = hcp0001('Ru', a=a_Ru, size=(2, 2, 2), vacuum=5.0)

# Other metals are possible, for example Rhodium
# Rhodium is FCC so you should use fcc111(...) to set up the system
# (same arguments).
# Remember to set the FCC lattice constant, get it from OQMD.org.

# a_Rh = 3.793
# slab = fcc111('Rh', a=a_Rh, size=(2, 2, 2), vacuum=5.0)

view(slab)
# snippet-setup-slab-end

# snippet-calc-define-start
calc = GPAW(xc='PBE',
            mode=PW(350),
            kpts={'size': (4, 4, 1), 'gamma': True},
            convergence={'eigenstates': 1e-6})
slab.calc = calc
# snippet-calc-define-end

# snippet-optimize-slab-start
z = slab.positions[:, 2]
constraint = FixAtoms(mask=(z < z.min() + 1.0))
slab.set_constraint(constraint)
dyn = QuasiNewton(slab, trajectory='Ru.traj')
t = time.time()
dyn.run(fmax=0.05)
print(f'Wall time: {(time.time() - t) / 60} min.')
# snippet-optimize-slab-end

# snippet-read-traj-start
iter0 = read('Ru.traj', index=0)
print('Energy: ', iter0.get_potential_energy())
print('Forces: ', iter0.get_forces())
# snippet-read-traj-end

# snippet-final-energy-start
e_slab = slab.get_potential_energy()
print(e_slab)
# snippet-final-energy-end

# snippet-optimize-n2-start
d = 1.10
molecule = Atoms('2N', positions=[(0., 0., 0.), (0., 0., d)])
molecule.set_cell(slab.get_cell())
molecule.center()
calc_mol = GPAW(xc='PBE', mode=PW(350))
molecule.calc = calc_mol
dyn2 = QuasiNewton(molecule, trajectory='N2.traj')
dyn2.run(fmax=0.05)
e_N2 = molecule.get_potential_energy()
# snippet-optimize-n2-end

# snippet-bond-length-start
d_N2 = molecule.get_distance(0, 1)
print(d_N2)
# snippet-bond-length-end

# snippet-setup-adsorption-start
h = 1.9  # guess at the binding height
d = 1.2  # guess at the binding distance
slab.positions[4, 2] += 0.2  # pre-relax the binding metal atom.

molecule = Atoms('2N', positions=[(0, 0, 0), (0, 0, d)])
p = slab.get_positions()[4]
molecule.translate(p + (0, 0, h))
slabN2 = slab + molecule
constraint = FixAtoms(mask=(z < z.min() + 1.0))
slabN2.set_constraint(constraint)
view(slabN2)
# snippet-setup-adsorption-end

# snippet-optimize-adsorption-start
slabN2.calc = calc
dyn = QuasiNewton(slabN2, trajectory='N2Ru-top.traj', maxstep=0.02)
t = time.time()
dyn.run(fmax=0.05)
print(f'Wall time: {(time.time() - t) / 60} min.')
# snippet-optimize-adsorption-end

# snippet-adsorption-energy-start
print('Adsorption energy:', slabN2.get_potential_energy() - (e_slab + e_N2))
# snippet-adsorption-energy-end

# teacher:
print('N2 bond length:', slabN2.get_distance(8, 9))

#
#
#
#
#
# Exercise

# snippet-adsorption-hollow-start
slab = read('Ru.traj')
view(slab)
# snippet-adsorption-hollow-end

# teacher:
# This block is roughly what they should write themselves
h = 2.0
molecule = Atoms('2N', positions=[(0, 0, 0), (d_N2, 0, 0)])
# molecule.rotate('z',np.pi/6.,'COM')
hollow = (slab.positions[5] + slab.positions[6] + slab.positions[7]) / 3
molecule.translate(-molecule.get_center_of_mass() + hollow + (0, 0, h))
slabN2_new = slab + molecule
a = slabN2_new.repeat((2, 2, 1))
a.cell = slabN2_new.cell
write('N2Ru_hollow.png', a, show_unit_cell=1)

# teacher:
calc = GPAW(xc='PBE',
            mode=PW(350),
            kpts={'size': (4, 4, 1), 'gamma': True},
            convergence={'eigenstates': 1e-6})
slabN2_new.calc = calc
dyn = QuasiNewton(slabN2_new, trajectory='N2Ru.traj', maxstep=0.02)
t = time.time()
dyn.run(fmax=0.05)
print('Wall time:', time.time() - t)

# teacher:

# Note:  Ends up with N-N of 1.287 Å and 1.65 Å above the surface

# teacher:
p1 = (slab.positions[4] +
      slab.positions[5] +
      slab.positions[6]) / 3 + (0, 0, 2.0)
p2 = p1 + slab.positions[5] - slab.positions[4]
N1 = Atoms('N', positions=[p1])
N2 = Atoms('N', positions=[p2])
slab2Nads = slab + N1 + N2
a = slab2Nads.repeat((2, 2, 1))
a.cell = slab2Nads.cell
write('2NadsRu.png', a, show_unit_cell=1)

# teacher:
constraint = FixAtoms(mask=(z < z.min() + 1.0))
slab2Nads.set_constraint(constraint)
calc = GPAW(xc='PBE',
            mode=PW(350),
            kpts={'size': (4, 4, 1), 'gamma': True},
            convergence={'eigenstates': 1e-6})
slab2Nads.calc = calc
dyn = QuasiNewton(slab2Nads, trajectory='2Nads.traj')
dyn.run(fmax=0.05)
e_slab = slab.get_potential_energy()
e_N2 = read('N2.traj').get_potential_energy()
print('Adsorption energy:', slab2Nads.get_potential_energy() - (e_slab + e_N2))
