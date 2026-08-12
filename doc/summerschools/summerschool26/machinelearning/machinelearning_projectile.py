# Disable plotting of atoms
if True:
    import ase.visualize as viz
    viz.view = lambda atoms, repeat=None: None

from ase.io import read, write, Trajectory
from ase.visualize import view
from ase.md import Langevin, Bussi, VelocityVerlet
from ase import units, Atom
from nequip.integrations.ase import NequIPCalculator

import numpy as np
import matplotlib.pyplot as plt

mlip = 'NequIP-OAM-S_cpu.nequip.pt2'

# Read the thermalize config in again
atoms = read('md.traj')
atoms.set_constraint(None)
cell = atoms.cell
print('Unit cell:')
print(cell.array)
print('Boundary conditions:')
print(atoms.get_pbc())

# Position of projectile
z = 5.0  # Ångström above highest atom
xy = (cell[0] + cell[1]) / 2
print('XY position:', xy)
z += atoms.positions[:,2].max()
initpos = xy + np.array([0, 0, z])

print('Initial position of missile:', initpos)
assert initpos[2] < cell[2,2], 'Atom should be inside cell'

init_energy = 50.0  # eV

# snippet-addprojectile-start
projectile = Atom('Au', position=initpos)
momentum = np.sqrt(2 * projectile.mass * init_energy)
projectile.momentum = np.array([0.0, 0.0, -momentum])
atoms.append(projectile)

view(atoms)
# snippet-addprojectile-end

# Now run MD
class DataCollector:
    "Collects data about MD simulation."
    def __init__(self, atoms, dynamics):
        self.atoms = atoms
        self.dyn = dynamics
        self.energies = []
        self.temperatures = []

    def update(self):
        e = self.atoms.get_potential_energy() + self.atoms.get_kinetic_energy()
        T = self.atoms.get_temperature()
        self.energies.append(e)
        self.temperatures.append(T))

atoms.calc = NequIPCalculator.from_compiled_model(
             compile_path=mlip,
             chemical_species_to_atom_type_map = True,
             device='cpu')

# snippet-mdsetup-start
timestep = 1.0
dyn = VelocityVerlet(atoms, timestep * units.fs, logfile='-')
traj = Trajectory('md_impact.traj', 'w', atoms)
dyn.attach(traj)  # Save every time step.
traj.write()  # Write starting configuration
# snippet-mdsetup-end

datacollector = DataCollector(atoms, dyn)
dyn.attach(datacollector.update)

# Run the MD:
dyn.run(200)

# Plot the total energy as a check
fig, ax = plt.subplots(1, 1, layout='constrained')
erg = np.array(datacollector.energies)
erg -= erg[0]
erg /= len(atoms)
t = np.arange(len(erg)) * (timestep / 1000)

ax.plot(t, erg, '.')
ax.set_xlabel('Time [ps]')
ax.set_ylabel('Energy per atom [eV]')
fig.savefig('Impact_energyconservation.png', dpi=300)
fig.show()

plt.pause(0.1)


# Delay before exiting to see plots
plt.pause(5)
