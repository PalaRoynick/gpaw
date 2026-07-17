from ase.io import read, write, Trajectory
from ase.visualize import view
from ase.constraints import FixCom
from ase.md.velocitydistribution import thermalize_momenta
from ase.md import Langevin, Bussi, VelocityVerlet
from ase import units, Atom
from nequip.integrations.ase import NequIPCalculator

import numpy as np
import matplotlib.pyplot as plt
import time

# Specify input file (structure), size, and MLIP model
supercell = (2, 2, 1)
unitcellfile = 'MoS2.xyz'  # student: unitcellfile = ????  # Your chosen material .xyz file
mlip = 'NequIP-OAM-S_cpu.nequip.pt2'

# Read the structure and verify visually that it makes sense
atoms = read(unitcellfile) * supercell
atoms.center(vacuum=8.0, axis=2)  # Ensure enough vacuum in the unit cell
print(atoms.get_chemical_formula())
# Check that the periodic boundaries are as expected
assert (atoms.get_pbc() == [True, True, False]).all()
# view(atoms)

# Thermalize
elements_dict = {e:e for e in atoms.get_chemical_symbols()}
print('Elements dictionary:', elements_dict)
atoms.calc = NequIPCalculator.from_compiled_model(
             compile_path=mlip,
             chemical_species_to_atom_type_map = elements_dict,
             device='cpu')

T_wanted = 300  # Kelvin
timestep = 2.0  # femtoseconds
tau = 250       # Characteristic time of thermostat, femtoseconds

# Set the double temperature as half the kinetic energy goes to potential energy
atoms.set_constraint(FixCom())   # Keep center of mass fixed
thermalize_momenta(atoms, 2 * T_wanted)

rng = np.random.default_rng(seed=42)  # Make the dynamics reproducible
dyn = Langevin(atoms,
               timestep=timestep * units.fs,
               temperature_K=T_wanted,
               friction=1 / (tau * units.fs),
               fixcm=False,     # Suppress a warning due to backward compatibility stuff
               logfile='-',     # Print a line with key quantities ...
               loginterval=10,  # ... but not every time step.
               rng=rng)

# Save the atomic configuration every five time steps
traj = Trajectory('md_small.traj', 'w', atoms)
dyn.attach(traj, interval=5)
traj.write()  # Write starting configuration

starttime = time.perf_counter()
nstep = 1000
dyn.run(nstep)
elapsed = time.perf_counter() - starttime
print(f'MD simulation took {elapsed:.1f} seconds for {nstep} time steps.')

# Add a projectile

# Position of projectile
z = 5.0  # Ångström above highest atom
xy = (atoms.cell[0] + atoms.cell[1]) / 2
print('XY position:', xy)
z += atoms.positions[:,2].max()
initpos = xy + np.array([0, 0, z])

print('Initial position of missile:', initpos)
assert initpos[2] < atoms.cell[2,2], 'Atom should be inside cell'

init_energy = 50.0  # eV

projectile = Atom('Au', position=initpos)
momentum = np.sqrt(2 * projectile.mass * init_energy)
projectile.momentum = np.array([0.0, 0.0, -momentum])

# Remove constraint from thermalization and add projectile
atoms.set_constraint(None)
atoms.append(projectile)

# view(atoms)

# Now run MD, keeping track of when the forces are largest.
class DataCollector:
    "Collects data about MD simulation."
    def __init__(self, atoms, dynamics):
        self.atoms = atoms
        self.dyn = dynamics
        self.energies = []
        self.temperatures = []
        self.largest_force = []

    def update(self):
        e = self.atoms.get_potential_energy() + self.atoms.get_kinetic_energy()
        T = self.atoms.get_temperature()
        self.energies.append(e)
        self.temperatures.append(T)
        f = np.linalg.norm(self.atoms.get_forces(), axis=1)
        assert f.shape == (len(self.atoms),)
        self.largest_force.append(f.max())

# New calculator as a new element may have been introduced
elements_dict = {e:e for e in atoms.get_chemical_symbols()}
print('Elements dictionary:', elements_dict)
atoms.calc = NequIPCalculator.from_compiled_model(
             compile_path=mlip,
             chemical_species_to_atom_type_map = elements_dict,
             device='cpu')

# Run the MD:
timestep = 1.0
dyn = VelocityVerlet(atoms, timestep * units.fs, logfile='-')
traj = Trajectory('md_impact_small.traj', 'w', atoms)
dyn.attach(traj)  # Save every time step.
traj.write()  # Write starting configuration

datacollector = DataCollector(atoms, dyn)
dyn.attach(datacollector.update)

dyn.run(200)

# Plot the total energy as a check and the max force for information
fig, ax = plt.subplots(1, 2, figsize=(8,4), layout='constrained')
erg = np.array(datacollector.energies)
erg -= erg[0]
erg /= len(atoms)
t = np.arange(len(erg)) * (timestep / 1000)

ax[0].plot(t, erg, '.')
ax[0].set_xlabel('Time [ps]')
ax[0].set_ylabel('Energy per atom [eV]')

mf = np.array(datacollector.largest_force)
ax[1].plot(mf, '.-')
ax[1].set_xlabel('Time step')
ax[1].set_ylabel('Maximal force [eV/Å]')

fig.savefig('Impact_maxforce.png', dpi=300)
fig.show()

# NOTE:
# In this case interesting frames occur between time step 50 and 150,
# perhaps just sample evenly.

plt.pause(0.1)

# Delay before exiting to see plots
plt.pause(5)
