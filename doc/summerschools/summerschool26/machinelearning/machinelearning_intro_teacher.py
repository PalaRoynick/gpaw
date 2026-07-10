# Initialization when running this script while testing

import subprocess, os
if not os.path.exists('NequIP-OAM-S_cpu.nequip.pt2'):
    subprocess.run('nequip-compile nequip.net:mir-group/NequIP-OAM-S:0.1 '
                   'NequIP-OAM-S_cpu.nequip.pt2 --mode aotinductor '
                   '--device cpu --target ase',
                   shell=True, check=True)
if not os.path.exists('MoS2.xyz'):
    subprocess.run('curl -o MoS2.xyz '
                   'https://c2db.fysik.dtu.dk/material/1MoS2-1/download/xyz',
                   shell=True, check=True)

# Disable plotting
import ase.visualize as viz
viz.view = lambda atoms, repeat=None: None

# snippet-initialize-start
from ase.io import read, write, Trajectory
from ase.visualize import view
from ase.filters import FrechetCellFilter
from ase.optimize import QuasiNewton
from ase.constraints import FixCom
from ase.md.velocitydistribution import thermalize_momenta
from ase.md import Langevin, Bussi, VelocityVerlet
from ase import units, Atom
from nequip.integrations.ase import NequIPCalculator

import numpy as np
import matplotlib.pyplot as plt
import time

# Specify input file (structure), size, and MLIP model
supercell = (4, 4, 1)
unitcellfile = 'MoS2.xyz'  # student: unitcellfile = ????  # Your chosen material .xyz file
mlip = 'NequIP-OAM-S_cpu.nequip.pt2'

# Read the structure and verify visually that it makes sense
atoms = read(unitcellfile) * supercell
print(atoms.get_chemical_formula())
view(atoms)
# snippet-initialize-end

# snippet-mlip-start
elements_dict = {e:e for e in atoms.get_chemical_symbols()}
print('Elements dictionary:', elements_dict)

atoms.calc = NequIPCalculator.from_compiled_model(
             compile_path=mlip,
             chemical_species_to_atom_type_map = elements_dict,
             device='cpu')

print('Potential energy of system:', atoms.get_potential_energy())
# snippet-mlip-end

# snippet-minimization-start
old_scaled_pos = atoms.get_scaled_positions(wrap=False)
old_cell = np.array(atoms.cell)

# Run the energy minimzation.
ecf = FrechetCellFilter(atoms, mask=(True, True, False, False, False, True))
# Sometimes QuasiNewton is not stable, then try LBFGS instead.
qn = QuasiNewton(ecf)
traj = Trajectory('minimization.traj', 'w', atoms)
qn.attach(traj)
qn.run(fmax=0.02)

# Print what has changed
print('Old unit cell:')
print(old_cell)
print('Change in unit cell:')
with np.printoptions(precision=4, floatmode='fixed', suppress=True):
    print(np.abs(atoms.cell - old_cell))
print('Max change in scaled atomic coordinate:')
print(np.abs(atoms.get_scaled_positions(wrap=False) - old_scaled_pos).max())
# snippet-minimization-end

# snippet-md-start
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
        self.temperatures.append(T)

# Read the atoms from the input file
atoms = read(unitcellfile) * supercell
print(atoms.get_chemical_formula())

elements_dict = {e:e for e in atoms.get_chemical_symbols()}
print('Elements dictionary:', elements_dict)
print('Machine learning potential:', mlip)

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

dyn = Langevin(atoms,
               timestep=timestep * units.fs,
               temperature_K=T_wanted,
               friction=1 / (tau * units.fs),
               fixcm=False,     # Suppress a warning due to backward compatibility stuff
               logfile='-',     # Print a line with key quantities ...
               loginterval=10)  # ... but not every time step.

# Save the atomic configuration every five time steps
traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj, interval=5)
traj.write()  # Write starting configuration

# Collect the temperature while running
datacollector = DataCollector(atoms, dyn)
dyn.attach(datacollector.update)

# Run the MD:
starttime = time.perf_counter()
nstep = 1000
dyn.run(nstep)
elapsed = time.perf_counter() - starttime
print(f'MD simulation took {elapsed:.1f} seconds for {nstep} time steps.')
# snippet-md-end

# snippet-plotT-start
fig, ax = plt.subplots(1, 1, layout='constrained')
temp = np.array(datacollector.temperatures)
# Average over last two thirds of simulation
twothirds = len(temp) * 2 // 3
avgtemp = temp[-twothirds:].mean()
t = np.arange(len(temp)) * (timestep / 1000)

ax.plot(t, temp)
ax.plot([t[-twothirds], t[-1]], [avgtemp, avgtemp], 'r--')
ax.set_xlabel('Time [ps]')
ax.set_ylabel('Temperature [K]')
fig.savefig('Thermalization.png', dpi=300)
fig.show()

print(f'Average temperature: {avgtemp:.1f} K')
# snippet-plotT-end
plt.pause(0.1)

# Extra output for teacher experimentation
print(f'Expected temperature: {T_wanted * (len(atoms)-1) / len(atoms):.1f} K')
print(atoms.get_number_of_degrees_of_freedom(), len(atoms)*3)

# Delay before exiting to see plots
plt.pause(5)