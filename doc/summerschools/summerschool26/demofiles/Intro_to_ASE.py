# %% 
# Intro to the ASE

# Import modules

import numpy as np
from ase import units
from ase.io.trajectory import Trajectory
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.visualize import view
from asap3 import EMT
import os

# %%
# ### Set up the system to be simulated
# 
# 1. Make a single unit cell
# 
# 2. Repeat it to generate a (slightly) bigger system.
# 
# 3. Remove the periodic boundary conditions, and center the atoms in the unit cell.

atoms = bulk("Cu", "fcc", cubic=1)
atoms = atoms.repeat([3,3,3])
atoms.set_pbc(False)
atoms.center(vacuum=5.0)

# %%
# It is always healthy to check what you did visually.

view(atoms)

# %% 
# Give one atom a momentum (to get some action)
# 
# This will give atom 0 a kinetic energy of 1.0 eV

atoms[0].momentum = np.array([0, -11.3, 0])
print(atoms.get_kinetic_energy())

# %%
# ### Prepare to do Molecular Dynamics, forces described by EMT
# 
# The Effective Medium Theory is a computationally fast model for late transition metals.
# 
# Velocity Verlet is a standard algorithm for Molecular Dynamics.

atoms.calc = EMT()
dyn = VelocityVerlet(atoms, 5.0*units.fs, logfile='-', loginterval=5)

# Save output in a Trajectory file, but only every fifth step.
trajectory = Trajectory("MD-output.traj", "w", atoms)
dyn.attach(trajectory, interval=5)

# %% 
# Run the simulation (1000 time steps)

dyn.run(1000)

# %% 
# View the result.
# 
# We call the ASE GUI command line interface

os.system('ase gui MD-output.traj')




