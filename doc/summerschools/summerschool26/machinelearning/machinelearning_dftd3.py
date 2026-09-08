from ase.io import read, write, Trajectory
from ase.visualize import view
from ase.calculators.dftd3 import DFTD3
from ase import units, Atom
from nequip.integrations.ase import NequIPCalculator
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt
import time

# snippet-initialize-start
# Specify input file (structure), size, and MLIP model
supercell = (4, 4, 1)
basename = 'MoS2'
unitcellfile = basename + '.xyz'
mlip = 'NequIP-OAM-S_cpu.nequip.pt2'

# Read the structure and verify visually that it makes sense
atoms = read(unitcellfile) * supercell
atoms.set_pbc(True)  # Now also periodic in z-direction
# snippet-initialize-end

# snippet-dftd3-start
# Find layer distance when using DFT-D3
nequip_calc = NequIPCalculator.from_compiled_model(
    compile_path=mlip,
    chemical_species_to_atom_type_map = True,
    device='cpu')

with TemporaryDirectory() as tempdir:
    atoms.calc = DFTD3(dft=nequip_calc, xc='PBE', directory=tempdir)

    distances = np.linspace(3.0, 5.0, 101)
    energies = []
    traj = Trajectory(f'layerdistance-{basename}.traj', "w")
    for dist in distances:
        print('Distance:', dist)
        atoms.center(vacuum=dist/2, axis=2)
        energies.append(atoms.get_potential_energy())
        traj.write(atoms)
# snippet-dftd3-end

m = np.argmin(energies)
mindist = distances[m]
print('Optimal distance:', mindist)

# Plot the total energy as a check
fig, ax = plt.subplots(1, 1, layout='constrained')
erg = np.array(energies)

ax.plot(distances, erg, '.')
ax.plot([mindist, mindist], [erg.min(), erg.max()], 'r--')
ax.set_xlabel('Interlayer distance [Å]')
ax.set_ylabel('Total energy [eV]')
fig.savefig('layerdistance_energy.png', dpi=300)
fig.show()

plt.pause(0.1)


# Delay before exiting to see plots
plt.pause(5)
