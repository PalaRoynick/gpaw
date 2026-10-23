# web-page: traces-Au111.png

import os
import numpy as np
from matplotlib import pyplot
import ase.io

fig, ax = pyplot.subplots(figsize=(4.6, 2.0))
fig.subplots_adjust(bottom=0.2, top=0.99, left=0.15, right=0.97)

data = np.loadtxt(os.path.join('sjm_traces.out', 'cavity.txt'),
                  delimiter=' ')
ax.plot(data[:, 0], data[:, 1], label='solvent')
data = np.loadtxt(os.path.join('sjm_traces.out', 'background_charge.txt'),
                  delimiter=' ')
# This trace is a charge density in e/Bohr^3 -- of order 1e-5 here, and
# negative for a positive number of excess electrons -- so plot its shape
# scaled to the 0-1 range of the cavity:
rho = abs(data[:, 1])
ax.plot(data[:, 0], rho / (rho.max() or 1.0), ':', label='jellium (scaled)')

# Also add atom dots.
atoms = ase.io.read('Au111.traj')
for atom in atoms:
    ax.plot(atom.z, 0.5, 'k.')

ax.set_xlabel('$z$')
ax.set_ylabel('$xy$-averaged value')
ax.legend()
fig.savefig('traces-Au111.png')
