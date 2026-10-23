# web-page: delta-ne-phi.png

import pickle
import numpy as np
from matplotlib import pyplot
import ase.io


def get_potential_trace(filename):
    with open(filename, 'rb') as f:
        phi = pickle.load(f)
    phi = phi.mean(axis=(0, 1))
    # Reference the trace to the vacuum below the slab -- the "bulk" side --
    # so that the change with charge shows up in the solvent.  Not to z = 0:
    # SJM pins the potential in the solvent at the top of the cell, and in PW
    # mode the cell is periodic, so z = 0 is that same pinned point and
    # subtracting it would leave the curves referenced to the solvent.
    zs = np.linspace(0., atoms.cell[2][2], num=len(phi))
    zref = 0.5 * atoms.positions[:, 2].min()  # halfway up to the lowest atom
    return phi - phi[int(np.argmin(abs(zs - zref)))]


def get_electron_trace(filename):
    with open(filename, 'rb') as f:
        n = pickle.load(f)
    # Multiply each point by its local cell volume to convert from
    # electron density to electrons in each cube.
    vol_adjust = atoms.cell.volume / np.prod(n.shape)
    n *= vol_adjust
    z_trace = np.sum(n, axis=(0, 1))
    z_trace = np.cumsum(z_trace)
    return z_trace


excess_electrons = [-0.2, -0.1, 0., 0.1, 0.2]
atoms = ase.io.read('atoms0.0000.traj')

fig, axes = pyplot.subplots(nrows=2, figsize=(6.4, 4.0), sharex=True)
fig.subplots_adjust(right=0.99, top=0.99)

###########################
# Potential axis.
###########################

ax = axes[0]
z_trace0 = get_potential_trace('esp0.0000.pckl')

for excess_electron in excess_electrons:
    label = f'{excess_electron:.4f}'
    z_trace = get_potential_trace(f'esp{label}.pckl')
    diff = z_trace - z_trace0
    # The electrostatic potential comes back on the fine grid and the
    # all-electron density on the coarse one -- in PW mode they differ by a
    # factor of two -- so each panel gets its own z axis.
    zs_phi = np.linspace(0., atoms.cell[2][2], num=len(diff))
    ax.plot(zs_phi, diff, color='C1', linewidth=2)
    ax.text(zs_phi[-1] + 1., diff[-1], f'{excess_electron:+.1f}',
            ha='left', va='center', color='C1')

###########################
# Electrons axis.
###########################

ax = axes[1]
z_trace0 = get_electron_trace('allelectrondensity0.0000.pckl')

for excess_electron in excess_electrons:
    label = f'{excess_electron:.4f}'
    z_trace = get_electron_trace(f'allelectrondensity{label}.pckl')
    diff = z_trace - z_trace0
    zs_n = np.linspace(0., atoms.cell[2][2], num=len(diff))
    ax.plot(zs_n, diff, color='C0')
    ax.text(zs_n[-1] + 1., diff[-1], f'{excess_electron:+.1f}',
            ha='left', va='center', color='C0')

###########################
# Touch up.
###########################

axes[0].set_xlim(0., zs_phi[-1] + 4.)
axes[0].set_ylabel(r'$\Delta \phi_{xy}$, V')
axes[1].set_ylabel(r'$\Delta n_{xy}$')
axes[1].set_xlabel(r'$z$ coordinate, $\mathrm{\AA}$')

fig.savefig('delta-ne-phi.png')
