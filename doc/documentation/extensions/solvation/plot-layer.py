# web-page: csm-layer.png
import numpy as np
import ase.io
from ase.data import covalent_radii as radii
from matplotlib.figure import Figure
from matplotlib.patches import Circle


fig = Figure(figsize=(6.4, 2.2))
ax_potential = fig.subplots()
fig.subplots_adjust(bottom=0.2, top=0.99, left=0.11, right=0.86)
atoms = ase.io.read('slab.traj')
data = np.load('corrected.npz')
z = data['z_layer']

###########################
# Potential axis.
###########################

ax_potential.plot(z, data['v_layer'], color='C1', linewidth=2)

###########################
# Charge axis.
###########################

ax_charge = ax_potential.twinx()
rho = data['rho_layer']
ax_charge.fill_between(z, rho, 0., color='C0', alpha=0.4, linewidth=0)
ax_charge.plot(z, rho, color='C0', linewidth=1)

###########################
# Atoms axis.
###########################

ax_atoms = ax_potential.twinx()

# Add the atoms to the plot as circles, including the periodic images
# of those that cross the top or bottom edge of the cell.
ycell = atoms.cell[1, 1]
for atom in atoms:
    radius = radii[atom.number]
    for y in (atom.y - ycell, atom.y, atom.y + ycell):
        if -radius < y < ycell + radius:
            circle = Circle((atom.z, y), radius, facecolor='none',
                            edgecolor='0.5', linewidth=0.5)
            ax_atoms.add_patch(circle)

###########################
# Touch up.
###########################

# Equal aspect for the atoms, centered on the cell in y.
zcell = atoms.cell[2, 2]
ax_potential.set_xlim(0., zcell)
bbox = ax_atoms.get_window_extent()
yspan = zcell * bbox.height / bbox.width
ax_atoms.set_ylim((ycell - yspan) / 2., (ycell + yspan) / 2.)
ax_potential.set_xlabel(r'$z$ coordinate, $\mathrm{\AA}$')
ax_atoms.set_yticks([])

# Center the charge axis on zero.
rhomax = 1.1 * abs(rho).max()
ax_charge.set_ylim(-rhomax, rhomax)

ax_potential.set_zorder(3)
ax_charge.set_zorder(2)
ax_atoms.set_zorder(1)
ax_potential.patch.set_visible(False)
ax_charge.patch.set_visible(False)

ax_potential.set_ylabel(r'$\phi_{xy}$, V', color='C1')
ax_charge.set_ylabel(r'$\rho_{xy}$, $e/\mathrm{\AA}^3$', color='C0')

fig.savefig('csm-layer.png')
