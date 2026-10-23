# web-page: csm-potential.png
import numpy as np
import ase.io
from ase.data import covalent_radii as radii
from matplotlib.figure import Figure
from matplotlib.patches import Circle


fig = Figure(figsize=(6.4, 2.2))
ax_potential = fig.subplots()
fig.subplots_adjust(bottom=0.2, top=0.99, left=0.11, right=0.99)
atoms = ase.io.read('slab.traj')

###########################
# Potential axis.
###########################

data = np.load('uncorrected.npz')
ax_potential.plot(data['z'], data['v'], color='C1', linewidth=2)
ax_potential.axhline(data['efermi'], color='0.3', linestyle='--',
                     linewidth=1)
ax_potential.text(0.3, data['efermi'], 'Fermi level', ha='left',
                  va='bottom', color='0.3')

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

ax_potential.set_zorder(2)
ax_atoms.set_zorder(1)
ax_potential.patch.set_visible(False)

ax_potential.set_ylabel(r'$\phi_{xy}$, V')

fig.savefig('csm-potential.png')
