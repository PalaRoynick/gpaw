# web-page: csm-overview.png

import numpy as np
from ase.io.cube import read_cube
from matplotlib.figure import Figure
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle


fig = Figure(figsize=(6., 2.3))
ax = fig.subplots()
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.99)

###########################
# Plot solvent.
###########################

with open('cavity.cube') as f:
    data = read_cube(f)

atoms = data['atoms']
cube = data['data']
flat = np.mean(cube, axis=0)

ax.imshow(flat, cmap='Blues', interpolation='spline16',
          origin='lower', vmax=1.5,
          extent=(0., atoms.cell[2, 2], 0., atoms.cell[1, 1]))

###########################
# Add atoms as circles.
###########################

atomslist = [atom for atom in atoms]
atomslist = sorted(atomslist, key=lambda atom: atom.x)
atomslist.reverse()

# Add the atoms to the plot as circles, including the periodic images
# of those that cross the top or bottom edge of the cell.
ycell = atoms.cell[1, 1]
for atom in atomslist:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    for y in (atom.y - ycell, atom.y, atom.y + ycell):
        if -radius < y < ycell + radius:
            circle = Circle((atom.z, y), radius, facecolor=color,
                            edgecolor='k', linewidth=1)
            ax.add_patch(circle)

###########################
# Touch up.
###########################

zmin = atoms.positions[:, 2].min()
zmax = atoms.positions[:, 2].max()
ymid = atoms.cell[1, 1] / 2.
ax.text(zmin / 2., ymid, 'solvent', ha='center', va='center')
ax.text((zmax + atoms.cell[2, 2]) / 2., ymid, 'solvent', ha='center',
        va='center')
ax.set_xlim(0., atoms.cell[2, 2])
ax.set_ylim(0., atoms.cell[1, 1])
ax.set_xlabel(r'$z$, $\mathrm{\AA}$')
ax.set_ylabel(r'$y$, $\mathrm{\AA}$')

fig.savefig('csm-overview.png')
