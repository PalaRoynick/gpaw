"""Build the Na-on-Al(100) slab and its solvent cavity (no DFT)."""
from ase.build import fcc100, add_adsorbate
from ase.io.cube import write_cube
from ase.units import Bohr

from gpaw.core import UGDesc
from gpaw.solvation import get_HW14_water_kwargs

# Same system as in the dipole-correction tutorial, but with 7.5 Angstrom
# of vacuum on each side, leaving room below the slab for the dipole
# layer. The slab is shifted laterally to put the Na in the middle of
# the cell.
slab = fcc100('Al', (2, 2, 2), a=4.05)
add_adsorbate(slab, 'Na', 4.0)
slab.center(vacuum=7.5, axis=2)
slab.translate([slab.cell[0, 0] / 2., slab.cell[1, 1] / 2., 0.])
slab.wrap()
slab.write('slab.traj')

# The same cavity as in the DFT calculations.
cavity = get_HW14_water_kwargs()['cavity']

# Evaluate the cavity on a uniform grid; it depends only on the atomic
# positions, so no electronic structure is needed.
grid = UGDesc.from_cell_and_grid_spacing(cell=slab.cell / Bohr,
                                         grid_spacing=0.1 / Bohr)
cavity.set_grid_descriptor(grid._gd)
cavity.allocate()
cavity.update_atoms(slab, print)
cavity.update(slab, None)

with open('cavity.cube', 'w') as f:
    write_cube(f, slab, cavity.g_g)
