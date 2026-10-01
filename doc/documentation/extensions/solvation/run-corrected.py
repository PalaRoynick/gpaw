"""Solvated Na-on-Al(100) slab in PW mode, with a dipole correction."""
import numpy as np
import ase.io
from ase.parallel import world

from gpaw import GPAW, PW
from gpaw.new.solvation import Solvation
from gpaw.solvation import LinearDielectric, get_HW14_water_kwargs

slab = ase.io.read('slab.traj')
slab.pbc = (True, True, False)  # The dipole correction requires this.

# Same weak dielectric as in run-uncorrected.py.
solvent = get_HW14_water_kwargs()
solvent['dielectric'] = LinearDielectric(epsinf=2.)
solvation = Solvation(**solvent, dipolelayer=True)

slab.calc = GPAW(mode=PW(400),
                 xc='PBE',
                 setups={'Na': '1'},
                 kpts=(4, 4, 1),
                 txt='corrected.txt',
                 extensions=[solvation])
slab.get_potential_energy()

# Save the xy-averaged electrostatic potential and the Fermi level.
v_z = slab.calc.get_electrostatic_potential().mean(axis=(0, 1))
z_z = np.linspace(0., slab.cell[2, 2], len(v_z), endpoint=False)
efermi = slab.calc.get_fermi_level()

# The work functions at the top of the cell and above the dipole layer
# (eV), and the dipole layer on its own: its charge density (e/Å³,
# positive for positive charge) and the potential it adds (eV, in the
# same convention as v_z).
wf_top, wf_bottom = slab.calc.dft.workfunctions()
layer = slab.calc.dft.pot_calc.extensions[0].poisson_solver
z_layer, rho_layer, v_layer = layer.applied_profile()
if world.rank == 0:
    np.savez('corrected.npz', z=z_z, v=v_z, efermi=efermi,
             wf_top=wf_top, wf_bottom=wf_bottom, z_layer=z_layer,
             rho_layer=rho_layer, v_layer=v_layer)
