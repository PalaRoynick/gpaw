"""Solvated Na-on-Al(100) slab in PW mode, without a dipole correction."""
import numpy as np
import ase.io
from ase.parallel import world

from gpaw import GPAW, PW
from gpaw.new.solvation import Solvation
from gpaw.solvation import LinearDielectric, get_HW14_water_kwargs

slab = ase.io.read('slab.traj')
slab.pbc = (True, True, False)  # As in run-corrected.py.

# Water's cavity, but a weak dielectric: water's epsinf = 78 would screen
# the slab's dipole so strongly that its artificial field is hard to see.
solvation = get_HW14_water_kwargs()
solvation['dielectric'] = LinearDielectric(epsinf=2.)

slab.calc = GPAW(mode=PW(400),
                 xc='PBE',
                 setups={'Na': '1'},
                 kpts=(4, 4, 1),
                 txt='uncorrected.txt',
                 extensions=[Solvation(**solvation)])
slab.get_potential_energy()

# Save the xy-averaged electrostatic potential and the Fermi level.
v_z = slab.calc.get_electrostatic_potential().mean(axis=(0, 1))
z_z = np.linspace(0., slab.cell[2, 2], len(v_z), endpoint=False)
efermi = slab.calc.get_fermi_level()
if world.rank == 0:
    np.savez('uncorrected.npz', z=z_z, v=v_z, efermi=efermi)
