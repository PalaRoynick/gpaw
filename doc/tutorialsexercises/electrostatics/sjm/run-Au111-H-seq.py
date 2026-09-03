from ase import Atom
import ase.io
from ase.units import Pascal, m
from ase.optimize import BFGS
from ase.constraints import FixAtoms

from gpaw import GPAW
from gpaw.new.electrocatalysis.sjm import SJM
from gpaw.solvation.sjm import SJMPower12Potential
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)

# Add an H adsorbate.
atoms = ase.io.read('Au111.traj')
atoms.append(Atom('H', atoms[2].position + (0., 0., 1.5)))

# Fix some atoms.
atoms.set_constraint(FixAtoms(indices=[0, 1]))

# Implicit solvent parameters.
epsinf = 78.36  # dielectric constant of water at 298 K
gamma = 18.4 * 1e-3 * Pascal * m
cavity = EffectivePotentialCavity(
    effective_potential=SJMPower12Potential(
        H2O_layer={'style': 'ghost_atoms'}),
    temperature=298.15,  # K
    surface_calculator=GradientSurface())
dielectric = LinearDielectric(epsinf=epsinf)
interactions = [SurfaceInteraction(surface_tension=gamma)]

# The calculator
calc = GPAW(
    # General GPAW parameters.
    mode='fd',
    txt='Au111-H-seq.txt',
    gpts=(16, 16, 136),
    kpts=(9, 9, 1),
    xc='PBE',
    maxiter=1000,
    extensions=[SJM(
        # Solvated jellium parameters.
        target_potential=4.2,
        # Implicit solvent parameters.
        cavity=cavity,
        dielectric=dielectric,
        interactions=interactions)])
atoms.calc = calc

# Run the calculation.
opt = BFGS(atoms, trajectory='qn-Au111-H-seq.traj',
           logfile='qn-Au111-H-seq.log')
opt.run()
