import ase.io
from ase.units import Pascal, m
from ase.optimize import BFGS

from gpaw import GPAW
from gpaw.extensions.electrocatalysis import SJM
from gpaw.solvation.sjm import SJMPower12Potential
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)

# Open the previous simulation and move the H to a hollow site.
atoms = ase.io.read('qn-Au111-H-seq.traj')
atoms[6].x = atoms[0].x
atoms[6].y = atoms[0].y

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

# The calculator.  We start with a loose potential tolerance, to move the
# atoms cheaply, and tighten it below.
calc = GPAW(
    # General GPAW parameters.
    mode='fd',
    txt='Au111-H-hollow.txt',
    gpts=(16, 16, 136),
    kpts=(9, 9, 1),
    xc='PBE',
    maxiter=1000,
    extensions=[SJM(
        # Solvated jellium parameters.
        target_potential=4.2,
        tol=0.2,
        excess_electrons=-0.01232,  # guess from previous
        # Implicit solvent parameters.
        cavity=cavity,
        dielectric=dielectric,
        interactions=interactions)])
atoms.calc = calc

# Run the calculation.
opt = BFGS(atoms, trajectory='qn-Au111-H-hollow.traj',
           logfile='qn-Au111-H-hollow.log')
opt.run()

# Tighten the tolerance again.  change() keeps the wave functions.
calc.sjm.change(tol=0.01)
opt = BFGS(atoms, trajectory='qn-Au111-H-hollow-1.traj',
           logfile='qn-Au111-H-hollow-1.log')
opt.run()
