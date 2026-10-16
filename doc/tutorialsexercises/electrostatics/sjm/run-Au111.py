from ase.build import fcc111, molecule
from ase.units import Pascal, m

from gpaw import GPAW, PW
from gpaw.extensions import SJM
from gpaw.solvation.sjm import SJMPower12Potential
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)

# Build a tiny gold slab with a single water molecule above.
atoms = fcc111('Au', size=(1, 1, 3))
atoms.center(axis=2, vacuum=10.)
atoms.translate([0., 0., -2.])
water = molecule('H2O')
water.rotate('y', 90.)
water.positions += atoms[2].position + (0., 0., 4.4) - water[0].position
atoms.extend(water)

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

# The solvated jellium method is added to a regular GPAW calculator as an
# extension.  Keep it in a variable, to conveniently access and change its
# properties.
sjm = SJM(
    # Implicit solvent parameters.
    solvation=dict(cavity=cavity, dielectric=dielectric,
                   interactions=interactions),
    # Solvated jellium parameters.
    potentiostat=dict(target_potential=4.2))  # Desired potential

# The calculator.
calc = GPAW(
    # General GPAW parameters.
    mode=PW(400.),
    txt='Au111.txt',
    gpts=(16, 16, 120),
    kpts=(9, 9, 1),
    xc='PBE',
    maxiter=1000,
    extensions=[sjm])
atoms.calc = calc

# Run the calculation.
atoms.get_potential_energy()
atoms.write('Au111.traj')
sjm.write_sjm_traces(path='sjm_traces.out')  # .out for .gitignore
