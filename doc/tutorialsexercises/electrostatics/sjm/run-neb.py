import ase.io
from ase.units import Pascal, m
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.mep import interpolate, DyNEB

from gpaw import GPAW, PW
from gpaw.extensions.electrochemistry import SJM
from gpaw.solvation.sjm import SJMPower12Potential
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)


def make_calculator(index):
    # Every image needs a calculator of its own, and with it an SJM object
    # and solvent objects (cavity, dielectric, interactions) of its own:
    # each calculation keeps its state in them, so images sharing them
    # would overwrite each other's.  Hence they are all made here, afresh
    # for each image.

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
    calc = GPAW(
        # General GPAW parameters.
        mode=PW(400.),
        txt=f'gpaw-{index:d}.txt',
        gpts=(16, 16, 120),
        kpts=(9, 9, 1),
        xc='PBE',
        maxiter=1000,
        extensions=[SJM(
            # Implicit solvent parameters.
            solvation=dict(cavity=cavity, dielectric=dielectric,
                           interactions=interactions),
            # Potentiostat parameters.
            potentiostat=dict(
                target_potential=4.2,
                tol=0.01,
                excess_electrons=-0.00863))])  # guess from previous
    return calc


# Create the band of images, attaching a calc to each.
initial = ase.io.read('qn-Au111-H-seq.traj')
final = ase.io.read('qn-Au111-H-hollow-1.traj')

# Hold the water still, along with the two bottom gold layers.  Its
# rotational degrees of freedom are soft, and if the band is free to use them
# an image slides into a reoriented-water configuration that sits below the
# final state, putting a spurious well in the profile after the saddle.  The
# reaction coordinate we are after is the hydrogen moving from the top site
# to the hollow site, so freeze the solvent molecule and keep the band on it.
for image in [initial, final]:
    image.set_constraint(FixAtoms(indices=[0, 1, 3, 4, 5]))
images = [initial]
for index in range(5):
    images += [initial.copy()]
    images[-1].calc = make_calculator(index + 1)
images += [final]
# apply_constraint=True puts the frozen atoms at the same place in every
# image, so the band varies only in the coordinate we care about.
interpolate(images, apply_constraint=True)

# Create and relax the DyNEB.
neb = DyNEB(images, method='improvedtangent')
opt = FIRE(neb, logfile='neb.log', trajectory='neb.traj')
opt.run(fmax=0.05)
