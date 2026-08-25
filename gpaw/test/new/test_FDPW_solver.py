import pytest
from ase.build import fcc111
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction,
                            Power12Potential)
from gpaw.new.ase_interface import GPAW
from gpaw.new.solvation import Solvation
from gpaw import PW


@pytest.mark.ci
@pytest.mark.new_gpaw_ready
def test_FDPWsolver():
    atoms = fcc111('Au', size=(1, 1, 1))
    atoms.center(vacuum=5, axis=2)

    params = dict(mode=PW(100),
                  gpts=(8, 8, 48),
                  kpts=(1, 1, 1),
                  convergence={
                  'energy': 50000,
                  'density': 1e+4,
                  'eigenstates': 1e+4})

    solvation = dict(
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(),
            temperature=300,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=78),
        psolver='FDsolver',
        interactions=[SurfaceInteraction(surface_tension=0)])

    atoms.calc = GPAW(
        **params,
        experimental={'backwards_compatible': False},
        extensions=[Solvation(**solvation)])
    atoms.calc.initialize(atoms)
