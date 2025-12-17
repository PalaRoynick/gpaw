import pytest
from ase.build import fcc111
from gpaw import FermiDirac
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM, SJMPower12Potential


@pytest.fixture
def atoms():
    # pytest.skip('https://gitlab.com/gpaw/gpaw/-/issues/1381')
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10
    return atoms


@pytest.fixture
def calculator():
    return SJM(
        sj={'target_potential': 4.5,
            'jelliumregion': {'top': 10.},
            'tol': 0.5},
        gpts=(8, 8, 32),
        poissonsolver={'dipolelayer': 'xy'},
        kpts=(4, 4, 1),
        xc='PBE',
        spinpol=False,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1,
                     'density': 1.0,
                     'eigenstates': 4.0,
                     'bands': 'occupied',
                     'forces': float('inf'),
                     'work function': 1},
        mode='lcao',
        basis='dzp',
        cavity=EffectivePotentialCavity(
            effective_potential=SJMPower12Potential(),
            temperature=298.15,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=78.36),
        interactions=[SurfaceInteraction(surface_tension=0.00114843767916)])
