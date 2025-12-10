import pytest
from ase.build import fcc111
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM, SJMPower12Potential


@pytest.mark.old_gpaw_only
def test_cavitylike():
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10

    atoms.calc = SJM(
        sj={'target_potential': 4.5,
            'jelliumregion': {'top': 10.},
            'tol': 0.5},
        gpts=(8, 8, 32),
        poissonsolver={'dipolelayer': 'xy'},
        kpts=(4, 4, 1),
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
        interactions=[SurfaceInteraction(surface_tension=0)])

    charge = 0.2
    atoms.calc.set(sj={'jelliumregion': {'bottom': 'cavity_like'},
                       'target_potential': None,
                       'excess_electrons': 0.2})
    atoms.get_potential_energy()
    bc = atoms.calc.parameters.background_charge.todict()
    assert bc['charge'] == charge
    assert bc['z1'] == 'cavity_like'
