import pytest
from gpaw import FermiDirac
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM, SJMPower12Potential


# @pytest.mark.skip('See #1510')
@pytest.mark.old_gpaw_only
# @pytest.mark.ci maybe
def test_dirichlet(atoms, calculator):
    atoms.calc = calculator
    atoms.calc.set(sj={'tol': 0.01})
    E1 = atoms.get_potential_energy()
    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']

    atoms.calc = SJM(
        sj={'target_potential': 4.5,
            'jelliumregion': {'top': 10.},
            'tol': 0.01,
            'dirichlet': True},
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
    E2 = atoms.get_potential_energy()

    assert abs(atoms.calc.get_electrode_potential() - pot) < tol
    assert abs(-atoms.calc.get_fermi_level() - pot) < tol
    assert abs(E1 - E2) < 1e-4
