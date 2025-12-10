import pytest
from ase.build import fcc111
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw import restart


# Test wrting and reading of the SJM object into the gpw file
@pytest.mark.old_gpaw_only
def test_gpw(in_tmp_dir):
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

    atoms.calc.set(sj={'target_potential': None})

    E1 = atoms.get_potential_energy()
    atoms.calc.write('test.gpw', mode='all')

    atoms2, calc = restart('test.gpw', Class=SJM)
    E2 = atoms2.get_potential_energy()
    h = atoms2.calc.stuff_for_hamiltonian[0].effective_potential
    assert h.unsolv_backside
    assert E1 == E2
