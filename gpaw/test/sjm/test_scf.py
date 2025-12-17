import pytest


@pytest.mark.old_gpaw_only
@pytest.mark.ci
def test_scf(atoms, calculator):
    atoms.calc = calculator
    tol = 0.5
    pot = 3.64
    atoms.calc.set(sj={'target_potential': pot,
                       'excess_electrons': 0.02,
                       'tol': tol})
    atoms.get_potential_energy()
    assert abs(atoms.calc.get_electrode_potential() - pot) < tol
