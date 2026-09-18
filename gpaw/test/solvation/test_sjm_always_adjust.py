"""Stepping the charge on to the next geometry with always_adjust."""
import pytest
from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrocatalysis import SJM
from gpaw.test.solvation.test_modes import NoCavity, Vacuum

TARGET = 7.5  # V
TOL = 0.05  # V - loose on purpose; that is what always_adjust is for


def make_atoms(always_adjust: bool) -> Atoms:
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        # This one-electron system is only marginally convergent; see the
        # fixture in test_sjm_change.py.
        convergence={'eigenstates': 1e-5},
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], target_potential=TARGET,
                        excess_electrons=-0.0045, tol=TOL,
                        always_adjust=always_adjust)])
    return atoms


@pytest.mark.parametrize('always_adjust', [False, True])
def test_step_reaches_the_next_geometry(always_adjust, in_tmp_dir):
    atoms = make_atoms(always_adjust)
    atoms.get_potential_energy()
    sjm = atoms.calc.dft.sjm
    charge = sjm.excess_electrons

    assert sjm.get_electrode_potential() == pytest.approx(TARGET, abs=TOL)
    # Whatever is staged, the results belong to the charge that was converged:
    assert atoms.calc.results['excess_electrons'] == pytest.approx(charge)

    staged = sjm.potentiostat.pending_excess_electrons
    if not always_adjust:
        assert staged is None
    else:
        assert staged is not None
        assert staged != pytest.approx(charge)

    # An ionic step: the staged charge is picked up before the next SCF loop.
    atoms.positions[0, 2] += 0.02
    atoms.calc.move_atoms(atoms)
    assert sjm.potentiostat.pending_excess_electrons is None
    if always_adjust:
        assert sjm.excess_electrons == pytest.approx(staged)
        assert (sjm.potentiostat.charging.excess_electrons ==
                pytest.approx(staged))
    else:
        assert sjm.excess_electrons == pytest.approx(charge)


def test_change_drops_a_staged_step(in_tmp_dir):
    """An explicit charge - or turning always_adjust off - wins."""
    atoms = make_atoms(always_adjust=True)
    atoms.get_potential_energy()
    sjm = atoms.calc.dft.sjm
    assert sjm.potentiostat.pending_excess_electrons is not None

    sjm.change(excess_electrons=-0.01)
    assert sjm.excess_electrons == -0.01
    assert sjm.potentiostat.pending_excess_electrons is None

    # Nothing is staged while always_adjust is off:
    sjm.change(always_adjust=False)
    atoms.get_potential_energy()
    assert sjm.potentiostat.pending_excess_electrons is None

    sjm.change(always_adjust=True)
    atoms.positions[0, 2] += 0.02
    atoms.get_potential_energy()
    assert sjm.potentiostat.pending_excess_electrons is not None


def test_always_adjust_without_target_potential(in_tmp_dir):
    """It is a no-op in constant-charge mode, and says so."""
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], excess_electrons=-0.01,
                        always_adjust=True)])
    with pytest.warns(UserWarning, match='always_adjust'):
        atoms.get_potential_energy()
    assert atoms.calc.dft.sjm.excess_electrons == -0.01
    assert atoms.calc.dft.sjm.potentiostat.pending_excess_electrons is None
