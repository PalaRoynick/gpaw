"""SJM extension tests that need converged SCF loops.

They were too slow for the test suite in gpaw/test/solvation/, which keeps
the quick ones.  agts.py runs each function here as a separate task.
"""
import pytest
from ase import Atoms
from ase.units import Bohr

from gpaw.extensions.electrochemistry import SJM
from gpaw.test.big.solvation.forces_new import make_atoms as nacl_atoms
from gpaw.test.big.solvation.forces_new import make_calc
from gpaw.test.solvation.test_sjm_change import TOL, h_atoms

ADJUST_TARGET = 7.5  # V
ADJUST_TOL = 0.05  # V - loose on purpose; that is what always_adjust is for


def converged_atoms() -> tuple[Atoms, SJM]:
    atoms, sjm = h_atoms()
    atoms.get_potential_energy()
    return atoms, sjm


def adjusting_atoms(always_adjust: bool) -> tuple[Atoms, SJM]:
    # Marginally convergent, as h_atoms() explains:
    return h_atoms({'eigenstates': 1e-5}, target_potential=ADJUST_TARGET,
                   excess_electrons=-0.0045, tol=ADJUST_TOL,
                   always_adjust=always_adjust)


# Changing parameters without discarding the wave functions, through the
# SJM object handed to GPAW; it reads the live values back off the built
# extension, calc.dft.sjm.

def test_change_target_potential():
    atoms, sjm = converged_atoms()
    calc = atoms.calc
    ibzwfs = calc.dft.ibzwfs
    assert sjm.get_electrode_potential() == pytest.approx(7.5, abs=TOL)
    assert sjm.get_electrode_potential() == pytest.approx(
        calc.results['electrode_potential'])

    sjm.change(target_potential=7.2)
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.2, abs=TOL)
    # The SJM object reads the charge the potentiostat arrived at, while it
    # records the one it was given:
    assert sjm.excess_electrons == calc.dft.sjm.excess_electrons
    assert sjm.excess_electrons == calc.results['excess_electrons']
    assert sjm.excess_electrons != pytest.approx(-0.00316)
    assert sjm.todict()['excess_electrons'] == -0.00316
    # Same wave functions - nothing was rebuilt from scratch:
    assert calc.dft.ibzwfs is ibzwfs


def test_switch_to_constant_charge():
    atoms, sjm = converged_atoms()
    calc = atoms.calc
    ibzwfs = calc.dft.ibzwfs

    sjm.change(target_potential=None, excess_electrons=-0.01)
    atoms.get_potential_energy()
    assert sjm.excess_electrons == -0.01
    assert calc.results['excess_electrons'] == pytest.approx(-0.01)
    assert calc.dft.ibzwfs is ibzwfs

    # ... and back again, reaching the same potential as before:
    sjm.change(target_potential=7.5)
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.5, abs=TOL)
    assert calc.dft.ibzwfs is ibzwfs


def test_change_before_first_calculation():
    """The SJM object takes new parameters before there is a calculation."""
    atoms, sjm = h_atoms(excess_electrons=0.0)

    # No calculation yet, so this only stores the parameter:
    sjm.change(excess_electrons=-0.01)
    atoms.get_potential_energy()
    assert sjm.excess_electrons == -0.01
    assert atoms.calc.results['excess_electrons'] == pytest.approx(-0.01)

    # Afterwards, the same object forwards to the built extension, which
    # keeps the wave functions:
    ibzwfs = atoms.calc.dft.ibzwfs
    sjm.change(excess_electrons=-0.02)
    atoms.get_potential_energy()
    assert sjm.excess_electrons == -0.02
    assert atoms.calc.dft.ibzwfs is ibzwfs


# Stepping the charge on to the next geometry with always_adjust:

def test_step_reaches_the_next_geometry(always_adjust: bool):
    atoms, sjm = adjusting_atoms(always_adjust)
    atoms.get_potential_energy()
    pot = atoms.calc.dft.sjm.potentiostat
    charge = sjm.excess_electrons

    assert sjm.get_electrode_potential() == pytest.approx(ADJUST_TARGET,
                                                          abs=ADJUST_TOL)
    # Whatever is staged, the results belong to the charge that was converged:
    assert atoms.calc.results['excess_electrons'] == pytest.approx(charge)

    staged = pot.pending_excess_electrons
    if not always_adjust:
        assert staged is None
    else:
        assert staged is not None
        assert staged != pytest.approx(charge)

    # An ionic step: the staged charge is picked up before the next SCF loop.
    atoms.positions[0, 2] += 0.02
    atoms.calc.move_atoms(atoms)
    assert pot.pending_excess_electrons is None
    if always_adjust:
        assert sjm.excess_electrons == pytest.approx(staged)
        assert pot.charging.excess_electrons == pytest.approx(staged)
    else:
        assert sjm.excess_electrons == pytest.approx(charge)


def test_change_drops_a_staged_step():
    """An explicit charge - or turning always_adjust off - wins."""
    atoms, sjm = adjusting_atoms(always_adjust=True)
    atoms.get_potential_energy()
    pot = atoms.calc.dft.sjm.potentiostat
    assert pot.pending_excess_electrons is not None

    sjm.change(excess_electrons=-0.01)
    assert sjm.excess_electrons == -0.01
    assert pot.pending_excess_electrons is None

    # Nothing is staged while always_adjust is off:
    sjm.change(always_adjust=False)
    atoms.get_potential_energy()
    assert pot.pending_excess_electrons is None

    sjm.change(always_adjust=True)
    atoms.positions[0, 2] += 0.02
    atoms.get_potential_energy()
    assert pot.pending_excess_electrons is not None


# Geometry of the jellium region:

def _extent(sjm):
    """Actual z-extent of the jellium mask, in Angstrom."""
    mask_r = sjm.potentiostat.charging.mask_r.gather(broadcast=True)
    z_z = mask_r.desc.xyz()[0, 0, :, 2] * Bohr
    inside = mask_r.data.mean(axis=(0, 1)) > 0
    return z_z[inside].min(), z_z[inside].max()


def test_region_follows_atoms(fix_bottom: bool):
    """A relative lower limit tracks the atoms unless it is fixed."""
    atoms, _ = h_atoms(
        excess_electrons=-0.01,
        jelliumregion={'bottom': -3.0, 'fix_bottom': fix_bottom})
    atoms.get_potential_energy()
    bottom, top = _extent(atoms.calc.dft.sjm)
    assert bottom == pytest.approx(7.0, abs=0.2)  # 3 Å above the atom

    atoms.positions[0, 2] += 1.0
    atoms.get_potential_energy()
    new_bottom, new_top = _extent(atoms.calc.dft.sjm)
    assert new_top == pytest.approx(top)  # the top is tied to the cell
    if fix_bottom:
        assert new_bottom == pytest.approx(bottom)
    else:
        assert new_bottom == pytest.approx(bottom + 1.0, abs=0.2)


# Forces:

def test_sjm_zero_charge_forces_match_solvation():
    """SJM with no excess electrons must reproduce plain Solvation.

    An empty jellium region adds nothing to the density, so the two
    calculations differ only in that SJM reaches the solvation model
    through a wrapped extension that is not itself in the extension
    list -- every hook, the force correction included, has to be
    forwarded by hand.  Dropping the forwarding silently loses the whole
    solvation force correction, leaving forces that disagree with the
    energy and a relaxation that walks uphill.

    The two build the very same Poisson solver, once Solvation is told
    to make the dipole correction that SJM always makes, so the agreement
    here is to round-off.
    """
    atoms = nacl_atoms([10., 10., 12.], [True, True, False])

    atoms.calc = make_calc('pw', 'nacl_pw_solv.txt',
                           dipolelayer={'zero_vacuum': True})
    F_solvation = atoms.get_forces()

    atoms.calc = make_calc('pw', 'nacl_pw_sjm.txt', sjm=True)
    assert atoms.get_forces() == pytest.approx(F_solvation, abs=1e-6)
