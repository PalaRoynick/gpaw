"""SJM extension tests that run a calculator.

They are too slow for the test suite in gpaw/test/solvation/, which keeps
the quick ones.  agts.py runs each function here as a separate task.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrochemistry import (SJM, PotentialOptimizer,
                                              Potentiostat, WorkFunctionProbe)
from gpaw.test.big.solvation.forces_new import make_atoms as nacl_atoms
from gpaw.test.big.solvation.forces_new import make_calc
from gpaw.test.solvation.test_sjm_keywords import NoCavity, Vacuum

TOL = 0.005  # V
ADJUST_TARGET = 7.5  # V
ADJUST_TOL = 0.05  # V - loose on purpose; that is what always_adjust is for


def h_atoms(convergence=None, **potentiostat) -> tuple[Atoms, SJM]:
    """One H atom in a 12 Å cell, with a vacuum SJM extension.

    Without keywords it is H at 7.5 V, the starting point of the change()
    tests: the charge it converges to and its slope are given, which takes
    the equilibration from six SCF loops to two.
    """
    if not potentiostat:
        potentiostat = dict(target_potential=7.5, excess_electrons=-0.00316,
                            slope=-384.0, tol=TOL)
        # After the mixer is reset for a new charge, the eigensolver of this
        # one-electron system plateaus at a few times 1e-6, short of the
        # default 4e-8; the potential is accurate to well within TOL.
        convergence = {'eigenstates': 1e-5}
    sjm = SJM(solvation=dict(cavity=NoCavity(), dielectric=Vacuum(),
                             interactions=[]),
              potentiostat=potentiostat)
    atoms = Atoms('H', cell=[1.4, 1.4, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        convergence=convergence, extensions=[sjm])
    return atoms, sjm


def converged_atoms() -> tuple[Atoms, SJM]:
    atoms, sjm = h_atoms()
    atoms.get_potential_energy()
    return atoms, sjm


def adjusting_atoms(always_adjust: bool) -> tuple[Atoms, SJM]:
    # Marginally convergent, as h_atoms() explains:
    return h_atoms({'eigenstates': 1e-5}, target_potential=ADJUST_TARGET,
                   excess_electrons=-0.0045, tol=ADJUST_TOL,
                   always_adjust=always_adjust)


# Changing settings without discarding the wave functions, through the SJM
# object handed to GPAW; its potentiostat is the running one,
# calc.dft.potentiostat.

def test_change_target_potential():
    atoms, sjm = converged_atoms()
    calc = atoms.calc
    ibzwfs = calc.dft.ibzwfs
    assert sjm.get_electrode_potential() == pytest.approx(7.5, abs=TOL)
    assert sjm.get_electrode_potential() == pytest.approx(
        calc.results['electrode_potential'])

    sjm.change(potentiostat={'target_potential': 7.2})
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.2, abs=TOL)
    # The SJM object holds the potentiostat that ran, so the charge it
    # arrived at is the one read back -- and recorded:
    pot = sjm.potentiostat
    assert pot is calc.dft.potentiostat
    assert pot.excess_electrons == calc.results['excess_electrons']
    assert pot.excess_electrons != pytest.approx(-0.00316)
    assert (sjm.todict()['potentiostat']['charging']['excess_electrons'] ==
            pot.excess_electrons)
    # Same wave functions - nothing was rebuilt from scratch:
    assert calc.dft.ibzwfs is ibzwfs


def test_switch_to_constant_charge():
    atoms, sjm = converged_atoms()
    calc = atoms.calc
    ibzwfs = calc.dft.ibzwfs

    sjm.change(potentiostat={'target_potential': None,
                             'excess_electrons': -0.01})
    atoms.get_potential_energy()
    assert sjm.potentiostat.excess_electrons == -0.01
    assert calc.results['excess_electrons'] == pytest.approx(-0.01)
    assert calc.dft.ibzwfs is ibzwfs

    # ... and back again, reaching the same potential as before:
    sjm.change(potentiostat={'target_potential': 7.5})
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.5, abs=TOL)
    assert calc.dft.ibzwfs is ibzwfs


def test_change_before_first_calculation():
    """The SJM object takes new settings before there is a calculation."""
    atoms, sjm = h_atoms(excess_electrons=0.0)

    # No calculation yet, so this only stores the setting:
    sjm.change(potentiostat={'excess_electrons': -0.01})
    atoms.get_potential_energy()
    assert sjm.potentiostat.excess_electrons == -0.01
    assert atoms.calc.results['excess_electrons'] == pytest.approx(-0.01)

    # Afterwards, the same object forwards to the built extension, which
    # keeps the wave functions:
    ibzwfs = atoms.calc.dft.ibzwfs
    sjm.change(potentiostat={'excess_electrons': -0.02})
    atoms.get_potential_energy()
    assert sjm.potentiostat.excess_electrons == -0.02
    assert atoms.calc.dft.ibzwfs is ibzwfs


# Stepping the charge on to the next geometry with always_adjust:

def test_step_reaches_the_next_geometry(always_adjust: bool):
    atoms, sjm = adjusting_atoms(always_adjust)
    atoms.get_potential_energy()
    pot = atoms.calc.dft.potentiostat
    charge = pot.excess_electrons

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
        assert pot.charging.excess_electrons == pytest.approx(staged)
    else:
        assert pot.excess_electrons == pytest.approx(charge)


def test_change_drops_a_staged_step():
    """An explicit charge - or turning always_adjust off - wins."""
    atoms, sjm = adjusting_atoms(always_adjust=True)
    atoms.get_potential_energy()
    pot = atoms.calc.dft.potentiostat
    assert pot.pending_excess_electrons is not None

    sjm.change(potentiostat={'excess_electrons': -0.01})
    assert pot.excess_electrons == -0.01
    assert pot.pending_excess_electrons is None

    # Nothing is staged while always_adjust is off:
    sjm.change(potentiostat={'always_adjust': False})
    atoms.get_potential_energy()
    assert pot.pending_excess_electrons is None

    sjm.change(potentiostat={'always_adjust': True})
    atoms.positions[0, 2] += 0.02
    atoms.get_potential_energy()
    assert pot.pending_excess_electrons is not None


# Geometry of the jellium region:

def _extent(pot):
    """Actual z-extent of the jellium mask, in Angstrom."""
    mask_r = pot.charging.mask_r.gather(broadcast=True)
    z_z = mask_r.desc.xyz()[0, 0, :, 2] * Bohr
    inside = mask_r.data.mean(axis=(0, 1)) > 0
    return z_z[inside].min(), z_z[inside].max()


def test_region_follows_atoms(fix_bottom: bool):
    """A relative lower limit tracks the atoms unless it is fixed."""
    atoms, _ = h_atoms(
        excess_electrons=-0.01,
        jelliumregion={'bottom': -3.0, 'fix_bottom': fix_bottom})
    atoms.get_potential_energy()
    bottom, top = _extent(atoms.calc.dft.potentiostat)
    assert bottom == pytest.approx(7.0, abs=0.2)  # 3 Å above the atom

    atoms.positions[0, 2] += 1.0
    atoms.get_potential_energy()
    new_bottom, new_top = _extent(atoms.calc.dft.potentiostat)
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


# Quick checks of change() and always_adjust on a built calculation, or after
# one SCF loop:

def test_grand_output_toggle():
    atoms, sjm = h_atoms()
    # This one needs results, but one SCF loop at constant charge will do:
    sjm.change(potentiostat={'target_potential': None})
    atoms.get_potential_energy()
    calc = atoms.calc
    grand = calc.results['energy']

    # Toggling only changes which energy is reported - no new SCF loop,
    # and the calculation is not finished (and logged) all over again:
    niter = calc.dft.scf_loop.niter

    def finished_again(dft):
        raise AssertionError('post_converged() called again')

    calc.dft.potentiostat.post_converged = finished_again
    sjm.change(potentiostat={'grand_output': False})
    canonical = calc.results['energy']
    assert calc.dft.scf_loop.niter == niter
    assert canonical == pytest.approx(
        calc.dft.energies.total_extrapolated * Ha)
    assert grand == pytest.approx(
        canonical - sjm.potentiostat.excess_electrons *
        calc.get_fermi_level())

    # ... and it is reversible:
    sjm.change(potentiostat={'grand_output': True})
    assert calc.results['energy'] == pytest.approx(grand)


def test_unknown_keyword():
    atoms, sjm = h_atoms()
    atoms.calc.initialize(atoms)
    with pytest.raises(ValueError, match='Unexpected potentiostat key'):
        sjm.change(potentiostat={'excess_electron': -0.01})


def test_nothing_to_report_before_an_scf_loop():
    """Built, but not run: no potential yet, and no traces."""
    atoms, sjm = h_atoms()
    atoms.calc.initialize(atoms)
    assert np.isnan(sjm.get_electrode_potential())
    with pytest.raises(RuntimeError, match='No results'):
        sjm.write_sjm_traces()


def test_settings_live_on_the_potentiostat():
    """change() hands every setting straight to the potentiostat."""
    atoms, sjm = h_atoms()
    atoms.calc.initialize(atoms)
    sjm.change(potentiostat=dict(
        max_iters=3, max_reruns=4, max_potential_stepsize=1.5,
        grand_output=False, always_adjust=True, mixer=0.5,
        slope_regression_depth=3, slope=-20.0))
    pot = atoms.calc.dft.potentiostat
    assert pot is sjm.potentiostat
    assert (pot.grand_output, pot.always_adjust) == (False, True)
    opt = pot.optimizer
    assert (opt.max_iters, opt.max_reruns, opt.max_potential_stepsize,
            opt.mixer, opt.slope_regression_depth, opt.slope) == (
        3, 4, 1.5, 0.5, 3, -20.0)


def test_tol_set_at_constant_charge_survives_the_switch():
    """A tol changed while at constant charge is kept across the switch.

    It used to be kept on the extension alone while at constant charge and
    had to be carried over by hand when the mode switched; now the
    optimizer holds it, which the potentiostat keeps in either mode.
    """
    atoms, sjm = h_atoms(excess_electrons=0.0, tol=0.001)
    atoms.calc.initialize(atoms)
    pot = atoms.calc.dft.potentiostat

    sjm.change(potentiostat={'tol': 0.05})  # constant-charge mode
    assert pot.optimizer.tol == pytest.approx(0.05)

    sjm.change(potentiostat={'target_potential': 3.0})
    assert pot.optimizer.tol == pytest.approx(0.05)

    # A single combined call is the same, whichever branch runs first:
    sjm.change(potentiostat={'target_potential': None})
    sjm.change(potentiostat={'tol': 0.02, 'target_potential': 4.0})
    assert pot.optimizer.tol == pytest.approx(0.02)


def test_parts_of_your_own_are_the_ones_that_run():
    """A potentiostat subclass and its parts reach the calculation as is."""
    class MyPotentiostat(Potentiostat):
        pass

    opt, probe = PotentialOptimizer(), WorkFunctionProbe()
    pot = MyPotentiostat(optimizer=opt, potentialprobe=probe)
    atoms, sjm = h_atoms()
    sjm = SJM(solvation=sjm.solvation, potentiostat=pot)
    atoms.calc = GPAW(mode='fd', kpts=(2, 2, 1), txt=None, extensions=[sjm])
    atoms.calc.initialize(atoms)
    assert atoms.calc.dft.potentiostat is pot
    assert (pot.optimizer, pot.potentialprobe) == (opt, probe)

    # Settings the optimizer does not have are refused, and the
    # calculation is left as it was:
    with pytest.raises(ValueError, match='PotentialOptimizer has no mixer'):
        sjm.change(potentiostat={'mixer': 0.5, 'tol': 0.1})
    assert opt.tol == 0.005
    sjm.change(potentiostat={'tol': 0.01})  # every optimizer has one
    assert opt.tol == 0.01


def test_always_adjust_without_target_potential():
    """It is a no-op in constant-charge mode, and says so."""
    atoms, _ = h_atoms(excess_electrons=-0.01, always_adjust=True)
    with pytest.warns(UserWarning, match='always_adjust'):
        atoms.get_potential_energy()
    pot = atoms.calc.dft.potentiostat
    assert pot.excess_electrons == -0.01
    assert pot.pending_excess_electrons is None
