"""Changing SJM settings without discarding the wave functions.

Settings are changed through the SJM object handed to GPAW, whose
potentiostat is the one running in the calculation, calc.dft.potentiostat.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.units import Ha

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrochemistry import (SJM, PotentialOptimizer,
                                              Potentiostat, WorkFunctionProbe)
from gpaw.test.solvation.test_modes import NoCavity, Vacuum

TOL = 0.005  # V


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


@pytest.fixture
def built(in_tmp_dir):
    """The built calculation, with no SCF loop: change() needs no more."""
    atoms, sjm = h_atoms()
    atoms.calc.initialize(atoms)
    return atoms, sjm


def test_grand_output_toggle(in_tmp_dir, monkeypatch):
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

    monkeypatch.setattr(calc.dft.potentiostat, 'post_converged',
                        finished_again)
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


def test_unknown_keyword(built):
    atoms, sjm = built
    with pytest.raises(ValueError, match='Unexpected potentiostat key'):
        sjm.change(potentiostat={'excess_electron': -0.01})


def test_nothing_to_report_before_an_scf_loop(built):
    """Built, but not run: no potential yet, and no traces."""
    atoms, sjm = built
    assert np.isnan(sjm.get_electrode_potential())
    with pytest.raises(RuntimeError, match='No results'):
        sjm.write_sjm_traces()


def test_settings_live_on_the_potentiostat(built):
    """change() hands every setting straight to the potentiostat."""
    atoms, sjm = built
    sjm.change(potentiostat=dict(
        max_iters=3, max_reruns=4, max_potential_stepsize=1.5,
        grand_output=False, always_adjust=True, mixer=0.5,
        slope_regression_depth=3, slope=-20.0))
    pot = atoms.calc.dft.potentiostat
    assert pot is sjm.potentiostat
    assert (pot.max_iters, pot.max_reruns, pot.max_potential_stepsize,
            pot.grand_output, pot.always_adjust) == (3, 4, 1.5, False, True)
    opt = pot.optimizer
    assert (opt.mixer, opt.slope_regression_depth, opt.slope) == (
        0.5, 3, -20.0)


def test_tol_set_at_constant_charge_survives_the_switch(in_tmp_dir):
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


def test_parts_of_your_own_are_the_ones_that_run(in_tmp_dir):
    """A potentiostat subclass and its parts reach the calculation as is."""
    class MyPotentiostat(Potentiostat):
        pass

    opt, probe = PotentialOptimizer(), WorkFunctionProbe()
    pot = MyPotentiostat(target_potential=7.5, optimizer=opt,
                         potentialprobe=probe)
    atoms, sjm = h_atoms(target_potential=7.5)
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
