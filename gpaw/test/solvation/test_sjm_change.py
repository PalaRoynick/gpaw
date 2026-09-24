"""Changing SJM parameters without discarding the wave functions.

Parameters are changed through the SJM object handed to GPAW, which
forwards to the built extension, calc.dft.sjm; the live values are read
back off the latter.
"""
import pytest
from ase import Atoms
from ase.units import Ha

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrochemistry import SJM
from gpaw.test.solvation.test_modes import NoCavity, Vacuum

TOL = 0.005  # V


def h_atoms() -> tuple[Atoms, SJM]:
    """H at 7.5 V, the starting point of every change() test."""
    sjm = SJM(cavity=NoCavity(), dielectric=Vacuum(), interactions=[],
              target_potential=7.5, excess_electrons=-0.00316, slope=-384.0,
              tol=TOL)
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        # This one-electron system is only marginally convergent: after the
        # mixer is reset for a new charge, the eigensolver plateaus at a
        # residual of a few times 1e-6 and never reaches the default 4e-8,
        # so the equilibration runs out of SCF iterations even though it has
        # already found the right charge.  The potential it settles on is
        # accurate to well within TOL, which is all this test needs.
        convergence={'eigenstates': 1e-5},
        # The change() tests in gpaw/test/big/solvation/sjm/ start from a
        # converged constant-potential calculation and then change
        # something, so the equilibration itself is setup cost, not what is
        # under test (that is test_sjm_slope.py's job).  Starting from the
        # charge it converges to, and telling it the slope instead of
        # leaving it to guess a capacitance, takes it from six SCF loops to
        # two.
        extensions=[sjm])
    return atoms, sjm


def converged_atoms() -> tuple[Atoms, SJM]:
    atoms, sjm = h_atoms()
    atoms.get_potential_energy()
    return atoms, sjm


@pytest.fixture
def built(in_tmp_dir):
    """The built calculation, with no SCF loop: change() needs no more."""
    atoms, sjm = h_atoms()
    atoms.calc.initialize(atoms)
    return atoms, sjm


def test_grand_output_toggle(in_tmp_dir):
    atoms, sjm = h_atoms()
    # This one needs results, but one SCF loop at constant charge will do:
    sjm.change(target_potential=None)
    atoms.get_potential_energy()
    calc = atoms.calc
    grand = calc.results['energy']

    # Toggling only changes which energy is reported - no new SCF loop:
    niter = calc.dft.scf_loop.niter
    sjm.change(grand_output=False)
    canonical = calc.results['energy']
    assert calc.dft.scf_loop.niter == niter
    assert canonical == pytest.approx(
        calc.dft.energies.total_extrapolated * Ha)
    assert grand == pytest.approx(
        canonical - calc.dft.sjm.excess_electrons * calc.get_fermi_level())

    # ... and it is reversible:
    sjm.change(grand_output=True)
    assert calc.results['energy'] == pytest.approx(grand)


def test_unknown_keyword(built):
    atoms, sjm = built
    # Both routes check the keywords, before anything is changed:
    for obj in [sjm, atoms.calc.dft.sjm]:
        with pytest.raises(ValueError, match='Unexpected SJM keyword'):
            obj.change(excess_electron=-0.01)


def test_settings_live_on_the_potentiostat(built):
    """change() hands every setting straight to the potentiostat."""
    atoms, sjm = built
    sjm.change(max_iters=3, max_reruns=4, max_potential_stepsize=1.5,
               grand_output=False, always_adjust=True, mixer=0.5,
               slope_regression_depth=3, slope=-20.0)
    pot = atoms.calc.dft.sjm.potentiostat
    assert (pot.max_iters, pot.max_reruns, pot.max_potential_stepsize,
            pot.grand_output, pot.always_adjust) == (3, 4, 1.5, False, True)
    opt = pot.optimizer
    assert (opt.mixer, opt.slope_regression_depth, opt.slope) == (
        0.5, 3, -20.0)


def test_tol_set_at_constant_charge_survives_the_switch(in_tmp_dir):
    """A tol changed while at constant charge is kept across the switch.

    It used to be kept on the extension alone while at constant charge and
    had to be carried over by hand when the mode switched; now the
    potentiostat holds it in either mode.
    """
    sjm = SJM(cavity=NoCavity(), dielectric=Vacuum(), interactions=[],
              excess_electrons=0.0, tol=0.001)
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        extensions=[sjm])
    atoms.calc.initialize(atoms)
    pot = atoms.calc.dft.sjm.potentiostat

    sjm.change(tol=0.05)  # constant-charge mode
    assert pot.tol == pytest.approx(0.05)

    sjm.change(target_potential=3.0)
    assert pot.optimizer is not None
    assert pot.tol == pytest.approx(0.05)

    # A single combined call is the same, whichever branch runs first:
    sjm.change(target_potential=None)
    sjm.change(tol=0.02, target_potential=4.0)
    assert pot.tol == pytest.approx(0.02)


def test_parameters_are_readable_but_read_only(built):
    """Every keyword reads back under its own name; none can be assigned."""
    atoms, sjm = built
    ext = atoms.calc.dft.sjm
    assert ext.target_potential == 7.5
    assert ext.tol == TOL
    assert ext.excess_electrons == ext.charge  # Extension API
    assert ext.max_iters == 10
    assert ext.max_reruns == 10
    assert ext.max_potential_stepsize == 2
    assert ext.grand_output is True
    assert ext.always_adjust is False
    assert ext.mixer == 0.2
    assert ext.slope_regression_depth == 4
    assert ext.jelliumregion is ext.region

    for key, value in [('charge', -0.01),
                       ('excess_electrons', -0.01),
                       ('target_potential', 7.2),
                       ('tol', 0.01),
                       ('slope', -1.0),
                       ('jelliumregion', {})]:
        with pytest.raises(AttributeError, match='change'):
            setattr(ext, key, value)

    # ... while change() does get through, to both:
    sjm.change(tol=0.01)
    assert ext.tol == 0.01
    assert sjm.tol == 0.01
