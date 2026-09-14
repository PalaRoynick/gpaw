"""Changing SJM parameters without discarding the wave functions."""
import pytest
from ase import Atoms
from ase.units import Ha

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrocatalysis.sjm import SJM
from gpaw.test.solvation.test_modes import NoCavity, Vacuum

TOL = 0.005  # V


@pytest.fixture
def atoms(in_tmp_dir):
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
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], target_potential=7.5,
                        excess_electrons=-0.0045, tol=TOL)])
    atoms.get_potential_energy()
    return atoms


def test_change_target_potential(atoms):
    calc = atoms.calc
    sjm = calc.dft.sjm
    ibzwfs = calc.dft.ibzwfs
    assert sjm.get_electrode_potential() == pytest.approx(7.5, abs=TOL)

    sjm.change(target_potential=7.2)
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.2, abs=TOL)
    # Same wave functions - nothing was rebuilt from scratch:
    assert calc.dft.ibzwfs is ibzwfs


def test_switch_to_constant_charge(atoms):
    calc = atoms.calc
    sjm = calc.dft.sjm
    ibzwfs = calc.dft.ibzwfs

    sjm.change(target_potential=None, excess_electrons=-0.01)
    atoms.get_potential_energy()
    assert sjm.charge == -0.01
    assert calc.results['excess_electrons'] == pytest.approx(-0.01)
    assert calc.dft.ibzwfs is ibzwfs

    # ... and back again, reaching the same potential as before:
    sjm.change(target_potential=7.5)
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.5, abs=TOL)
    assert calc.dft.ibzwfs is ibzwfs


def test_grand_output_toggle(atoms):
    calc = atoms.calc
    sjm = calc.dft.sjm
    grand = calc.results['energy']

    # Toggling only changes which energy is reported - no new SCF loop:
    niter = calc.dft.scf_loop.niter
    sjm.change(grand_output=False)
    canonical = calc.results['energy']
    assert calc.dft.scf_loop.niter == niter
    assert canonical == pytest.approx(
        calc.dft.energies.total_extrapolated * Ha)
    assert grand == pytest.approx(
        canonical - sjm.charge * calc.get_fermi_level())

    # ... and it is reversible:
    sjm.change(grand_output=True)
    assert calc.results['energy'] == pytest.approx(grand)


def test_unknown_keyword(atoms):
    with pytest.raises(ValueError, match='Unexpected SJM keyword'):
        atoms.calc.sjm.change(excess_electron=-0.01)


def test_change_before_first_calculation(in_tmp_dir):
    """calc.sjm reaches the parameters before there is a calculation."""
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], excess_electrons=0.0)])

    # No calculation yet, so this only stores the parameter:
    atoms.calc.sjm.change(excess_electrons=-0.01)
    with pytest.raises(AttributeError, match='No calculation'):
        atoms.calc.dft

    atoms.get_potential_energy()
    assert atoms.calc.dft.sjm.charge == -0.01
    assert atoms.calc.results['excess_electrons'] == pytest.approx(-0.01)

    # Once built, calc.sjm is the extension itself:
    assert atoms.calc.sjm is atoms.calc.dft.sjm

    # ... and setting through it still keeps the wave functions:
    ibzwfs = atoms.calc.dft.ibzwfs
    atoms.calc.sjm.change(excess_electrons=-0.02)
    atoms.get_potential_energy()
    assert atoms.calc.dft.sjm.charge == -0.02
    assert atoms.calc.dft.ibzwfs is ibzwfs


def test_unknown_extension_name(atoms):
    with pytest.raises(AttributeError, match='no attribute'):
        atoms.calc.nosuchextension


def test_settings_live_on_the_potentiostat(atoms):
    """change() hands every setting straight to the potentiostat."""
    sjm = atoms.calc.dft.sjm
    sjm.change(max_iters=3, max_reruns=4, max_potential_stepsize=1.5,
               grand_output=False, always_adjust=True, mixer=0.5,
               slope_regression_depth=3, slope=-20.0)
    pot = sjm.potentiostat
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
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        convergence={'eigenstates': 1e-5},
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], excess_electrons=0.0, tol=0.001)])
    atoms.get_potential_energy()
    sjm = atoms.calc.dft.sjm

    sjm.change(tol=0.05)  # constant-charge mode
    assert sjm.potentiostat.tol == pytest.approx(0.05)

    sjm.change(target_potential=3.0)
    assert sjm.potentiostat.optimizer is not None
    assert sjm.potentiostat.tol == pytest.approx(0.05)

    # A single combined call is the same, whichever branch runs first:
    sjm.change(target_potential=None)
    sjm.change(tol=0.02, target_potential=4.0)
    assert sjm.potentiostat.tol == pytest.approx(0.02)
