"""Changing SJM parameters without discarding the wave functions."""
import pytest
from ase import Atoms
from ase.units import Ha

from gpaw import GPAW, FermiDirac
from gpaw.new.electrocatalysis.sjm import SJM
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

    sjm.set(target_potential=7.2)
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.2, abs=TOL)
    # Same wave functions - nothing was rebuilt from scratch:
    assert calc.dft.ibzwfs is ibzwfs


def test_switch_to_constant_charge(atoms):
    calc = atoms.calc
    sjm = calc.dft.sjm
    ibzwfs = calc.dft.ibzwfs

    sjm.set(target_potential=None, excess_electrons=-0.01)
    atoms.get_potential_energy()
    assert sjm.charge == -0.01
    assert calc.results['excess_electrons'] == pytest.approx(-0.01)
    assert calc.dft.ibzwfs is ibzwfs

    # ... and back again, reaching the same potential as before:
    sjm.set(target_potential=7.5)
    atoms.get_potential_energy()
    assert sjm.get_electrode_potential() == pytest.approx(7.5, abs=TOL)
    assert calc.dft.ibzwfs is ibzwfs


def test_grand_output_toggle(atoms):
    calc = atoms.calc
    sjm = calc.dft.sjm
    grand = calc.results['energy']

    # Toggling only changes which energy is reported - no new SCF loop:
    niter = calc.dft.scf_loop.niter
    sjm.set(grand_output=False)
    canonical = calc.results['energy']
    assert calc.dft.scf_loop.niter == niter
    assert canonical == pytest.approx(
        calc.dft.energies.total_extrapolated * Ha)
    assert grand == pytest.approx(
        canonical - sjm.charge * calc.get_fermi_level())

    # ... and it is reversible:
    sjm.set(grand_output=True)
    assert calc.results['energy'] == pytest.approx(grand)


def test_unknown_keyword(atoms):
    with pytest.raises(ValueError, match='Unexpected SJM keyword'):
        atoms.calc.sjm.set(excess_electron=-0.01)


def test_set_before_first_calculation(in_tmp_dir):
    """calc.sjm reaches the parameters before there is a calculation."""
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], excess_electrons=0.0)])

    # No calculation yet, so this only stores the parameter:
    atoms.calc.sjm.set(excess_electrons=-0.01)
    with pytest.raises(AttributeError, match='No calculation'):
        atoms.calc.dft

    atoms.get_potential_energy()
    assert atoms.calc.dft.sjm.charge == -0.01
    assert atoms.calc.results['excess_electrons'] == pytest.approx(-0.01)

    # Once built, calc.sjm is the extension itself:
    assert atoms.calc.sjm is atoms.calc.dft.sjm

    # ... and setting through it still keeps the wave functions:
    ibzwfs = atoms.calc.dft.ibzwfs
    atoms.calc.sjm.set(excess_electrons=-0.02)
    atoms.get_potential_energy()
    assert atoms.calc.dft.sjm.charge == -0.02
    assert atoms.calc.dft.ibzwfs is ibzwfs


def test_unknown_extension_name(atoms):
    with pytest.raises(AttributeError, match='no attribute'):
        atoms.calc.nosuchextension


def test_tol_set_at_constant_charge_survives_the_switch(in_tmp_dir):
    """A tol changed while at constant charge takes effect on the switch.

    The 'tol' handler skips the potentiostat in constant-charge mode (there
    is nothing to equilibrate), so _switch_mode has to carry the new value
    across -- it used to be baked in when the whole potential-control object
    was rebuilt.
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

    sjm.set(tol=0.05)  # constant-charge mode: kept on the extension only
    assert sjm.potentiostat.tol == pytest.approx(0.001)

    sjm.set(target_potential=3.0)
    assert sjm.potentiostat.optimizer is not None
    assert sjm.potentiostat.tol == pytest.approx(0.05)

    # A single combined call is the same: the 'tol' branch runs first, while
    # the mode has not flipped yet, so _switch_mode is what has to adopt it.
    sjm.set(target_potential=None)
    sjm.set(tol=0.02, target_potential=4.0)
    assert sjm.potentiostat.tol == pytest.approx(0.02)
