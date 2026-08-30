"""Changing SJM parameters without discarding the wave functions."""
import pytest
from ase import Atoms
from ase.units import Ha

from gpaw import GPAW, FermiDirac
from gpaw.new.sjm import SJM
from gpaw.test.solvation.test_modes import NoCavity, Vacuum

TOL = 0.005  # V


@pytest.fixture
def atoms(in_tmp_dir):
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
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
        atoms.calc.dft.sjm.set(excess_electron=-0.01)
