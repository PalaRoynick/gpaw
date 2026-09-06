"""Geometry of the SJM jellium counter-charge region."""
import numpy as np
import pytest
from ase.units import Bohr

from gpaw.core import UGDesc
from gpaw.extensions.electrocatalysis.potentiostat import (
    DEFAULT_BOTTOM, DEFAULT_TOP, JelliumRegion)

HEIGHT = 20.0  # Å
ZMAX = 6.0  # Å, highest atom


def limits(**kwargs):
    return JelliumRegion(**kwargs).limits(HEIGHT, ZMAX)


def test_defaults():
    # 3 Å above the highest atom, 1 Å below the top of the cell:
    assert limits() == (ZMAX - DEFAULT_BOTTOM, HEIGHT + DEFAULT_TOP)


def test_absolute():
    assert limits(bottom=8.0, top=15.0) == (8.0, 15.0)


def test_relative():
    # Negative top: below the upper cell boundary.
    # Negative bottom: above the highest atom.
    assert limits(bottom=-2.5, top=-4.0) == (ZMAX + 2.5, HEIGHT - 4.0)


def test_only_top():
    assert limits(top=15.0) == (ZMAX - DEFAULT_BOTTOM, 15.0)


def test_only_bottom():
    assert limits(bottom=8.0) == (8.0, HEIGHT + DEFAULT_TOP)


def test_thickness():
    assert limits(thickness=4.0) == (ZMAX - DEFAULT_BOTTOM,
                                     ZMAX - DEFAULT_BOTTOM + 4.0)
    assert limits(bottom=8.0, thickness=4.0) == (8.0, 12.0)
    assert limits(top=15.0, thickness=4.0) == (11.0, 15.0)


def test_follows_atoms():
    # Only a relative lower limit makes the region track the atoms:
    assert JelliumRegion().follows_atoms
    assert JelliumRegion(bottom=-3.0).follows_atoms
    assert not JelliumRegion(bottom=8.0).follows_atoms
    assert not JelliumRegion(bottom=-3.0, fix_bottom=True).follows_atoms
    assert not JelliumRegion(top=15.0, thickness=4.0).follows_atoms


def test_fix_bottom():
    region = JelliumRegion(bottom=-3.0, fix_bottom=True)
    assert region.limits(HEIGHT, ZMAX) == (9.0, HEIGHT + DEFAULT_TOP)
    # The atoms move up, but the frozen lower limit stays put:
    assert region.limits(HEIGHT, ZMAX + 1.0) == (9.0, HEIGHT + DEFAULT_TOP)
    # ... whereas without fix_bottom it follows:
    region = JelliumRegion(bottom=-3.0)
    assert region.limits(HEIGHT, ZMAX)[0] == 9.0
    assert region.limits(HEIGHT, ZMAX + 1.0)[0] == 10.0


@pytest.mark.parametrize('kwargs, match', [
    ({'top': 1.0, 'bottom': 2.0, 'thickness': 3.0}, 'overspecified'),
    ({'bottom': 'cavity_like'}, 'cavity-like'),
    ({'top': HEIGHT + 1.0}, 'outside of the unit cell'),
    ({'top': 8.0, 'bottom': 12.0}, 'not below the top'),
    ({'top': 8.0, 'bottom': 8.0}, 'not below the top'),
])
def test_bad_input(kwargs, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        limits(**kwargs)


def test_mask():
    # UGDesc works in atomic units:
    grid = UGDesc.from_cell_and_grid_spacing(
        cell=np.array([4.0, 4.0, HEIGHT]) / Bohr,
        grid_spacing=0.2, pbc=(1, 1, 0))
    relpos_ac = np.array([[0.5, 0.5, ZMAX / HEIGHT]])
    mask_r, bottom, top = JelliumRegion(bottom=8.0, top=12.0).mask(
        grid, relpos_ac)
    assert (bottom, top) == (8.0, 12.0)
    # The mask is 1 strictly inside the region and 0 outside:
    z_r = grid.xyz()[:, :, :, 2] * Bohr
    assert mask_r.data[(z_r > bottom) & (z_r < top)] == pytest.approx(1.0)
    assert mask_r.data[(z_r < bottom) | (z_r > top)] == pytest.approx(0.0)


def _extent(sjm):
    """Actual z-extent of the jellium mask, in Angstrom."""
    mask_r = sjm.potentiostat.charging.mask_r.gather(broadcast=True)
    z_z = mask_r.desc.xyz()[0, 0, :, 2] * Bohr
    inside = mask_r.data.mean(axis=(0, 1)) > 0
    return z_z[inside].min(), z_z[inside].max()


@pytest.mark.parametrize('fix_bottom', [False, True])
def test_region_follows_atoms(fix_bottom, in_tmp_dir):
    """A relative lower limit tracks the atoms unless it is fixed."""
    from ase import Atoms

    from gpaw import GPAW, FermiDirac
    from gpaw.extensions.electrocatalysis.sjm import SJM
    from gpaw.test.solvation.test_modes import NoCavity, Vacuum

    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], excess_electrons=-0.01,
                        jelliumregion={'bottom': -3.0,
                                       'fix_bottom': fix_bottom})])
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
