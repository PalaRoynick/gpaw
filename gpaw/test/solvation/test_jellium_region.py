"""Geometry of the SJM jellium counter-charge region."""
import numpy as np
import pytest
from ase.units import Bohr

from gpaw.core import UGDesc
from gpaw.extensions.electrochemistry.potentiostat import (
    CavityShapedRegion, JelliumCharging,
    SlabRegion, build_jellium_region)

HEIGHT = 20.0  # Å
ZMAX = 6.0  # Å, highest atom
# UGDesc works in atomic units:
grid = UGDesc.from_cell_and_grid_spacing(
    cell=np.array([4.0, 4.0, HEIGHT]) / Bohr, grid_spacing=0.2, pbc=(1, 1, 0))
relpos_ac = np.array([[0.5, 0.5, ZMAX / HEIGHT]])


def limits(**kwargs):
    return SlabRegion(**kwargs).limits(HEIGHT, ZMAX)


def test_defaults():
    # 3 Å above the highest atom, 1 Å below the top of the cell:
    assert limits() == (ZMAX + 3.0, HEIGHT - 1.0)


def test_absolute():
    assert limits(bottom=8.0, top=15.0) == (8.0, 15.0)


def test_relative():
    # Negative top: below the upper cell boundary.
    # Negative bottom: above the highest atom.
    assert limits(bottom=-2.5, top=-4.0) == (ZMAX + 2.5, HEIGHT - 4.0)


def test_only_top():
    assert limits(top=15.0) == (ZMAX + 3.0, 15.0)


def test_only_bottom():
    assert limits(bottom=8.0) == (8.0, HEIGHT - 1.0)


def test_thickness():
    assert limits(thickness=4.0) == (ZMAX + 3.0, ZMAX + 7.0)
    assert limits(bottom=8.0, thickness=4.0) == (8.0, 12.0)
    assert limits(top=15.0, thickness=4.0) == (11.0, 15.0)


def test_follows_atoms():
    # Only a relative lower limit makes the region track the atoms:
    assert SlabRegion().follows_atoms
    assert SlabRegion(bottom=-3.0).follows_atoms
    assert not SlabRegion(bottom=8.0).follows_atoms
    assert not SlabRegion(bottom=-3.0, fix_bottom=True).follows_atoms
    assert not SlabRegion(top=15.0, thickness=4.0).follows_atoms


def test_fix_bottom():
    region = SlabRegion(bottom=-3.0, fix_bottom=True)
    assert region.limits(HEIGHT, ZMAX) == (9.0, HEIGHT - 1.0)
    # The atoms move up, but the frozen lower limit stays put:
    assert region.limits(HEIGHT, ZMAX + 1.0) == (9.0, HEIGHT - 1.0)
    # ... whereas without fix_bottom it follows:
    region = SlabRegion(bottom=-3.0)
    assert region.limits(HEIGHT, ZMAX)[0] == 9.0
    assert region.limits(HEIGHT, ZMAX + 1.0)[0] == 10.0


@pytest.mark.parametrize('kwargs, match', [
    ({'top': 1.0, 'bottom': 2.0, 'thickness': 3.0}, 'overspecified'),
    ({'bottom': 'sideways'}, 'Unknown lower limit'),
    ({'top': HEIGHT + 1.0}, 'outside of the unit cell'),
    ({'top': 8.0, 'bottom': 12.0}, 'not below the top'),
    ({'top': 8.0, 'bottom': 8.0}, 'not below the top'),
])
def test_bad_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        limits(**kwargs)


def test_build_jellium_region():
    """The jelliumregion input is mapped onto a region object."""
    # The default and a plain dict give a slab:
    assert isinstance(build_jellium_region(None), SlabRegion)
    region = build_jellium_region({'top': 15.0, 'thickness': 4.0})
    assert isinstance(region, SlabRegion)
    assert region.limits(HEIGHT, ZMAX) == (11.0, 15.0)

    # 'shape' selects the kind; 'cavity_like' is the legacy spelling:
    assert isinstance(build_jellium_region({'shape': 'cavity', 'top': -1.0}),
                      CavityShapedRegion)
    assert isinstance(build_jellium_region({'bottom': 'cavity_like'}),
                      CavityShapedRegion)

    # An already-built region passes straight through:
    slab = SlabRegion(bottom=8.0)
    assert build_jellium_region(slab) is slab

    # todict() gives the input that builds the region again:
    for spec in [{'top': 15.0, 'thickness': 4.0}, {'shape': 'cavity'}]:
        region = build_jellium_region(spec)
        again = build_jellium_region(region.todict())
        assert type(again) is type(region)
        assert again.todict() == region.todict()

    with pytest.raises(ValueError, match='Unknown jellium region shape'):
        build_jellium_region({'shape': 'blob'})

    # The cavity-shaped region is only a stub in this version:
    with pytest.raises(NotImplementedError, match='cavity-like'):
        build_jellium_region({'shape': 'cavity'}).mask(grid, relpos_ac)


def test_move_skips_an_unchanged_region():
    """A move that leaves the region alone does not rebuild the mask."""
    # Two atoms; only the upper one sets the lower limit of the region:
    relpos_ac = np.array([[0.5, 0.5, ZMAX / HEIGHT], [0.5, 0.5, 0.1]])
    # A relative bottom, so the region follows the atoms:
    charging = JelliumCharging(jelliumregion={'bottom': -3.0})
    z1, z2 = charging.initialize(grid, None, relpos_ac, cavity=None)

    # No key stored yet, so the first move rebuilds:
    assert charging.move(relpos_ac, grid) == (z1, z2)
    mask_r = charging.mask_r

    # The lower atom moves and the highest one does not, so the region is
    # unchanged and the mask is left alone -- the same object, not a copy:
    moved_ac = relpos_ac.copy()
    moved_ac[1, 2] = 0.2
    assert charging.move(moved_ac, grid) is None
    assert charging.mask_r is mask_r

    # Moving the highest atom does move the region:
    moved_ac[0, 2] += 1.0 / HEIGHT
    zs = charging.move(moved_ac, grid)
    assert zs is not None
    assert zs[0] == pytest.approx(z1 + 1.0)
    assert charging.mask_r is not mask_r


def test_move_always_rebuilds_without_a_key():
    """A region that reports no key is rebuilt on every move."""

    class KeylessRegion(SlabRegion):
        def key(self, grid, relpos_ac):
            return None  # as the JelliumRegion base does

    charging = JelliumCharging(jelliumregion=KeylessRegion(bottom=-3.0))
    charging.initialize(grid, None, relpos_ac, cavity=None)
    assert charging.move(relpos_ac, grid) is not None
    mask_r = charging.mask_r
    assert charging.move(relpos_ac, grid) is not None  # same geometry
    assert charging.mask_r is not mask_r


def test_mask():
    mask_r, bottom, top = SlabRegion(bottom=8.0, top=12.0, width=0.0).mask(
        grid, relpos_ac)
    assert (bottom, top) == (8.0, 12.0)
    # The mask is 1 strictly inside the region and 0 outside:
    z_r = grid.xyz()[:, :, :, 2] * Bohr
    assert mask_r.data[(z_r > bottom) & (z_r < top)] == pytest.approx(1.0)
    assert mask_r.data[(z_r < bottom) | (z_r > top)] == pytest.approx(0.0)
