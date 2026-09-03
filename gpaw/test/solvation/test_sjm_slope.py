"""Slope estimate behind the SJM potential equilibration."""
import numpy as np
import pytest
from ase.units import Bohr, Ha, _e

from gpaw.core import UGDesc
from gpaw.extensions.jellium import (FixedPotentialJelliumExtension,
                                     regress_slope)

WIDTH = 4.0  # Å
AREA = WIDTH**2  # Å^2
TARGET = 4.5  # V, as a work function


def make_jellium(**kwargs) -> FixedPotentialJelliumExtension:
    """A fixed-potential jellium on a bare grid - no DFT involved."""
    grid = UGDesc.from_cell_and_grid_spacing(
        cell=np.array([WIDTH, WIDTH, 20.0]) / Bohr,
        grid_spacing=0.4, pbc=(1, 1, 0))
    mask_r = grid.zeros()
    mask_r.data[:, :, -10:] = 1.0
    kwargs.setdefault('workfunction_target', TARGET)
    return FixedPotentialJelliumExtension(mask_r, pw=grid, **kwargs)


def fermi_level(potential: float) -> float:
    """Fermi level, in Hartree, of an electrode at this potential."""
    return -potential / Ha


def line(slope: float, charge: float, potential0: float = 5.0) -> float:
    """Fermi level of a system whose potential is linear in the charge."""
    return fermi_level(potential0 + slope * charge)


def test_capacitance_guess():
    """With nothing to regress from, the slope is a plate capacitor's."""
    jellium = make_jellium()
    assert jellium.slope is None
    slope = jellium.update_slope()
    # Å^2 -> cm^2 is 1e16 and F -> muF is 1e6; 10 muF/cm^2 is the guessed
    # apparent capacitance in
    # FixedPotentialJelliumExtension._capacitance_slope:
    assert slope == pytest.approx(-_e * 1e22 / (AREA * 10))
    assert slope < 0.0  # more electrons is a lower potential


def test_given_slope_is_kept():
    """A slope from the user survives until there is something to fit."""
    jellium = make_jellium(slope=-42.0)
    assert jellium.update_slope() == pytest.approx(-42.0)


def test_regression_is_exact():
    """Without damping, a linear system gives its slope back exactly."""
    jellium = make_jellium(mixer=0.0)
    for charge in [0.0, -0.1, -0.2]:
        jellium.history.append((charge, line(-30.0, charge)))
    assert jellium.update_slope() == pytest.approx(-30.0)
    assert regress_slope(jellium.history) == pytest.approx(-30.0)


def test_mixing():
    """The new estimate is damped with the old one."""
    jellium = make_jellium(slope=-10.0, mixer=0.25)
    for charge in [0.0, -0.1]:
        jellium.history.append((charge, line(-30.0, charge)))
    assert jellium.update_slope() == pytest.approx(
        0.25 * -10.0 + 0.75 * -30.0)


def test_regression_depth():
    """Only the last few attempts are regressed over."""
    jellium = make_jellium(mixer=0.0, slope_regression_depth=2)
    assert jellium.slope_regression_depth == 2
    # An old segment with a different slope than the last two attempts:
    jellium.history.append((0.2, line(-90.0, 0.2)))
    jellium.history.append((0.1, line(-90.0, 0.1)))
    for charge in [0.0, -0.1]:
        jellium.history.append((charge, line(-30.0, charge)))
    assert len(jellium.history) == 2
    assert jellium.update_slope() == pytest.approx(-30.0)

    # Widening the window keeps the attempts that are still in it; 4 is
    # FixedPotentialJelliumExtension's own default slope_regression_depth:
    jellium.slope_regression_depth = 4
    assert len(jellium.history) == 2
    assert jellium.slope_regression_depth == 4


def test_bad_regression_depth():
    with pytest.raises(ValueError, match='at least 2'):
        make_jellium(slope_regression_depth=1)
    with pytest.raises(ValueError, match='at least 2'):
        make_jellium().slope_regression_depth = 0


def test_next_charge_hits_a_linear_target():
    """One step is enough when the slope is right."""
    jellium = make_jellium(slope=-30.0, excess_electrons_guess=-0.05)
    charge = jellium.next_charge(line(-30.0, -0.05))
    # 5.0 - 30 * charge == TARGET:
    assert charge == pytest.approx((5.0 - TARGET) / 30.0)
    # The attempt was recorded for the next regression:
    assert jellium.history[-1] == (-0.05, line(-30.0, -0.05))


def test_trust_region():
    """A step never overshoots the previous one, however bad the slope."""
    jellium = make_jellium(slope=-1e-6, mixer=1.0)  # a hopeless slope
    jellium.history.append((0.0, fermi_level(5.0)))
    jellium.charge = -0.1
    # The potential is above the target, so electrons are wanted; the slope
    # asks for thousands of them and gets the last step size instead:
    charge = jellium.next_charge(fermi_level(5.0))
    assert charge == pytest.approx(0.0)


def test_reset_slope():
    jellium = make_jellium(slope=-30.0)
    jellium.history.append((0.0, fermi_level(5.0)))
    jellium.reset_slope()
    assert jellium.slope is None
    assert len(jellium.history) == 0
