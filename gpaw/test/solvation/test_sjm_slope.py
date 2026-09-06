"""Slope estimate and charge stepping behind the SJM potential control."""
import pytest
from ase.units import Ha, _e

from gpaw.extensions.electrocatalysis.potentiostat import (
    DirectOptimizer, Potentiostat, regress_slope)

WIDTH = 4.0  # Å
AREA = WIDTH**2  # Å^2
TARGET = 4.5  # V, as a work function


def make_optimizer(**kwargs) -> DirectOptimizer:
    """A direct potential optimizer - no DFT, no grid involved."""
    return DirectOptimizer(**kwargs)


def line(slope: float, charge: float, potential0: float = 5.0) -> float:
    """Potential, in V, of a system whose potential is linear in the charge."""
    return potential0 + slope * charge


def test_capacitance_guess():
    """With nothing to regress from, the slope is a plate capacitor's."""
    opt = make_optimizer()
    assert opt.slope is None
    slope = opt.update_slope(AREA)
    # Å^2 -> cm^2 is 1e16 and F -> muF is 1e6; 10 muF/cm^2 is the guessed
    # apparent capacitance in DirectOptimizer._capacitance_slope:
    assert slope == pytest.approx(-_e * 1e22 / (AREA * 10))
    assert slope < 0.0  # more electrons is a lower potential


def test_given_slope_is_kept():
    """A slope from the user survives until there is something to fit."""
    opt = make_optimizer(slope=-42.0)
    assert opt.update_slope(AREA) == pytest.approx(-42.0)


def test_regression_is_exact():
    """Without damping, a linear system gives its slope back exactly."""
    opt = make_optimizer(mixer=0.0)
    for charge in [0.0, -0.1, -0.2]:
        opt.history.append((charge, line(-30.0, charge)))
    assert opt.update_slope(AREA) == pytest.approx(-30.0)
    assert regress_slope(opt.history) == pytest.approx(-30.0)


def test_mixing():
    """The new estimate is damped with the old one."""
    opt = make_optimizer(slope=-10.0, mixer=0.25)
    for charge in [0.0, -0.1]:
        opt.history.append((charge, line(-30.0, charge)))
    assert opt.update_slope(AREA) == pytest.approx(
        0.25 * -10.0 + 0.75 * -30.0)


def test_regression_depth():
    """Only the last few attempts are regressed over."""
    opt = make_optimizer(mixer=0.0, slope_regression_depth=2)
    assert opt.slope_regression_depth == 2
    # An old segment with a different slope than the last two attempts:
    opt.history.append((0.2, line(-90.0, 0.2)))
    opt.history.append((0.1, line(-90.0, 0.1)))
    for charge in [0.0, -0.1]:
        opt.history.append((charge, line(-30.0, charge)))
    assert len(opt.history) == 2
    assert opt.update_slope(AREA) == pytest.approx(-30.0)

    # Widening the window keeps the attempts that are still in it; 4 is
    # DirectOptimizer's own default slope_regression_depth:
    opt.slope_regression_depth = 4
    assert len(opt.history) == 2
    assert opt.slope_regression_depth == 4


def test_bad_regression_depth():
    with pytest.raises(ValueError, match='at least 2'):
        make_optimizer(slope_regression_depth=1)
    with pytest.raises(ValueError, match='at least 2'):
        make_optimizer().slope_regression_depth = 0


def test_next_charge_hits_a_linear_target():
    """One step is enough when the slope is right."""
    opt = make_optimizer(slope=-30.0)
    charge = opt.next_charge(-0.05, line(-30.0, -0.05), TARGET, area=AREA)
    # 5.0 - 30 * charge == TARGET:
    assert charge == pytest.approx((5.0 - TARGET) / 30.0)
    # The attempt was recorded for the next regression:
    assert opt.history[-1] == (-0.05, line(-30.0, -0.05))


def test_trust_region():
    """A step never overshoots the previous one, however bad the slope."""
    opt = make_optimizer(slope=-1e-6, mixer=1.0)  # a hopeless slope
    opt.history.append((0.0, 5.0))
    # The potential is above the target, so electrons are wanted; the slope
    # asks for thousands of them and gets the last step size instead:
    charge = opt.next_charge(-0.1, 5.0, TARGET, area=AREA)
    assert charge == pytest.approx(0.0)


def test_reset():
    opt = make_optimizer(slope=-30.0)
    opt.history.append((0.0, 5.0))
    opt.reset()
    assert opt.slope is None
    assert len(opt.history) == 0


class _FakeIBZWFS:
    """Just enough of an IBZWaveFunctions for the potentiostat to drive."""

    def __init__(self, potential: float):
        # WorkFunctionProbe reads the potential as -fermi_level * Ha:
        self.fermi_level = -potential / Ha
        self.occs_calls: list[float] = []

    def calculate_occs(self, occ_calc, nelectrons: float) -> None:
        self.occs_calls.append(nelectrons)


class _ExplodingIBZWFS:
    @property
    def fermi_level(self) -> float:
        raise AssertionError('the potential must not be probed here')


class _StubCharging:
    def __init__(self, charge: float = 0.0):
        self.charge = charge
        self.region_area = AREA


class _RecordingMixer:
    def __init__(self):
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _silent(*args, **kwargs) -> None:
    pass


def test_potentiostat_within_tolerance_is_converged():
    charging = _StubCharging(charge=-0.2)
    pot = Potentiostat(charging=charging, target_potential=TARGET,
                       optimizer=DirectOptimizer(slope=-30.0), tol=0.01)
    ibzwfs = _FakeIBZWFS(TARGET + 0.005)  # inside the tolerance
    mixer = _RecordingMixer()
    assert pot.equilibrate(ibzwfs, 10.0, object(), mixer, _silent)
    assert ibzwfs.occs_calls == []
    assert mixer.resets == 0
    assert charging.charge == -0.2


def test_potentiostat_steps_the_charge_along_the_slope():
    charging = _StubCharging(charge=-0.2)
    pot = Potentiostat(charging=charging, target_potential=TARGET,
                       optimizer=DirectOptimizer(slope=-30.0), tol=0.001)
    potential = TARGET + 0.5  # 0.5 V above the target
    ibzwfs = _FakeIBZWFS(potential)
    mixer = _RecordingMixer()
    converged = pot.equilibrate(ibzwfs, 10.0, object(), mixer, _silent)
    assert not converged
    dc = (TARGET - potential) / -30.0
    assert charging.charge == pytest.approx(-0.2 + dc)
    assert ibzwfs.occs_calls == [pytest.approx(10.0 + dc)]
    assert mixer.resets == 1


def test_potentiostat_constant_charge_is_a_noop():
    charging = _StubCharging(charge=-0.2)
    pot = Potentiostat(charging=charging)  # no target, no optimizer
    assert pot.equilibrate(_ExplodingIBZWFS(), 10.0, object(),
                           _RecordingMixer(), _silent)
    assert charging.charge == -0.2
