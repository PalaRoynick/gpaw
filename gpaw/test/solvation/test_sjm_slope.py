"""Slope estimate and charge stepping behind the SJM potential control."""
import pytest
from ase.units import Ha, _e

from gpaw.extensions.electrochemistry.potentiostat import (
    DirectOptimizer, EquilibrationResult, PotentialConvergenceError,
    Potentiostat, regress_slope)

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


def test_wrong_signed_regression_falls_back_to_capacitance():
    """A wrong-signed regression is rejected for the capacitance guess."""
    opt = make_optimizer(mixer=0.0)
    for charge in [0.0, -0.1, -0.2]:
        opt.history.append((charge, line(30.0, charge)))  # wrong sign
    slope = opt.update_slope(AREA)
    assert slope == pytest.approx(-_e * 1e22 / (AREA * 10))
    assert slope < 0.0


def test_next_step_falls_back_when_the_regression_is_wrong_signed():
    """The charge steps the right way even when the fit points the
    wrong way."""
    opt = make_optimizer(mixer=0.0)
    # Two points that regress to a positive (wrong-signed) slope of +50:
    opt.history.append((0.0, line(50.0, 0.0)))
    opt.history.append((-0.1, line(50.0, -0.1)))
    potential = line(50.0, -0.05)  # on the same (wrong-signed) line
    charge = opt.next_excess_electrons(-0.05, potential, TARGET, area=AREA)
    # potential (2.5 V) is below target (4.5 V), so fewer electrons are
    # wanted: charge should decrease, not increase the way the (rejected)
    # positive slope would incorrectly ask for:
    assert charge < -0.05


def test_correctly_signed_regression_is_not_rejected():
    """The guard does not disturb an ordinary, correctly-signed regression."""
    opt = make_optimizer(mixer=0.0)
    for charge in [0.0, -0.1, -0.2]:
        opt.history.append((charge, line(-30.0, charge)))
    assert opt.update_slope(AREA) == pytest.approx(-30.0)


def test_zero_slope_regression_falls_back_to_capacitance():
    """A flat potential-vs-charge history is as unusable as a wrong sign."""
    opt = make_optimizer(mixer=0.0)
    for charge in [0.0, -0.1, -0.2]:
        opt.history.append((charge, 5.0))  # potential does not move
    assert opt.update_slope(AREA) == pytest.approx(-_e * 1e22 / (AREA * 10))


def test_a_badly_wrong_signed_regression_flips_the_damped_average():
    """A large wrong-signed regression can flip even a damped average."""
    opt = make_optimizer(slope=-1.0, mixer=0.2)  # mostly trusts the new value
    for charge in [0.0, -0.1]:
        opt.history.append((charge, line(50.0, charge)))
    assert opt.update_slope(AREA) == pytest.approx(-_e * 1e22 / (AREA * 10))


def test_a_mildly_wrong_signed_regression_can_be_damped_away():
    """Heavy damping rescues a mildly wrong-signed regression, without
    overreacting."""
    # Heavy trust in the old value:
    opt = make_optimizer(slope=-30.0, mixer=0.8)
    for charge in [0.0, -0.1]:
        opt.history.append((charge, line(2.0, charge)))  # mildly wrong-signed
    slope = opt.update_slope(AREA)
    assert slope == pytest.approx(0.8 * -30.0 + 0.2 * 2.0)
    assert slope < 0.0


def test_bad_regression_depth():
    with pytest.raises(ValueError, match='at least 2'):
        make_optimizer(slope_regression_depth=1)
    with pytest.raises(ValueError, match='at least 2'):
        make_optimizer().slope_regression_depth = 0


def test_next_step_hits_a_linear_target():
    """One step is enough when the slope is right."""
    opt = make_optimizer(slope=-30.0)
    charge = opt.next_excess_electrons(-0.05, line(-30.0, -0.05), TARGET,
                                       area=AREA)
    # 5.0 - 30 * charge == TARGET:
    assert charge == pytest.approx((5.0 - TARGET) / 30.0)
    # The attempt was recorded for the next regression:
    assert opt.history[-1] == (-0.05, line(-30.0, -0.05))


def test_trust_region():
    """A fully damped slope never steps beyond the previous step."""
    opt = make_optimizer(slope=-1e-6, mixer=1.0)  # a hopeless slope
    opt.history.append((0.0, 5.0))
    # The potential is above the target, so electrons are wanted; the slope
    # asks for thousands of them and gets the last step size instead:
    charge = opt.next_excess_electrons(-0.1, 5.0, TARGET, area=AREA)
    assert charge == pytest.approx(0.0)


def _step_from(mixer: float) -> float:
    """Size of one step on a line of slope -30, after a step of 0.01.

    The seeded attempt and the one made here both sit on that line, so the
    regression returns -30 whatever the mixer does with it: only the trust
    region distinguishes the cases below.
    """
    opt = make_optimizer(slope=-30.0, mixer=mixer)
    opt.history.append((0.0, line(-30.0, 0.0)))
    return 0.01 + opt.next_excess_electrons(-0.01, line(-30.0, -0.01),
                                            TARGET, area=AREA)


def test_trust_region_lets_a_step_grow():
    """A step may exceed the previous one by 1 - mixer of the excess."""
    # The slope asks for (TARGET - 5.3) / -30 = 0.026667 after a 0.01 step,
    # and with mixer=0.2 keeps 0.01 + 0.8 * (0.026667 - 0.01):
    wanted = (TARGET - line(-30.0, -0.01)) / -30.0
    assert _step_from(0.2) == pytest.approx(0.01 + 0.8 * (wanted - 0.01))
    # Which is bigger than the previous step -- the point of the change --
    # but still short of the whole step the slope points at:
    assert 0.01 < _step_from(0.2) < wanted


def test_trust_region_endpoints():
    """mixer=1 never grows; mixer=0 takes the whole step."""
    wanted = (TARGET - line(-30.0, -0.01)) / -30.0
    assert _step_from(1.0) == pytest.approx(0.01)
    assert _step_from(0.0) == pytest.approx(wanted)


def test_a_tiny_first_step_does_not_cap_the_rest():
    """Steps recover from a small one instead of being capped by it.

    The old trust region clamped every step to the previous one, so the
    first step became a ceiling for all of them and equilibration crawled
    whenever it happened to be small.
    """
    tiny = 1e-4  # the step that got us to the current charge
    opt = make_optimizer(slope=-30.0, mixer=0.2)
    opt.history.append((0.0, line(-30.0, 0.0)))
    charge = tiny
    steps = []
    for _ in range(6):
        previous = charge
        charge = opt.next_excess_electrons(charge, line(-30.0, charge),
                                           TARGET, area=AREA)
        steps.append(abs(charge - previous))
    # The old clamp would have pinned every step to `tiny`, needing ~170 of
    # them to cross the 0.0167 electrons to the target.  Now the very first
    # step is already most of the way there:
    assert steps[0] > 100 * tiny
    assert line(-30.0, charge) == pytest.approx(TARGET, abs=1e-6)


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
    def __init__(self, excess_electrons: float = 0.0):
        self.excess_electrons = excess_electrons
        self.region_area = AREA


class _RecordingMixer:
    def __init__(self):
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _silent(*args, **kwargs) -> None:
    pass


def test_potentiostat_within_tolerance_is_converged():
    charging = _StubCharging(excess_electrons=-0.2)
    pot = Potentiostat(charging=charging, target_potential=TARGET,
                       optimizer=DirectOptimizer(slope=-30.0), tol=0.01)
    ibzwfs = _FakeIBZWFS(TARGET + 0.005)  # inside the tolerance
    mixer = _RecordingMixer()
    assert (pot.equilibrate(ibzwfs, 10.0, object(), mixer, _silent)
            is EquilibrationResult.converged)
    assert ibzwfs.occs_calls == []
    assert mixer.resets == 0
    assert charging.excess_electrons == -0.2


def test_potentiostat_steps_the_charge_along_the_slope():
    charging = _StubCharging(excess_electrons=-0.2)
    pot = Potentiostat(charging=charging, target_potential=TARGET,
                       optimizer=DirectOptimizer(slope=-30.0), tol=0.001)
    potential = TARGET + 0.5  # 0.5 V above the target
    ibzwfs = _FakeIBZWFS(potential)
    mixer = _RecordingMixer()
    result = pot.equilibrate(ibzwfs, 10.0, object(), mixer, _silent)
    assert result is EquilibrationResult.stepped
    dc = (TARGET - potential) / -30.0
    assert charging.excess_electrons == pytest.approx(-0.2 + dc)
    assert ibzwfs.occs_calls == [pytest.approx(10.0 + dc)]
    assert mixer.resets == 1


def test_potentiostat_constant_charge_is_a_noop():
    charging = _StubCharging(excess_electrons=-0.2)
    pot = Potentiostat(charging=charging)  # no target, no optimizer
    assert (pot.equilibrate(_ExplodingIBZWFS(), 10.0, object(),
                            _RecordingMixer(), _silent)
            is EquilibrationResult.converged)
    assert charging.excess_electrons == -0.2


def test_potentiostat_default_max_potential_stepsize():
    """The V-space step clamp is on by default, at the legacy 2 V."""
    pot = Potentiostat(charging=_StubCharging())
    assert pot.max_potential_stepsize == 2


def _stepping_potentiostat(charge, *, history, max_potential_stepsize=2):
    """A potentiostat mid-equilibration: one accepted attempt on record."""
    optimizer = DirectOptimizer(slope=-30.0)
    for attempt in history:
        optimizer.history.append(attempt)
    return Potentiostat(charging=_StubCharging(excess_electrons=charge),
                        target_potential=TARGET, optimizer=optimizer,
                        tol=0.001,
                        max_potential_stepsize=max_potential_stepsize)


def test_equilibrate_rejects_a_step_that_left_the_linear_region():
    """A step over max_potential_stepsize, further from target, is halved."""
    # Last accepted attempt: 0.0 electrons at 5.0 V.  The step that followed
    # landed at 2.0 electrons and 40 V -- 35 V away, and further from the
    # 4.5 V target than 5.0 V was.
    pot = _stepping_potentiostat(2.0, history=[(0.0, 5.0)])
    ibzwfs = _FakeIBZWFS(40.0)
    mixer = _RecordingMixer()
    result = pot.equilibrate(ibzwfs, 10.0, object(), mixer, _silent)
    assert result is EquilibrationResult.rejected
    # Charge halved back toward the last accepted one:
    assert pot.excess_electrons == pytest.approx(1.0)
    assert ibzwfs.occs_calls == [pytest.approx(10.0 + (1.0 - 2.0))]
    assert mixer.resets == 1
    # The rejected point is not fed to the slope regression:
    assert list(pot.optimizer.history) == [(0.0, 5.0)]


def test_equilibrate_does_not_reject_a_big_step_toward_the_target():
    """Overshooting the *size* limit is fine as long as it closes the gap."""
    pot = _stepping_potentiostat(2.0, history=[(0.0, 40.0)])
    ibzwfs = _FakeIBZWFS(6.0)  # moved 34 V, but toward the 4.5 V target
    result = pot.equilibrate(ibzwfs, 10.0, object(), _RecordingMixer(),
                             _silent)
    assert result is EquilibrationResult.stepped
    assert list(pot.optimizer.history)[-1] == (2.0, 6.0)  # recorded normally


def test_equilibrate_max_potential_stepsize_none_disables_the_clamp():
    pot = _stepping_potentiostat(2.0, history=[(0.0, 5.0)],
                                 max_potential_stepsize=None)
    result = pot.equilibrate(_FakeIBZWFS(40.0), 10.0, object(),
                             _RecordingMixer(), _silent)
    assert result is EquilibrationResult.stepped


def test_equilibrate_never_rejects_the_first_attempt():
    """With no accepted attempt to measure against, the first step is taken
    however far off the potential is."""
    pot = _stepping_potentiostat(0.0, history=[])
    result = pot.equilibrate(_FakeIBZWFS(40.0), 10.0, object(),
                             _RecordingMixer(), _silent)
    assert result is EquilibrationResult.stepped


class _ScriptedPotentiostat(Potentiostat):
    """A potentiostat whose attempts replay a fixed list of results."""

    def __init__(self, results, **kwargs):
        super().__init__(charging=_StubCharging(), target_potential=TARGET,
                         optimizer=DirectOptimizer(), tol=0.001, **kwargs)
        self._results = list(results)

    def equilibrate(self, *args, **kwargs) -> EquilibrationResult:
        return self._results.pop(0)


def _post_scf(pot: Potentiostat) -> bool:
    return pot.post_scf_convergence(_FakeIBZWFS(5.0), 10.0, object(),
                                    _RecordingMixer(), _silent)


def test_post_scf_a_rejected_step_does_not_count_as_an_attempt():
    R = EquilibrationResult
    pot = _ScriptedPotentiostat(
        [R.rejected, R.rejected, R.stepped, R.converged], max_iters=5)
    assert _post_scf(pot) is False
    assert (pot.niters, pot.reruns) == (0, 1)
    assert _post_scf(pot) is False
    assert (pot.niters, pot.reruns) == (0, 2)
    # An accepted step: the attempt counter advances and the streak clears.
    assert _post_scf(pot) is False
    assert (pot.niters, pot.reruns) == (1, 0)
    assert _post_scf(pot) is True
    assert (pot.niters, pot.reruns) == (2, 0)


def test_post_scf_raises_once_the_halving_runs_out():
    pot = _ScriptedPotentiostat([EquilibrationResult.rejected] * 5,
                                max_iters=10, max_reruns=3)
    for _ in range(3):
        assert _post_scf(pot) is False
    assert pot.reruns == 3
    with pytest.raises(PotentialConvergenceError, match='halvings'):
        _post_scf(pot)


def test_post_scf_raises_once_the_attempts_run_out():
    pot = _ScriptedPotentiostat([EquilibrationResult.stepped] * 3,
                                max_iters=2)
    assert _post_scf(pot) is False
    with pytest.raises(PotentialConvergenceError, match='2 attempts'):
        _post_scf(pot)


def test_always_adjust_stages_a_charge_for_the_next_geometry():
    pot = _ScriptedPotentiostat([EquilibrationResult.converged],
                                always_adjust=True)
    assert _post_scf(pot) is True
    assert pot.pending_excess_electrons is not None
    staged = pot.pending_excess_electrons
    assert pot.new_geometry() == staged
    assert pot.excess_electrons == staged
    assert (pot.pending_excess_electrons, pot.niters) == (None, 0)
