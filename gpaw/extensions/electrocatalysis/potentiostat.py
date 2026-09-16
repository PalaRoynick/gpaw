from __future__ import annotations

import enum
import textwrap
from collections import deque

import numpy as np
from ase.units import Bohr, Ha, _e

from gpaw import ConvergenceError
from gpaw.core import PWArray, UGArray
from gpaw.extensions.jellium import JelliumExtension

#: Default top of the jellium region: 1 Å below the top of the unit cell.
DEFAULT_TOP = -1.0
#: Default bottom of the jellium region: 3 Å above the highest atom.
DEFAULT_BOTTOM = -3.0


class JelliumRegion:
    """A region for the jellium counter-charge to fill.

    Turns the current geometry into a mask on the fine grid.  Subclass this
    to define the region a different way: :class:`SlabRegion` (a slab
    perpendicular to z) is the default, and a cavity-shaped one -- where
    the counter-charge follows the implicit solvent -- is planned.
    :func:`build_jellium_region` maps the user's ``jelliumregion`` input
    onto one of these.
    """

    @property
    def follows_atoms(self) -> bool:
        """Does :meth:`mask` have to be rebuilt when the atoms move?"""
        return False

    def mask(self, grid, relpos_ac) -> tuple[UGArray, float, float]:
        """Mask of the region on the fine grid, plus its z limits.

        Returns the un-normalized (0/1 or weighted) mask and the lower and
        upper boundary of the region in Å, the latter only for logging.
        """
        raise NotImplementedError

    def key(self, grid, relpos_ac) -> object | None:
        """Cheap value identifying the region for this geometry.

        :meth:`JelliumCharging.move` skips rebuilding the mask when the
        key has not changed since the last one, so an implementation must
        be much cheaper than :meth:`mask` and must change whenever the
        mask would.  ``None`` -- the default -- means "always rebuild".
        """
        return None


class SlabRegion(JelliumRegion):
    """A jellium region shaped as a slab perpendicular to z.

    This is the region SJM has always used and the default.  It is
    specified by at most two of ``top``, ``bottom`` and ``thickness`` (the
    third one follows).

    Parameters
    ----------
    top:
        Upper boundary of the region.  A positive value is the z coordinate
        in Å; a negative value is the distance below the upper cell
        boundary.  Default: -1 Å.
    bottom:
        Lower boundary of the region.  A positive value is the z coordinate
        in Å; a negative value is the distance above the highest atom, in
        which case the region follows the atoms during a relaxation unless
        ``fix_bottom`` is used.  Default: -3 Å.
    thickness:
        Thickness of the region in Å.  Can only be given together with one
        of ``top`` and ``bottom``.
    fix_bottom:
        Freeze a relative (negative) lower limit at the value it has for
        the first geometry, so that it does not adapt to a change in the
        atomic positions during a relaxation.
    """

    def __init__(self,
                 top: float | None = None,  # Å
                 bottom: float | str | None = None,  # Å
                 thickness: float | None = None,  # Å
                 fix_bottom: bool = False):
        if isinstance(bottom, str):
            # 'cavity_like' is intercepted by build_jellium_region before it
            # gets here; anything else is a mistake.
            raise ValueError(
                f'Unknown lower limit for the jellium region: {bottom!r}.  '
                'Give a number of Angstrom instead.')

        specified = [top is not None,
                     bottom is not None,
                     thickness is not None]
        if all(specified):
            raise ValueError(
                'The jellium region has been overspecified: give at most two '
                "of 'top', 'bottom' and 'thickness'.")
        if not any(specified):
            top, bottom = DEFAULT_TOP, DEFAULT_BOTTOM
        elif specified == [True, False, False]:
            bottom = DEFAULT_BOTTOM
        elif specified == [False, True, False]:
            top = DEFAULT_TOP
        elif specified == [False, False, True]:
            bottom = DEFAULT_BOTTOM

        self.top: float | None = top
        self.bottom: float | None = bottom
        self.thickness: float | None = thickness
        self.fix_bottom = fix_bottom
        self._frozen_bottom: float | None = None

    def __repr__(self):
        return (f'SlabRegion(top={self.top}, bottom={self.bottom}, '
                f'thickness={self.thickness}, fix_bottom={self.fix_bottom})')

    @property
    def follows_atoms(self) -> bool:
        """Does the region move when the atoms move?"""
        return (self.bottom is not None and self.bottom < 0.0 and
                not self.fix_bottom)

    def limits(self,
               height: float,  # Å
               zmax: float,  # Å
               ) -> tuple[float, float]:
        """Lower and upper boundary in Å.

        ``height`` is the height of the unit cell and ``zmax`` the z
        coordinate of the highest atom.
        """
        top, bottom, thickness = self.top, self.bottom, self.thickness

        if top is not None and top < 0.0:
            top = height + top
        if bottom is not None and bottom < 0.0:
            if self._frozen_bottom is not None:
                bottom = self._frozen_bottom
            else:
                bottom = zmax - bottom
                if self.fix_bottom:
                    self._frozen_bottom = bottom

        # __init__ leaves at most one of the three unspecified:
        if top is None:
            assert bottom is not None and thickness is not None
            top = bottom + thickness
        if bottom is None:
            assert thickness is not None
            bottom = top - thickness

        if top > height:
            raise ValueError(
                f'The upper limit of the jellium region ({top:.3f} Å) lies '
                f'outside of the unit cell ({height:.3f} Å).  If you did not '
                'set it manually, increase your unit cell size or translate '
                'the atomic system down along the z axis.')
        if bottom >= top:
            raise ValueError(
                f'Your jellium region has a bottom at {bottom:.3f} Å, which '
                f'is not below the top at {top:.3f} Å.')
        return bottom, top

    def key(self, grid, relpos_ac) -> tuple[float, float]:
        """The z limits -- all that this region's shape depends on.

        Only reached when :attr:`follows_atoms` is true, which rules out
        ``fix_bottom``, so :meth:`limits` cannot freeze anything here.
        """
        height = grid.cell_cv[2, 2] * Bohr
        return self.limits(height, relpos_ac[:, 2].max() * height)

    def mask(self, grid, relpos_ac) -> tuple[UGArray, float, float]:
        """Build the 0/1 mask of the region, plus its limits in Å."""
        height = grid.cell_cv[2, 2] * Bohr
        bottom, top = self.limits(height, relpos_ac[:, 2].max() * height)
        z_r = grid.xyz()[:, :, :, 2]
        mask_r = grid.empty()
        mask_r.data[:] = np.logical_and(z_r > bottom / Bohr,
                                        z_r < top / Bohr)
        return mask_r, bottom, top


class CavityShapedRegion(JelliumRegion):
    """A jellium region that follows the implicit-solvent cavity.

    Everything inside the solvent cavity and below ``top`` is filled,
    weighted by the cavity function ``g_g`` -- the counter-charge then
    hugs the solvated slab instead of sitting in a flat slab above it.
    This mirrors the legacy ``gpaw.solvation.sjm.CavityShapedJellium``;
    it is not wired up in this version of SJM yet.

    Parameters
    ----------
    top:
        Upper cut-off of the region.  A positive value is the z coordinate
        in Å; a negative value is the distance below the upper cell
        boundary.  Default: -1 Å.
    cavity:
        The solvation cavity to read ``g_g`` from.
    """

    def __init__(self, top: float | None = None, *, cavity=None):
        self.top = DEFAULT_TOP if top is None else top
        self.cavity = cavity

    def __repr__(self):
        return f'CavityShapedRegion(top={self.top})'

    @property
    def follows_atoms(self) -> bool:
        return True  # the cavity moves with the atoms

    def mask(self, grid, relpos_ac) -> tuple[UGArray, float, float]:
        raise NotImplementedError(
            'A cavity-like jellium region is not implemented in this '
            'version of SJM.  Use gpaw.solvation.sjm instead.')


def build_jellium_region(spec: dict | JelliumRegion | None,
                         *, cavity=None) -> JelliumRegion:
    """Map the user's ``jelliumregion`` input onto a region object.

    ``spec`` is the ``jelliumregion`` keyword: a dict of parameters, an
    already-built :class:`JelliumRegion`, or ``None`` for the default slab.
    A ``'shape'`` key (``'slab'`` -- the default -- or ``'cavity'``)
    selects the kind; the rest of the dict is passed to it.  The legacy
    ``bottom='cavity_like'`` spelling is still understood.
    """
    if isinstance(spec, JelliumRegion):
        return spec
    spec = dict(spec or {})
    kind = spec.pop('shape', 'slab')
    if spec.get('bottom') == 'cavity_like':  # legacy alias
        kind = 'cavity'
        del spec['bottom']
    if kind == 'slab':
        return SlabRegion(**spec)
    if kind == 'cavity':
        return CavityShapedRegion(**spec, cavity=cavity)
    raise ValueError(
        f'Unknown jellium region shape: {kind!r}.  '
        "Use 'slab' (the default) or 'cavity'.")


class ChargingScheme:
    """Puts an excess charge into the cell.

    It also reports the geometry of the electrode "plate" that the charge
    faces, which the potential optimizer needs for its first-guess slope.
    """

    charge: float

    def add_charge(self, nt_r: UGArray) -> None:
        """Add the counter-charge to a real-space pseudo-density."""
        raise NotImplementedError

    def add_charge_pw(self, nt_g: PWArray | None) -> None:
        """Add the counter-charge to a plane-wave pseudo-density."""
        raise NotImplementedError

    def set_region(self, mask_r: UGArray) -> None:
        """Store (and normalize) the shape of the charged region."""
        raise NotImplementedError

    def move(self, relpos_ac, grid) -> tuple[float, float] | None:
        """Rebuild the region for a new geometry if it follows the atoms.

        Returns the new lower and upper limit in Å, or ``None`` if there
        was nothing to rebuild -- either the region does not move with the
        atoms at all, or this geometry leaves it exactly where it was.
        """
        return None

    @property
    def region_area(self) -> float:
        """Area of the electrode plate in the xy plane, in Å²."""
        raise NotImplementedError


class JelliumCharging(JelliumExtension, ChargingScheme):
    """A uniform jellium counter-charge filling a masked region.

    This is the charging scheme SJM has always used: the excess electrons
    of the atomic system are balanced by a jellium of the opposite sign
    smeared uniformly over the region, which is a :class:`SlabRegion` by
    default but can be any :class:`JelliumRegion`.

    Parameters
    ----------
    mask_r:
        Mask of the region on the fine grid.  It is normalized in place to
        integrate to one.
    pw:
        Plane-wave (or real-space) descriptor of the interpolated density,
        used to cache a reciprocal-space copy of the mask for PW mode.
    charge:
        Number of excess electrons (with the opposite sign in the jellium).
    region:
        The :class:`JelliumRegion` the mask was built from, kept so that
        the region can follow the atoms during a relaxation.  Defaults to a
        :class:`SlabRegion`.

    The mask itself -- normalizing it and putting the charge into the
    density -- is handled by :class:`gpaw.extensions.jellium.JelliumExtension`.
    """

    def __init__(self,
                 mask_r: UGArray,
                 *,
                 pw,
                 charge: float = 0.0,
                 region: JelliumRegion | None = None):
        self.region = region or SlabRegion()
        self._key: object | None = None
        super().__init__(mask_r, charge=charge, pw=pw)

    def set_region(self, mask_r: UGArray, key: object | None = None) -> None:
        """Normalize and store the shape of the jellium region.

        ``key`` is :meth:`JelliumRegion.key` for the geometry the mask was
        built from; :meth:`move` compares against it to spot a later
        geometry that needs no new mask.  Leaving it ``None`` just means
        the next :meth:`move` rebuilds unconditionally.
        """
        self.set_mask(mask_r)
        self._key = key

    def add_charge(self, nt_r: UGArray) -> None:
        self.update1(nt_r)

    def add_charge_pw(self, nt_g: PWArray | None) -> None:
        self.update1pw(nt_g)

    def move(self, relpos_ac, grid) -> tuple[float, float] | None:
        if not self.region.follows_atoms:
            return None
        key = self.region.key(grid, relpos_ac)
        if key is not None and key == self._key:
            # Same region as last time -- rebuilding the mask would cost a
            # grid pass, a norm and (in PW mode) a gather plus an FFT, and
            # hand back exactly the mask we already have.
            return None
        mask_r, z1, z2 = self.region.mask(grid, relpos_ac)
        self.set_region(mask_r, key)
        return z1, z2

    @property
    def region_area(self) -> float:
        cell_cv = self.mask_r.desc.cell_cv * Bohr
        return abs(np.linalg.det(cell_cv[:2, :2]))


class PotentialProbe:
    """Reads the electrode potential off the converged SCF state."""

    def potential(self, ibzwfs) -> float:
        """Electrode potential in V, on the vacuum scale."""
        raise NotImplementedError


class WorkFunctionProbe(PotentialProbe):
    """Electrode potential as the work function: minus the Fermi level.

    Assumes the Poisson solver has zeroed the vacuum electrostatic
    potential, as the SJM solver does.
    """

    def potential(self, ibzwfs) -> float:
        return -ibzwfs.fermi_level * Ha


class PotentialOptimizer:
    """Proposes the next excess-electron count for the target potential."""

    def next_charge(self,
                    charge: float,  # |e|
                    potential: float,  # V
                    target: float,  # V
                    *,
                    area: float) -> float:  # Å²
        """Number of excess electrons the method points at next."""
        raise NotImplementedError

    def last_attempt(self) -> tuple[float, float] | None:
        """The most recent (charge, potential) attempt, if one is kept.

        ``None`` from a method that records no attempts, which leaves the
        step clamp in :meth:`Potentiostat.equilibrate` nothing to compare
        against, so it never rejects.
        """
        return None

    def forget_attempts(self) -> None:
        """Drop the recorded attempts, but keep the slope estimate."""

    def restart(self) -> None:
        """Drop the attempts, but keep the estimate learned from them."""
        self.forget_attempts()

    def reset(self) -> None:
        """Forget the slope estimate and the attempts behind it."""

    def slope_estimate(self) -> float | None:
        """Estimated d(potential)/d(charge) in V per electron, if any."""
        return None


class DirectOptimizer(PotentialOptimizer):
    """Step the charge along a damped estimate of d(potential)/d(charge).

    The method SJM has always used.  ``slope`` (V/electron) is the initial
    guess; ``None`` guesses one from the apparent capacitance of a plate
    capacitor of the electrode ``area``.  Once there are two or more
    attempts to regress a slope from, the estimate is updated, damped by
    ``mixer``: ``new = mixer * old + (1 - mixer) * regressed``.  The
    regression uses the last ``slope_regression_depth`` attempts.

    ``mixer`` also sets how fast the step size may grow: a step is allowed
    to exceed the previous one by the fraction ``1 - mixer`` of the excess
    the slope asks for, so steps can recover after a small one instead of
    being capped by it forever.
    """

    def __init__(self,
                 *,
                 slope: float | None = None,  # V/electron
                 mixer: float = 0.2,
                 slope_regression_depth: int = 4):
        self.slope = slope
        self.mixer = mixer
        # (Charge, potential) history of the last few attempts, in V:
        self.history: deque[tuple[float, float]] = deque(
            maxlen=_check_depth(slope_regression_depth))

    @property
    def slope_regression_depth(self) -> int:
        return self.history.maxlen  # type: ignore[return-value]

    @slope_regression_depth.setter
    def slope_regression_depth(self, depth: int) -> None:
        # A deque cannot change its maxlen, so make a new one - keeping the
        # most recent attempts, which is what the deque itself would do.
        self.history = deque(self.history, maxlen=_check_depth(depth))

    def last_attempt(self) -> tuple[float, float] | None:
        """The most recent (charge, potential) attempt, if there is one."""
        return self.history[-1] if self.history else None

    def forget_attempts(self) -> None:
        """Drop the recorded attempts, but keep the slope estimate.

        The attempts belong to one geometry: once the atoms have moved, the
        potential at a given charge shifts, and a regression across the move
        would fit a line through two different systems.
        """
        self.history.clear()

    def reset(self) -> None:
        """Forget the slope estimate and the attempts behind it."""
        self.slope = None
        self.history.clear()

    def restart(self) -> None:
        """Drop the attempts, but keep the estimate learned from them.

        A slope not yet damped into :attr:`slope` is still carried over,
        by fixing it at what the attempts regress to.
        """
        self.slope = self.slope_estimate()
        self.history.clear()

    def slope_estimate(self) -> float | None:
        """The damped slope that the steps follow, in V per electron.

        Before there is one, the slope that the recorded attempts regress
        to, or ``None`` if they are too few.
        """
        if self.slope is not None:
            return self.slope
        return regress_slope(self.history)

    def _capacitance_slope(self, area: float) -> float:
        """Slope, in V/electron, of a plate capacitor of the cell's area."""
        # Å^2 -> cm^2 is 1e16 and F -> muF is 1e6:
        return -_e * 1e22 / (area * 10)

    def update_slope(self, area: float) -> float:
        """Update and return d(potential)/d(charge) in V/electron."""
        slope = regress_slope(self.history)
        if slope is None:
            pass  # too few distinct attempts to regress a slope from
        elif self.slope is None:
            self.slope = slope
        else:
            # Damp the new estimate: regressing a slope from a couple of
            # attempts can be noisy, particularly when the positions have
            # changed as well.
            self.slope = self.mixer * self.slope + (1.0 - self.mixer) * slope
        if self.slope is None or self.slope > -1e-10:
            # No estimate yet, or one that is not usefully negative (adding
            # electrons must lower the potential) -- fall back to the
            # capacitance guess rather than trust a bad regression:
            self.slope = self._capacitance_slope(area)
        return self.slope

    def next_charge(self,
                    charge: float,
                    potential: float,
                    target: float,
                    *,
                    area: float) -> float:
        """Charge that the slope points at, given this potential.

        The attempt is recorded in the history and the slope estimate is
        updated, so this must be called at most once per attempt.
        """
        self.history.append((charge, potential))
        slope = self.update_slope(area)
        dc = (target - potential) / slope
        if len(self.history) > 1:
            (c2, _), (c1, _) = list(self.history)[-2:]
            previous = abs(c2 - c1)
            if previous > 0.0 and abs(dc) > previous:
                # Trust region.  A step may grow beyond the previous one,
                # but the increase the slope asks for is only granted in
                # part: both the slope and (target - potential) are local
                # estimates, so a step much larger than the one that
                # produced them is extrapolating.  The fraction granted is
                # the weight the fresh regression carries in update_slope,
                # 1 - mixer -- confidence in the new information from the
                # same source.  A fully damped slope (mixer = 1) keeps the
                # old behaviour of never growing; an undamped one
                # (mixer = 0) takes the whole step the slope points at.
                growth = min(max(1.0 - self.mixer, 0.0), 1.0)
                dc *= (previous + growth * (abs(dc) - previous)) / abs(dc)
        return charge + dc


class EquilibrationResult(enum.Enum):
    """Outcome of one :meth:`Potentiostat.equilibrate` call."""

    #: The potential is within tolerance -- the equilibration is done.
    CONVERGED = enum.auto()
    #: The charge was stepped; another SCF pass is needed.  Counts as an
    #: attempt against ``max_iters``.
    STEPPED = enum.auto()
    #: The last step left the linear region (moved the potential by more
    #: than ``max_potential_stepsize`` and further from the target); it
    #: has been undone halfway and another SCF pass is needed.  Does *not*
    #: count as an attempt.
    REJECTED = enum.auto()


class PotentialConvergenceError(ConvergenceError):
    """Raised if the electrode potential did not equilibrate."""


class Potentiostat:
    """Holds a target electrode potential by adjusting an excess charge.

    Composes three pluggable parts:

    * ``charging``  -- a :class:`ChargingScheme`: how the excess charge is
      put into the cell (default: a jellium slab).
    * ``probe``     -- a :class:`PotentialProbe`: how the potential is read
      off the converged SCF state (default: the work function).
    * ``optimizer`` -- a :class:`PotentialOptimizer`: how the charge is
      stepped toward ``target_potential``.  Keep one in constant-charge
      mode too, so that its settings and its slope estimate survive a
      switch between the modes; ``None`` only suits a front end that never
      targets a potential.

    The potentiostat holds every setting of the potential control and the
    state of the equilibration.  A front end such as
    :class:`gpaw.extensions.electrocatalysis.SJM` hands the settings
    over, with defaults of its own, and reads them back from here.

    Parameters
    ----------
    charging:
        The charging scheme.
    target_potential:
        Electrode potential to equilibrate to, in V on the vacuum scale.
        ``None`` gives a constant-charge calculation: the charge is simply
        held where it is.
    optimizer:
        The potential optimizer.
    probe:
        The potential probe.  Defaults to :class:`WorkFunctionProbe`.
    tol:
        Tolerance for the deviation of the target potential, in V.
    max_potential_stepsize:
        If an accepted step moves the potential by more than this many volts
        *and* ends up further from the target than it started, the step is
        rejected: the change in excess electrons is halved and another SCF
        pass is run, without counting the rejected point as an attempt (see
        :attr:`EquilibrationResult.REJECTED`).  This keeps the optimizer
        from leaving the linear region on a bad slope estimate.  Set to
        ``None`` to turn the rejection/halving off entirely.  Default: 2 V.
    max_iters:
        Maximum number of attempts to equilibrate the potential for one
        geometry before a :class:`PotentialConvergenceError` is raised.
        Default: 10.
    max_reruns:
        How many times in a row a step may be rejected and halved before a
        :class:`PotentialConvergenceError` is raised.  Default: 10.
    always_adjust:
        Once the potential is within ``tol``, stage one more step in the
        charge anyway, to be applied to the next geometry by
        :meth:`new_geometry` instead of being re-converged.  Default: False.
    grand_output:
        Report the grand-potential energy rather than the canonical energy.
        Read by the front end, which writes the results.  Default: True.
    """

    def __init__(self,
                 *,
                 charging: ChargingScheme,
                 target_potential: float | None = None,  # V
                 optimizer: PotentialOptimizer | None = None,
                 probe: PotentialProbe | None = None,
                 tol: float = 0.001,  # V
                 max_potential_stepsize: float | None = 2,  # V
                 max_iters: int = 10,
                 max_reruns: int = 10,
                 always_adjust: bool = False,
                 grand_output: bool = True):
        self.charging = charging
        self.target_potential = target_potential
        self.optimizer = optimizer
        self.probe = probe or WorkFunctionProbe()
        self.tol = tol
        self.max_potential_stepsize = max_potential_stepsize
        self.max_iters = max_iters
        self.max_reruns = max_reruns
        self.always_adjust = always_adjust
        self.grand_output = grand_output

        # Consecutive rejected (halved) steps since the last accepted one:
        self.reruns = 0
        # (Charge, potential in V) of every accepted attempt of the current
        # equilibration:
        self.attempts: list[tuple[float, float]] = []
        # Charge staged by always_adjust for the next geometry:
        self.pending_charge: float | None = None

    @property
    def constant_potential(self) -> bool:
        """Is there a target potential to equilibrate to?"""
        return self.target_potential is not None

    @property
    def charge(self) -> float:
        """Number of excess electrons."""
        return self.charging.charge

    @charge.setter
    def charge(self, q: float) -> None:
        self.charging.charge = q

    @property
    def niters(self) -> int:
        """Number of attempts made in the current equilibration."""
        return len(self.attempts)

    def set_charge(self, charge: float) -> None:
        """Set the number of excess electrons explicitly.

        An explicit charge wins over one staged by ``always_adjust``.
        """
        self.charge = charge
        self.pending_charge = None

    def set_target_potential(self, target: float | None) -> None:
        """Change the target potential; ``None`` switches to constant charge.

        A switch between the two modes starts a fresh equilibration: the
        optimizer keeps its slope estimate but not the attempts behind it.
        """
        if ((target is None) != (self.target_potential is None) and
                self.optimizer is not None):
            self.optimizer.restart()
        self.target_potential = target

    def slope_estimate(self) -> float | None:
        """Estimated d(potential)/d(charge) in V per electron, if any."""
        if self.optimizer is None:
            return None
        return self.optimizer.slope_estimate()

    def reset_attempts(self) -> None:
        """Give the next equilibration a fresh record and budget.

        The optimizer's slope estimate and its attempts are kept.
        """
        self.attempts = []

    def new_geometry(self) -> float | None:
        """Start equilibrating a new geometry.

        Applies a charge staged by ``always_adjust`` and returns it, or
        ``None`` if there was none.  The attempts are forgotten -- the
        optimizer's too, since a regression across the move would fit a
        line through two different systems -- but the slope estimate is
        carried over.
        """
        charge = self.pending_charge
        if charge is not None:
            self.set_charge(charge)
        self.reset_attempts()
        if self.optimizer is not None:
            self.optimizer.forget_attempts()
        return charge

    def post_scf_convergence(self,
                             ibzwfs,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        """Equilibrate once the SCF criteria have passed.

        Wraps :meth:`equilibrate` in the bookkeeping of a whole
        equilibration: accepted attempts count against ``max_iters`` and
        rejected steps in a row against ``max_reruns``, and a
        :class:`PotentialConvergenceError` is raised when either runs out.
        Returns ``True`` once the potential is equilibrated (always, in
        constant-charge mode) and ``False`` when another SCF pass is needed.
        """
        if not self.constant_potential:
            return True

        # Grab the values belonging to this attempt before the charge is
        # stepped on to the next one:
        charge = self.charge
        potential = self.probe.potential(ibzwfs)

        result = self.equilibrate(ibzwfs, nelectrons, occ_calc, mixer, log)

        if result is EquilibrationResult.REJECTED:
            # The step left the linear region and has been halved; re-run
            # the SCF but do not count it as an attempt.  Give up once the
            # halving has been repeated max_reruns times in a row.
            self.reruns += 1
            if self.reruns > self.max_reruns:
                raise PotentialConvergenceError(
                    self._convergence_error_message(
                        log, reason=f'{self.max_reruns:d} halvings of a step '
                        f'that kept overshooting max_potential_stepsize '
                        f'({self.max_potential_stepsize} V)'))
            return False

        # A step was accepted (or we are already converged): the rerun
        # streak, if any, is over.
        self.reruns = 0
        self.attempts.append((charge, potential))

        if result is EquilibrationResult.CONVERGED:
            if self.always_adjust:
                # Step the charge on anyway, but do not re-converge: the new
                # value is picked up by the next geometry (new_geometry()),
                # so that the potential can equilibrate alongside the
                # structure.
                self.pending_charge = self.propose_next(ibzwfs)
            return True

        if self.niters >= self.max_iters:
            raise PotentialConvergenceError(
                self._convergence_error_message(log))
        return False

    def _convergence_error_message(self, log,
                                   reason: str | None = None) -> str:
        reason = reason or f'{self.max_iters:d} attempts'
        msg = (f'The electrode potential could not be equilibrated to '
               f'{self.target_potential:.3f} +/- {self.tol:.3f} V in '
               f'{reason}.  This may indicate that your '
               'work function is noisier than your potential tolerance; you '
               'may try tightening the SCF convergence criteria.  The excess '
               'electrons and the resulting potentials are listed below; '
               'plotting them could give you insight into the problem.')
        msg = textwrap.fill(msg) + '\n'
        for charge, potential in self.attempts:
            msg += f'{charge:+.6f} {potential:.6f}\n'
        log(msg, flush=True)
        return msg

    def equilibrate(self,
                    ibzwfs,
                    nelectrons: float,
                    occ_calc,
                    mixer,
                    log) -> EquilibrationResult:
        """One equilibration attempt, after the SCF criteria have passed.

        Returns :attr:`EquilibrationResult.CONVERGED` when the potential is
        within tolerance (or in constant-charge mode, where there is
        nothing to do); :attr:`~EquilibrationResult.STEPPED` after stepping
        the charge and re-running the occupations for another SCF pass; and
        :attr:`~EquilibrationResult.REJECTED` when the last step left the
        linear region -- the change in charge is halved and another SCF
        pass is run, without the rejected point counting as an attempt or
        feeding the slope regression.
        """
        if not self.constant_potential:
            return EquilibrationResult.CONVERGED
        # constant_potential means both of these are set (see __init__):
        target = self.target_potential
        optimizer = self.optimizer
        assert target is not None and optimizer is not None
        potential = self.probe.potential(ibzwfs)
        log(f'charge: {self.charge:.6f} |e|, potential: {potential:.3f} V')

        # Reject a step that left the linear region: if the potential moved
        # by more than max_potential_stepsize *and* ended up further from
        # the target than it started, halve the change in excess electrons
        # and try again.  The rejected point is not recorded, so it does
        # not pull the slope regression around.  (This is the legacy SJM
        # ``max_step`` knob.)
        previous = optimizer.last_attempt()
        if self.max_potential_stepsize is not None and previous is not None:
            prev_charge, prev_potential = previous
            moved = abs(potential - prev_potential)
            overshot = abs(prev_potential - target) < abs(potential - target)
            if moved > self.max_potential_stepsize and overshot:
                halved = 0.5 * (prev_charge + self.charge)
                log(f'  step changed the potential by {moved:.3f} V, over '
                    'max_potential_stepsize '
                    f'({self.max_potential_stepsize} V), and away from the '
                    'target -- rejected; halving the charge step to '
                    f'{halved:+.6f} |e|.')
                ibzwfs.calculate_occs(
                    occ_calc, nelectrons + (halved - self.charge))
                self.charge = halved
                mixer.reset()
                return EquilibrationResult.REJECTED

        if abs(potential - target) <= self.tol:
            return EquilibrationResult.CONVERGED
        new_charge = optimizer.next_charge(
            self.charge, potential, target,
            area=self.charging.region_area)
        ibzwfs.calculate_occs(occ_calc,
                              nelectrons + (new_charge - self.charge))
        self.charge = new_charge
        mixer.reset()
        return EquilibrationResult.STEPPED

    def propose_next(self, ibzwfs) -> float:
        """Charge the optimizer points at, without re-converging it.

        Used by ``always_adjust`` to stage a step for the next geometry.
        """
        target = self.target_potential
        optimizer = self.optimizer
        assert target is not None and optimizer is not None
        return optimizer.next_charge(
            self.charge, self.probe.potential(ibzwfs), target,
            area=self.charging.region_area)

    # Density hooks, straight through to the charging scheme:
    def add_charge(self, nt_r) -> None:
        self.charging.add_charge(nt_r)

    def add_charge_pw(self, nt_g) -> None:
        self.charging.add_charge_pw(nt_g)


def _check_depth(depth: int) -> int:
    """Validate the number of attempts to regress the slope over."""
    if depth < 2:
        raise ValueError(
            f'slope_regression_depth must be at least 2, got {depth}.  Two '
            'attempts are the fewest that a slope can be fitted to.')
    return depth


def regress_slope(attempts) -> float | None:
    """Slope of potential vs. excess electrons, in V/electron.

    ``attempts`` is a sequence of (charge, potential) pairs, the charge in
    electrons and the potential in volts.  Returns ``None`` if there are
    too few distinct charges to fit a line.
    """
    attempts = list(attempts)
    if len(attempts) < 2:
        return None
    charge_i = np.array([charge for charge, _ in attempts])
    potential_i = np.array([potential for _, potential in attempts])
    if np.ptp(charge_i) < 1e-10:
        return None
    return float(np.polyfit(charge_i, potential_i, 1)[0])
