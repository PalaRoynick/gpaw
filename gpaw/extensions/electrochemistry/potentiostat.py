from __future__ import annotations

import enum
import textwrap
from collections import deque

import numpy as np
from ase.units import Bohr, Ha, _e

from gpaw import ConvergenceError
from gpaw.core import PWArray, UGArray
from gpaw.mpi import broadcast_float
from gpaw.extensions import Extension
from gpaw.extensions.jellium import JelliumExtension


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
    specified by at most two of ``top``, ``bottom`` and ``thickness``.

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
                 top: float | None = None,  # Å; default: -1.0
                 bottom: float | str | None = None,  # Å; default: -3.0
                 thickness: float | None = None,  # Å
                 fix_bottom: bool = False):
        if isinstance(bottom, str):
            raise ValueError(
                f'Unknown lower limit for the jellium region: {bottom!r}.  '
                'Give a number of Angstrom instead.')

        if all(x is not None for x in (top, bottom, thickness)):
            raise ValueError(
                'The jellium region has been overspecified: give at most two '
                "of 'top', 'bottom' and 'thickness'.")
        # A missing limit keeps its default, unless a thickness fixes it; a
        # thickness on its own grows upward from the default bottom.
        if thickness is None and top is None:
            top = -1.0  # Å below the upper cell boundary
        if bottom is None and (thickness is None or top is None):
            bottom = -3.0  # Å above the highest atom

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
    it is not implemented this version of SJM yet.

    Parameters
    ----------
    top:
        Upper cut-off of the region.  A positive value is the z coordinate
        in Å; a negative value is the distance below the upper cell
        boundary.  Default: -1 Å.
    cavity:
        The solvation cavity to read ``g_g`` from.
    """

    def __init__(self, top: float = -1.0, *, cavity=None):  # Å
        self.top = top
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
    A ``'shape'`` key in ``spec`` (``'slab'`` (default) or ``'cavity'``)
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

    # Excess electrons put into the cell; the counter charge is minus this.
    excess_electrons: float
    # Shape of the charged region.
    region: JelliumRegion
    # Its normalized mask on the fine grid, as stored by set_region().
    mask_r: UGArray

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
    excess_electrons:
        Number of excess electrons (the jellium holds minus this).
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
                 excess_electrons: float = 0.0,
                 region: JelliumRegion | None = None):
        self.region = region or SlabRegion()
        self._key: object | None = None
        super().__init__(mask_r, charge=excess_electrons, pw=pw)

    @property
    def charge(self) -> float:
        """The same number under the name the jellium base class uses.

        :class:`gpaw.extensions.jellium.JelliumExtension` stores and
        subtracts ``charge``, which puts minus the excess electrons into
        the density -- the counter charge.
        """
        return self.excess_electrons

    @charge.setter
    def charge(self, value: float) -> None:
        self.excess_electrons = value

    def set_region(self, mask_r: UGArray, key: object | None = None) -> None:
        """Normalize and store the shape of the jellium region.

        ``key`` is :meth:`JelliumRegion.key` for the geometry the mask was
        built from; :meth:`move` compares against it to spot a later
        geometry that needs no new mask.  Leaving it ``None`` just means
        the next :meth:`move` rebuilds unconditionally.
        """
        self.set_mask(mask_r)
        self._key = key

    add_charge = JelliumExtension.update1
    add_charge_pw = JelliumExtension.update1pw

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

    slope: float | None = None  # estimated d(potential)/d(charge), V/electron

    def next_excess_electrons(self,
                              excess_electrons: float,  # |e|
                              potential: float,  # V
                              target: float,  # V
                              *,
                              area: float) -> float:  # Å²
        """Number of excess electrons the method points at next.

        The attempt must have been handed to :meth:`record_attempt` first;
        this only proposes the step and records nothing.
        """
        raise NotImplementedError

    def record_attempt(self,
                       excess_electrons: float,  # |e|
                       potential: float,  # V
                       *,
                       area: float) -> None:  # Å²
        """Learn from an attempt, whether or not a step follows it."""

    def last_attempt(self) -> tuple[float, float] | None:
        """The most recent (excess electrons, potential) attempt, if kept.

        ``None`` from a method that records no attempts, which leaves the
        step clamp in :meth:`Potentiostat.equilibrate` nothing to compare
        against, so it never rejects.
        """
        return None

    def forget_attempts(self) -> None:
        """Drop the recorded attempts, but keep the slope estimate."""

    def reset(self) -> None:
        """Forget the slope estimate and the attempts behind it."""

    @property
    def n_attempts(self) -> int:
        """Number of attempts the slope estimate is based on."""
        return 0


class DirectOptimizer(PotentialOptimizer):
    """Step the charge along a damped estimate of d(potential)/d(charge).

    The method SJM has always used.  ``slope`` (V/electron) is the initial
    guess; ``None`` guesses one from the apparent capacitance of a plate
    capacitor of the electrode ``area``.  Once there are two or more
    attempts to regress a slope from, the estimate is updated, damped by
    ``mixer``: ``new = mixer * old + (1 - mixer) * regressed``.  The
    regression uses the last ``slope_regression_depth`` attempts.
    """

    def __init__(self,
                 *,
                 slope: float | None = None,  # V/electron
                 mixer: float = 0.2,
                 slope_regression_depth: int = 4):
        self.slope = slope
        self.mixer = mixer
        # (Charge, potential) history of the last few attempts, in V:
        self.history: deque[tuple[float, float]] = deque()
        self.slope_regression_depth = slope_regression_depth

    @property
    def slope_regression_depth(self) -> int:
        return self.history.maxlen  # type: ignore[return-value]

    @slope_regression_depth.setter
    def slope_regression_depth(self, depth: int) -> None:
        if depth < 2:
            raise ValueError(
                f'slope_regression_depth must be at least 2, got {depth}.  '
                'Two attempts are the fewest that a slope can be fitted to.')
        self.history = deque(self.history, maxlen=depth)

    def last_attempt(self) -> tuple[float, float] | None:
        return self.history[-1] if self.history else None

    def forget_attempts(self) -> None:
        self.history.clear()

    def reset(self) -> None:
        self.slope = None
        self.history.clear()

    @property
    def n_attempts(self) -> int:
        return len(self.history)

    def record_attempt(self,
                       excess_electrons: float,
                       potential: float,
                       *,
                       area: float) -> None:
        """Record an attempt and update the slope estimate from it.

        Every accepted attempt counts, including the one that lands within
        tolerance: the last two attempts of an equilibration are a
        measurement of the slope like any other.  With too few distinct
        attempts to regress, the estimate is left alone.
        """
        self.history.append((excess_electrons, potential))
        slope = regress_slope(self.history)
        if slope is None:
            return
        self.slope = (self.mixer * self.slope + (1.0 - self.mixer) * slope
                      if self.slope else slope)
        self._usable_slope(area)

    def _usable_slope(self, area: float) -> float:
        """The slope estimate, or the capacitance guess for a bad one."""
        if self.slope is None or self.slope > -1e-10:
            # No estimate yet, or one that is not usefully negative (adding
            # electrons must lower the potential) -- fall back to the
            # capacitance guess rather than trust a bad regression:
            self.slope = capacitance(10, area)  # 10 muF/cm^2; reciprocal
        return self.slope

    def next_excess_electrons(self,
                              excess_electrons: float,
                              potential: float,
                              target: float,
                              *,
                              area: float) -> float:
        """Excess electrons that the slope points at, given this potential.

        The attempt must have been recorded with :meth:`record_attempt`
        first: the slope estimate and the trust region both come from the
        history, which this leaves untouched.
        """
        slope = self._usable_slope(area)
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
                # the weight the fresh regression carries in record_attempt,
                # 1 - mixer -- confidence in the new information from the
                # same source.  A fully damped slope (mixer = 1) keeps the
                # old behaviour of never growing; an undamped one
                # (mixer = 0) takes the whole step the slope points at.
                growth = min(max(1.0 - self.mixer, 0.0), 1.0)
                dc *= (previous + growth * (abs(dc) - previous)) / abs(dc)
        return excess_electrons + dc


class EquilibrationResult(enum.Enum):
    """Outcome of one :meth:`Potentiostat.equilibrate` call."""

    # The potential is within tolerance -- the equilibration is done.
    converged = enum.auto()
    # The excess electrons were stepped; another SCF pass is needed.
    stepped = enum.auto()
    # The last step moved the potential by more than ``max_potential_stepsize``
    # and further from the target. It will be rerun with half the previous
    # step.  Does *not* count as an attempt.
    rejected = enum.auto()


class PotentialConvergenceError(ConvergenceError):
    """Raised if the electrode potential did not equilibrate."""


class Potentiostat(Extension):
    """Holds a target electrode potential by adjusting an excess charge.

    It is the GPAW extension of the potential control: it puts the charge
    into the density, moves it with the atoms and equilibrates after each
    SCF loop.

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
    state of the equilibration. A front end such as
    :class:`gpaw.extensions.electrochemistry.SJM` hands the settings
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
        Default: 0.005 V.
    max_potential_stepsize:
        If an accepted step moves the potential by more than this many volts
        *and* ends up further from the target than it started, the step is
        rejected: the change in excess electrons is halved and another SCF
        pass is run, without counting the rejected point as an attempt (see
        :attr:`EquilibrationResult.rejected`).  This keeps the optimizer
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
        Report the grand-potential energy rather than the canonical energy
        in the results (see :meth:`write_results`).  Default: True.
    log:
        Where the equilibration and the moves of the region are reported.
    """

    name = 'potentiostat'

    def __init__(self,
                 *,
                 charging: ChargingScheme,
                 target_potential: float | None = None,  # V
                 optimizer: PotentialOptimizer | None = None,
                 probe: PotentialProbe | None = None,
                 tol: float = 0.005,  # V
                 max_potential_stepsize: float | None = 2,  # V
                 max_iters: int = 10,
                 max_reruns: int = 10,
                 always_adjust: bool = False,
                 grand_output: bool = True,
                 log=None):
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
        self.log = log
        self.dft = None  # the last converged calculation

        # Consecutive rejected (halved) steps since the last accepted one:
        self.reruns = 0
        # (Charge, potential in V) of every accepted attempt of the current
        # equilibration:
        self.attempts: list[tuple[float, float]] = []
        # Charge staged by always_adjust for the next geometry:
        self.pending_excess_electrons: float | None = None

        # Set by post_converged(), for the last converged calculation:
        self.fermi_level = np.nan  # Ha
        self.electrode_potential = np.nan  # V
        self.converged_excess_electrons = np.nan
        self.canonical_free = np.nan  # Ha
        self.canonical_extrapolated = np.nan  # Ha
        self.omega_free = np.nan  # Ha
        self.omega_extrapolated = np.nan  # Ha

    @property
    def constant_potential(self) -> bool:
        """Is there a target potential to equilibrate to?"""
        return self.target_potential is not None

    @property
    def excess_electrons(self) -> float:
        """Number of excess electrons."""
        return self.charging.excess_electrons

    @excess_electrons.setter
    def excess_electrons(self, q: float) -> None:
        self.charging.excess_electrons = q

    # The name GPAW sums the charge of the cell from:
    charge = excess_electrons

    @property
    def niters(self) -> int:
        """Number of attempts made in the current equilibration."""
        return len(self.attempts)

    def set_excess_electrons(self, excess_electrons: float) -> None:
        """Set the number of excess electrons explicitly.

        An explicit value wins over one staged by ``always_adjust``.
        """
        self.excess_electrons = excess_electrons
        self.pending_excess_electrons = None

    def set_target_potential(self, target: float | None) -> None:
        """Change the target potential; ``None`` switches to constant charge.

        A switch between the two modes starts a fresh equilibration: the
        optimizer keeps its slope estimate but not the attempts behind it.
        """
        if ((target is None) != (self.target_potential is None) and
                self.optimizer is not None):
            self.optimizer.forget_attempts()
        self.target_potential = target

    def new_geometry(self) -> float | None:
        """Start equilibrating a new geometry.

        Applies excess electrons staged by ``always_adjust`` and returns
        them, or ``None`` if there were none.  The attempts are forgotten
        -- the optimizer's too, since a regression across the move would fit a
        line through two different systems -- but the slope estimate is
        carried over.
        """
        staged = self.pending_excess_electrons
        if staged is not None:
            self.set_excess_electrons(staged)
        self.attempts = []
        if self.optimizer is not None:
            self.optimizer.forget_attempts()
        return staged

    def update1(self, nt_r) -> None:
        self.charging.add_charge(nt_r)

    def update1pw(self, nt_g) -> None:
        self.charging.add_charge_pw(nt_g)

    def move_atoms(self, relpos_ac) -> None:
        # A fresh budget of equilibration attempts for the new geometry,
        # starting from the value staged by always_adjust, if any:
        if self.new_geometry() is not None:
            self.log(f'Excess electrons adjusted to '
                     f'{self.excess_electrons:+.5f} for the new geometry '
                     '(always_adjust).')
        zs = self.charging.move(relpos_ac, self.charging.mask_r.desc)
        if zs is not None:
            self.log(f'Jellium region moved to {zs[0]:.3f} to '
                     f'{zs[1]:.3f}  # Å')

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
        excess_electrons = self.excess_electrons
        potential = self.probe.potential(ibzwfs)

        result = self.equilibrate(ibzwfs, nelectrons, occ_calc, mixer, log)

        if result is EquilibrationResult.rejected:
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
        self.attempts.append((excess_electrons, potential))

        if result is EquilibrationResult.converged:
            if self.always_adjust:  # stage a step for the next geometry
                assert self.optimizer and self.target_potential is not None
                self.pending_excess_electrons = (
                    self.optimizer.next_excess_electrons(
                        excess_electrons, potential, self.target_potential,
                        area=self.charging.region_area))
            return True

        if self.niters >= self.max_iters:
            raise PotentialConvergenceError(
                self._convergence_error_message(log))
        return False

    def post_converged(self, dft) -> None:
        """Finish a converged calculation.

        Reads the electrode potential and the Fermi level off the converged
        state, reports the equilibration, Legendre-transforms the energies
        to the grand potential and writes the results (see
        :meth:`write_results`).  The next calculation -- the next step of a
        relaxation, say -- then gets a fresh equilibration budget and a
        fresh record to report.
        """
        self.dft = dft
        log = dft.log
        (fermi_level,) = dft.ibzwfs.fermi_levels
        self.fermi_level = broadcast_float(float(fermi_level), dft.comm)
        self.electrode_potential = broadcast_float(
            self.probe.potential(dft.ibzwfs), dft.comm)

        # Report where the equilibration ended up; the intermediate attempts
        # are summarized by the regressed slope:
        if self.constant_potential and self.attempts:
            excess_electrons, potential = self.attempts[-1]
            log(f'Potential found to be {potential:.5f} V '
                f'(with {excess_electrons:+.5f} excess electrons, '
                f'attempt {self.niters:d}/{self.max_iters:d}).')
            assert self.optimizer is not None  # or there were no attempts
            slope = self.optimizer.slope
            if slope is not None:
                log(f'Slope estimated from the last '
                    f'{self.optimizer.n_attempts:d} attempts is '
                    f'{slope:.4f} V/electron,')
                c = capacitance(slope, self.charging.region_area)
                log(f'or apparent capacitance of {c:.4f} muF/cm^2')
            log('Potential is within tolerance. Equilibrated.')
            if self.pending_excess_electrons is not None:
                log('always_adjust: the next geometry starts from '
                    f'{self.pending_excess_electrons:+.5f} excess electrons.')
            log(flush=True)

        # Legendre transform: Omega = E - N mu, with mu the Fermi level.
        N = self.converged_excess_electrons = self.excess_electrons
        self.canonical_free = broadcast_float(dft.energies.total_free,
                                              dft.comm)
        self.canonical_extrapolated = broadcast_float(
            dft.energies.total_extrapolated, dft.comm)
        mu_N = self.fermi_level * N
        self.omega_free = self.canonical_free - mu_N
        self.omega_extrapolated = self.canonical_extrapolated - mu_N

        log('Legendre-transformed energies '
            '(grand potential, Omega = E - N mu):')
        log(f'  N (excess electrons): {N:+14.6f}')
        log(f'  mu (-work function):  {self.fermi_level * Ha:+14.6f}  # eV')
        log(f'  (Grand) free energy:  {self.omega_free * Ha:+14.6f}')
        log(f'  (Grand) extrapolated: {self.omega_extrapolated * Ha:+14.6f}')
        log()

        self.write_results(dft.results, log)
        self.attempts = []  # a fresh record and budget for the next one

    def write_results(self, results: dict, log) -> None:
        """Write the last converged calculation into `results`.

        That is the excess electrons, the electrode potential and --
        depending on ``grand_output`` -- the grand-potential or the
        canonical energies.  It is cheap, so a change of ``grand_output``
        only needs to call this again.
        """
        # Unlike energy and forces, the first two are not in ASE's
        # all_properties, so an ASE trajectory silently drops them; they do
        # reach calc.results, a .gpw file and the text log.
        results['excess_electrons'] = self.converged_excess_electrons
        results['electrode_potential'] = self.electrode_potential / Ha
        if self.grand_output:
            results['free_energy'] = self.omega_free
            results['energy'] = self.omega_extrapolated
            log('Grand-potential energy was written into results.\n',
                flush=True)
        else:
            results['free_energy'] = self.canonical_free
            results['energy'] = self.canonical_extrapolated
            log('Canonical energy was written into results.\n', flush=True)

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
        for excess_electrons, potential in self.attempts:
            msg += f'{excess_electrons:+.6f} {potential:.6f}\n'
        log(msg, flush=True)
        return msg

    def equilibrate(self,
                    ibzwfs,
                    nelectrons: float,
                    occ_calc,
                    mixer,
                    log) -> EquilibrationResult:
        """One equilibration attempt, after the SCF criteria have passed.

        Returns :attr:`EquilibrationResult.converged`,
        :attr:`~EquilibrationResult.stepped` after stepping the charge and
        re-running the occupations for another SCF pass; and
        :attr:`~EquilibrationResult.rejected` when a rerun is needed.
        """
        if not self.constant_potential:
            return EquilibrationResult.converged
        target = self.target_potential
        optimizer = self.optimizer
        assert target is not None and optimizer is not None
        potential = self.probe.potential(ibzwfs)
        log(f'excess electrons: {self.excess_electrons:+.6f} |e|, '
            f'potential: {potential:.3f} V')

        previous = optimizer.last_attempt()
        if self.max_potential_stepsize is not None and previous is not None:
            prev_excess_electrons, prev_potential = previous
            moved = abs(potential - prev_potential)
            overshot = abs(prev_potential - target) < abs(potential - target)
            if moved > self.max_potential_stepsize and overshot:
                halved = 0.5 * (prev_excess_electrons + self.excess_electrons)
                log(f'  step changed the potential by {moved:.3f} V, over '
                    f'max_potential_stepsize ({self.max_potential_stepsize} '
                    'V), and away from the target -- rejected; halving the '
                    f'step in excess electrons to {halved:+.6f} |e|.')
                ibzwfs.calculate_occs(
                    occ_calc, nelectrons + (halved - self.excess_electrons))
                self.excess_electrons = halved
                mixer.reset()
                return EquilibrationResult.rejected

        area = self.charging.region_area
        optimizer.record_attempt(self.excess_electrons, potential, area=area)
        if abs(potential - target) <= self.tol:
            return EquilibrationResult.converged
        new_excess_electrons = optimizer.next_excess_electrons(
            self.excess_electrons, potential, target, area=area)
        ibzwfs.calculate_occs(
            occ_calc,
            nelectrons + (new_excess_electrons - self.excess_electrons))
        self.excess_electrons = new_excess_electrons
        mixer.reset()
        return EquilibrationResult.stepped


def capacitance(slope: float, area: float) -> float:
    """Apparent capacitance in muF/cm^2 of a slope in V per electron."""
    # Å^2 -> cm^2 is 1e16 and F -> muF is 1e6:
    return -_e * 1e22 / (area * slope)


def regress_slope(attempts) -> float | None:
    """Slope of potential vs. excess electrons, in V/electron.

    ``attempts`` is a sequence of (excess electrons, potential) pairs, in
    electrons and volts.  Returns ``None`` if there are too few distinct
    values to fit a line.
    """
    n_i, potential_i = np.array(list(attempts)).reshape(-1, 2).T
    if len(n_i) < 2 or np.ptp(n_i) < 1e-10:
        return None
    return float(np.polyfit(n_i, potential_i, 1)[0])
