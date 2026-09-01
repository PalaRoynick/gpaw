from __future__ import annotations

from collections import deque

import numpy as np
from ase.units import Bohr, Ha, _e

from gpaw.core import PWArray, PWDesc, UGArray
from gpaw.core.domain import Domain
from gpaw.dft import ExtensionInput
from gpaw.extensions import Extension
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.ibzwfs import IBZWaveFunctions

#: Default number of previous attempts that the slope is regressed over.
DEFAULT_REGRESSION_DEPTH = 4

#: Apparent capacitance, in muF/cm^2, behind the guessed slope.  A jellium a
#: few Angstrom above the slab is a plate capacitor of roughly this
#: capacitance.  (The legacy implementation guessed 10 muF/cm^2 -- typical of a
#: solvated double layer -- which is a 2.4 times larger, and for a small cell
#: distinctly less stable, first step.)
GUESSED_CAPACITANCE = 4.2


class Jellium(ExtensionInput):
    name = 'jellium'

    def __init__(self,
                 charge: float):
        self.charge = charge

    def todict(self):
        return {'charge': self.charge}

    def update_mask(self, mask_r) -> None:
        mask_r.data[:] = 1.0

    def build(self, builder: DFTComponentsBuilder):
        mask_r = builder.fine_grid.zeros()
        self.update_mask(mask_r)
        # PW-mode needs this one:
        # pw = builder.electrostatic_potential_desc
        pw = builder.interpolation_desc
        return JelliumExtension(mask_r, charge=self.charge, pw=pw)


class JelliumExtension(Extension):
    def __init__(self,
                 mask_r: UGArray,
                 *,
                 charge: float,
                 pw: Domain):
        self.charge = charge
        self.pw = pw
        self.set_mask(mask_r)

    def set_mask(self, mask_r: UGArray) -> None:
        """Normalize and store the shape of the jellium region.

        Can be called again later if the region has to follow the atoms.
        """
        norm = mask_r.integrate()
        if norm == 0.0:
            raise ValueError('The jellium region contains no grid points.')
        self.mask_r = mask_r
        mask_r.data *= 1.0 / norm
        self.mask_g = None
        if isinstance(self.pw, PWDesc):
            mask0_r = mask_r.gather()
            if mask0_r is not None:
                self.mask_g = mask0_r.fft(pw=self.pw.new(comm=None))

    def update1(self, nt_r: UGArray) -> None:
        nt_r.data -= self.mask_r.data * self.charge

    def update1pw(self, nt_g: PWArray | None) -> None:
        if nt_g is None:
            return  # only rank-0 needs to do anything
        assert self.mask_g is not None
        nt_g.data -= self.mask_g.data * self.charge


class FixedPotentialJelliumExtension(JelliumExtension):
    def __init__(self,
                 mask_r: UGArray,
                 *,
                 pw: Domain,
                 workfunction_target: float,  # eV
                 excess_electrons_guess=0.0,
                 tolerance: float = 0.001,  # eV
                 slope: float | None = None,  # V/electron
                 mixer: float = 0.5,
                 slope_regression_depth: int = DEFAULT_REGRESSION_DEPTH):
        """Adjust jellium charge to get the desired Fermi-level.

        The charge is stepped along an estimate of the slope of the
        potential vs. the number of excess electrons.  ``slope`` is the
        initial guess, in volts per electron; ``None`` guesses one from the
        apparent capacitance of a plate capacitor, see
        :data:`GUESSED_CAPACITANCE`.  Once there are two or more
        attempts to regress a slope from, the estimate is updated, damped by
        ``mixer``: ``new = mixer * old + (1 - mixer) * regressed``.  The
        regression uses the last ``slope_regression_depth`` attempts.
        """
        super().__init__(mask_r, charge=excess_electrons_guess or 0.0, pw=pw)
        self.workfunction_target = workfunction_target / Ha
        self.tolerance = tolerance / Ha
        self.slope = slope
        self.mixer = mixer
        # (Charge, Fermi-level) history of the last few attempts:
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

    def forget_attempts(self) -> None:
        """Drop the recorded attempts, but keep the slope estimate.

        The attempts belong to one geometry: once the atoms have moved, the
        potential at a given charge shifts, and a regression across the move
        would fit a line through two different systems.
        """
        self.history.clear()

    def reset_slope(self) -> None:
        """Forget the slope estimate and the attempts behind it."""
        self.slope = None
        self.history.clear()

    def _capacitance_slope(self) -> float:
        """Slope, in V/electron, of a plate capacitor of the cell's area."""
        area = abs(np.linalg.det(self.mask_r.desc.cell_cv[:2, :2])) * Bohr**2
        # Å^2 -> cm^2 is 1e16 and F -> muF is 1e6:
        return -_e * 1e22 / (area * GUESSED_CAPACITANCE)

    def update_slope(self) -> float:
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
        if self.slope is None or abs(self.slope) < 1e-10:
            # No estimate yet, or one that would send the charge to infinity:
            self.slope = self._capacitance_slope()
        return self.slope

    def next_charge(self, fermi_level: float) -> float:
        """Charge that the slope points at, given this Fermi level.

        The attempt is recorded in the history and the slope estimate is
        updated, so this must be called at most once per attempt.
        """
        self.history.append((self.charge, fermi_level))
        slope = self.update_slope()
        potential = -fermi_level * Ha
        target = self.workfunction_target * Ha
        dc = (target - potential) / slope
        if len(self.history) > 1:
            # Trust region: never step further than the previous step did.
            (c2, _), (c1, _) = list(self.history)[-2:]
            if abs(dc) > abs(c2 - c1) > 0.0:
                dc *= abs((c2 - c1) / dc)
        return self.charge + dc

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        fl1 = ibzwfs.fermi_level
        log(f'charge: {self.charge:.6f} |e|, Fermi-level: {fl1 * Ha:.3f} eV')
        fl = -self.workfunction_target
        if abs(fl1 - fl) <= self.tolerance:
            return True
        charge = self.next_charge(fl1)
        dc = charge - self.charge
        self.charge = charge
        nelectrons += dc
        ibzwfs.calculate_occs(occ_calc, nelectrons)
        mixer.reset()
        return False


def _check_depth(depth: int) -> int:
    """Validate the number of attempts to regress the slope over."""
    if depth < 2:
        raise ValueError(
            f'slope_regression_depth must be at least 2, got {depth}.  Two '
            'attempts are the fewest that a slope can be fitted to.')
    return depth


def regress_slope(attempts) -> float | None:
    """Slope of potential vs. excess electrons, in V/electron.

    ``attempts`` is a sequence of (charge, Fermi-level) pairs in atomic
    units.  Returns ``None`` if there are too few distinct charges to fit a
    line.
    """
    attempts = list(attempts)
    if len(attempts) < 2:
        return None
    charge_i = np.array([charge for charge, _ in attempts])
    potential_i = np.array([-fermi_level * Ha for _, fermi_level in attempts])
    if np.ptp(charge_i) < 1e-10:
        return None
    return float(np.polyfit(charge_i, potential_i, 1)[0])
