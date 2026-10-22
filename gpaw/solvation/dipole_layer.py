"""Gaussian-derivative dipole layer for solvated slabs, fd and pw modes."""
from abc import ABC, abstractmethod
from math import nan
from typing import Any

import numpy as np
from ase.units import Bohr, Ha

from gpaw import PoissonConvergenceError
from gpaw.core import PWArray, UGArray
from gpaw.dft import Parameter
from gpaw.fd_operators import laplace
from gpaw.mpi import broadcast, broadcast_float
from gpaw.new.poisson import PoissonSolver
from gpaw.solvation.poisson import PWSolvationPoissonSolver


class GaussianDerivativeDipole(Parameter):
    """Gaussian-derivative dipole correction for solvated slabs.

    Cancels a z-oriented dipole moment with a neutral dipole layer placed
    in the vacuum at the bottom of the cell::

        drho(z) = 4 alpha^(3/2) / sqrt(pi) * (z - center)
                  * exp(-alpha (z - center)^2),    alpha = 1 / width^2

    Each SCF step, the layer is scaled so that the xy-averaged field at the
    bottom face of the cell vanishes; since the layer is neutral, this
    flattens the potential on both sides of the slab. The layer's potential is
    solved with the same dielectric-weighted operator as the physical charge,
    so the correction is also valid for laterally varying dielectrics.
    Works in fd and pw mode.

    This object holds only the settings; the correction itself is a
    Poisson solver that wraps the solvation Poisson solver, made by build()
    once the mode is known.
    """

    def __init__(self, center: float = 1.1, width: float = 0.4,
                 zero_vacuum: bool = False):
        """
        Parameters
        ==========
        center: float
            Height (Å) of the layer above the bottom of the cell (default
            1.1 Å).
        width: float
            Width (Å) of the layer (default 0.4 Å).
        zero_vacuum: bool
            If True, set the potential in the vacuum at the top of the cell
            to zero. Otherwise uses the internal default potential reference
            from each mode.
        """
        self.center = center
        self.width = width
        self.zero_vacuum = zero_vacuum

    def build(self, ps: PoissonSolver) -> PoissonSolver:
        """Wrap the solvation Poisson solver in the correction."""
        Layer: type[_Layer]
        if isinstance(ps, PWSolvationPoissonSolver):
            Layer = _PWLayer
        else:
            Layer = _FDLayer
        return Layer(ps, center=self.center, width=self.width,
                     zero_vacuum=self.zero_vacuum)

    def todict(self) -> dict:
        return {'name': 'GaussianDerivativeDipole',
                'center': self.center,
                'width': self.width,
                'zero_vacuum': self.zero_vacuum}

    @classmethod
    def from_param(cls, dipolelayer):
        """Turn Solvation's dipolelayer input into a dipole-layer
        description, or None for no layer.

        True gives the default GaussianDerivativeDipole; False or None, no
        layer. A dict is passed as keyword arguments to
        GaussianDerivativeDipole (an optional 'name' key, as written by
        todict(), must name it). Any other object is returned as is: this
        is how to supply your own correction. It needs build(ps), which
        returns a Poisson solver wrapping the solvation Poisson solver ps,
        and todict(). That wrapper may define vacuum_levels() (see
        gpaw.new.poisson.PoissonSolver), used for the work functions, and a
        dipole_region_problem attribute (a message, or None), which the
        solvation extension warns about after the SCF.
        """
        if dipolelayer is None or dipolelayer is False:
            return None
        if dipolelayer is True:
            return cls()
        if isinstance(dipolelayer, dict):
            dct = dipolelayer.copy()
            name = dct.pop('name', 'GaussianDerivativeDipole')
            if name != 'GaussianDerivativeDipole':
                raise ValueError(f'Unknown dipole layer: {name}')
            return cls(**dct)
        if not hasattr(dipolelayer, 'build'):
            raise TypeError('dipolelayer must be a bool, a dict or an object '
                            f'with a build() method, not {dipolelayer!r}')
        return dipolelayer


class _Layer(PoissonSolver, ABC):
    """Mode-independent parts of the Gaussian-derivative dipole correction.

    This wraps the solvation Poisson solver as ``self.ps``.
    Subclasses sample the layer on the descriptor of self.ps in their
    __init__() (giving self._drho) and supply the abstract methods below.
    """

    def __init__(self, ps, center: float, width: float,
                 zero_vacuum: bool = False):
        """
        Parameters
        ==========
        ps: PoissonSolver
            The solvation Poisson solver to wrap.
        center: float
            Distance (Å) of the dipole layer from the bottom of the cell.
        width: float
            Width (Å) of the gaussian-derivative layer, 1 / sqrt(alpha).
        zero_vacuum: bool
            Shift the potential to zero in the vacuum at the top of the
            cell.
        """
        self.ps = ps
        self.center = center / Bohr
        self.width = width / Bohr
        self.alpha = self.width**-2
        self.zero_vacuum = zero_vacuum
        self._vacuum_levels = (nan, nan)
        self.dipole_region_problem: str | None = None  # see _check_region()
        self._drho: Any = None  # layer's charge (fixed)
        self._dphi: Any = None  # layer's potential (cached, unscaled)
        self._value_dip = nan  # _dphi at z=0 (cached, unscaled)
        self._slope_dip = nan  # _dphi slope at z=0 (cached, unscaled)
        self._above_dip = nan  # _dphi at center + 4 width (cached, unscaled)
        self._last_eps: Any = None  # dielectric _dphi was solved with
        self._scale = nan  # amplitude of the applied layer

    @property
    def extent(self) -> float:
        """Upper limit of the layer's charge distribution (z coordinate,
        bohr). All but exp(-16) ~ 1e-7 of the upper lobe lies below this point.
        """
        return self.center + 4 * self.width

    def _description(self) -> str:
        return (f'dipole layer: center: {self.center * Bohr:.3f}  # Å, '
                f'width: {self.width * Bohr:.3f}  # Å')

    def vacuum_levels(self, potential) -> tuple[float, float]:
        return self._vacuum_levels

    def _profile(self, z):
        """The layer's charge density at height z (bohr)."""
        z = z - self.center
        return (4 * self.alpha**1.5 / np.sqrt(np.pi) * z *
                np.exp(-self.alpha * z**2))

    def _correct(self, vHt, rhot, eps) -> None:
        """Correct the solved potential vHt in place.

        rhot is the charge density that produced vHt, which is just used
        to check for overlap with the dipole layer. The layer's own
        potential is only re-solved when the dielectric eps has changed
        since the last call.
        """
        if self._layer_calculation_needed(eps):
            self._dphi = self._solve_layer()
            levels = self._read_potential(self._dphi)
            self._value_dip, self._slope_dip, self._above_dip = levels
            self._slope_dip -= self._tail(eps)
        value, slope, above = self._read_potential(vHt)
        self._scale = slope / self._slope_dip
        top_plateau = value - self._scale * self._value_dip
        bottom_plateau = above - self._scale * self._above_dip
        shift = (top_plateau if self.zero_vacuum
                 else self._standard_shift(top_plateau, bottom_plateau))
        self._subtract_correction(vHt, self._scale, shift)
        self._vacuum_levels = (top_plateau - shift, bottom_plateau - shift)
        self._check_region(rhot, eps)

    def applied_profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The layer applied in the last correction, xy-averaged.

        Returns the heights z (Å, from the bottom of the cell), the layer's
        charge density (e/Å³, positive for positive charge) and the
        potential it added (eV, the convention of
        get_electrostatic_potential(); without the constant reference
        shift). The same on all ranks.
        """
        if self._dphi is None:
            raise RuntimeError('The dipole layer has not been applied yet.')
        z_z, drho_z, dphi_z = self._xy_averages()
        return (z_z * Bohr,
                self._scale * drho_z / Bohr**3,
                -self._scale * dphi_z * Ha)

    def _check_cell(self, pbc_c, cell_cv) -> None:
        """The layer needs a cell that is non-periodic along z, with the
        third cell vector along z and the first two perpendicular to it,
        and tall enough to hold the layer."""
        if pbc_c[2]:
            raise ValueError('The dipole-layer correction requires a cell '
                             'that is non-periodic along z.')
        tilt_c = (np.abs(cell_cv[:2, 2]) /
                  np.linalg.norm(cell_cv[:2], axis=1))
        if tilt_c.max() > 1e-10 or np.abs(cell_cv[2, :2]).max() > 1e-10:
            raise ValueError(
                'The dipole-layer correction requires the third cell vector '
                'to be along z and the first two perpendicular to it; got '
                f'cell (bohr):\n{cell_cv}')
        if self.extent >= cell_cv[2, 2]:
            raise ValueError(
                'Cell too short for the dipole layer: it extends to '
                f'z = center + 4 width = {self.extent * Bohr:.2f} Å, but the '
                f'cell is {cell_cv[2, 2] * Bohr:.2f} Å tall.')

    def _tail(self, eps) -> float:
        """Slope at z = 0 of the lower lobe that wraps around to the top of
        the cell with a periodic boundary condition; none in fd."""
        return 0.0

    def _standard_shift(self, top_plateau: float,
                        bottom_plateau: float) -> float:
        """Reference shift of the potential without zero_vacuum. By default
        none: keep the wrapped solver's reference (zero mean in pw)."""
        return 0.0

    @abstractmethod
    def solve(self, vHt, rhot) -> float:
        """Wrap the Poisson solver's solve method with the dipole correction;
        return the electrostatic energy in pw mode or nan in fd mode."""

    @abstractmethod
    def _check_region(self, rhot, eps) -> None:
        """Check the layer's surroundings; set self.dipole_region_problem
        to a message describing the problem, or None."""

    @abstractmethod
    def _layer_calculation_needed(self, new_eps) -> bool:
        """Return True if the layer's potential needs to be recalculated, based
        upon comparing the new value of eps to the stored one.
        Store the new one."""

    @abstractmethod
    def _solve_layer(self):
        """Calculate the potential, self._dphi, consistent with the layer's
        charge, self._drho, and the dielectric field."""

    @abstractmethod
    def _read_potential(self, phi) -> tuple[float, float, float]:
        """xy-averaged value and slope of the potential phi at the bottom
        face (z=0), and its value just above the layer (z≈self.extent).
        The bottom face has the same potential as the top of the cell
        (periodic in pw, zero on both faces in fd), so its value is the top
        vacuum level."""

    @abstractmethod
    def _subtract_correction(self, vHt, scale: float, shift: float) -> None:
        """Subtract scale * _dphi + shift from vHt."""

    @abstractmethod
    def _xy_averages(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Heights (bohr) at each z plane in the grid, and the associated
        xy-averages of _drho and _dphi."""


class _FDLayer(_Layer):
    """Dipole layer for the weighted finite-difference Poisson solver.

    Wraps a WeightedFDPoissonSolver (in a PoissonSolverWrapper). The cell
    must be non-periodic along z, with the potential zero on the faces.
    """

    def __init__(self, ps, center: float, width: float,
                 zero_vacuum: bool = False):
        """Sample the layer on the grid of the wrapped solver, for its
        Laplacian stencil."""
        super().__init__(ps, center, width, zero_vacuum)
        # Wrapped WeightedFDPoissonSolver uses old-style GridDescriptors.
        self.gd = gd = self.ps.solver.gd
        self.nn = nn = self.ps.solver.nn  # stencil neighbors per side
        self._check_cell(gd.pbc_c, gd.cell_cv)

        h = gd.h_cv[2, 2]  # grid spacing in z
        # Grid points along a non-periodic axis have global indices
        # 1, ..., N - 1 at z = index * h; the cell faces z = 0 and z = L are
        # the zero-potential boundaries.

        self.g_extent = int(round(self.extent / h))  # grid index at layer top

        # The lower tail of the dipole layer will always be clipped by the
        # cell boundary at z = 0 in a non-periodic cell. Two steps make the
        # correction exact anyway: the layer is rescaled to be neutral,
        # and the field is read as the discrete flux through the cell's
        # bottom face, F_{1/2} = sum_m b_m phi_m / h over planes m = 1..nn,
        # with b_m = sum_{m' >= m} a_m' from the stencil coefficients a_m.
        # Summing the stencil equation over planes 1..i telescopes to the
        # discrete Gauss law, F_{i+1/2} - F_{1/2} = -(4 pi / eps) h
        # sum_{j <= i} rho_j, so for any plane i above the neutral layer
        # F_{1/2} equals the field there, however close the layer is to the
        # face, provided that region holds no other charge and eps is
        # uniform where the flux is read (see _check_region()).

        # Create the layer's charge profile, and rescale the lower lobe to
        # enforce charge neutrality.
        self._drho = gd.zeros()  # layer charge density
        drho_z = self._profile(np.arange(1, gd.N_c[2]) * h)  # perfect layer
        lower = np.arange(1, gd.N_c[2]) * h < self.center  # mask
        drho_z[lower] *= -drho_z[~lower].sum() / drho_z[lower].sum()  # rescale
        self._drho[:] = drho_z[gd.beg_c[2] - 1:gd.end_c[2] - 1]  # domain slice

        # Planes and weights b_m of the face flux.
        a_m = np.asarray(laplace[nn], dtype=float)[1:]
        self.g_flux = np.arange(1, len(a_m) + 1)
        self.b_m = np.cumsum(a_m[::-1])[::-1] / h

        self.vcorr_g = None  # total correction subtracted in the last solve

    def __str__(self) -> str:
        return f'{self.ps}\n    {self._description()}'

    def solve(self, vHt_r: UGArray, rhot_r: UGArray) -> float:
        vHt_g = vHt_r.data
        if self.vcorr_g is not None:
            # Restore the uncorrected potential as the initial guess.
            vHt_g += self.vcorr_g
        self.ps.solve(vHt_r, rhot_r)
        self._correct(vHt_g, rhot_r.data,
                      self.ps.solver.dielectric.eps_gradeps[0])
        return nan

    def _plane_averages(self, a_g, g_z):
        """Calculate the xy-averages of a generic array a_g on the global
        z-planes with indices g_z."""
        gd = self.gd
        avg_z = np.zeros(len(g_z))
        nxy = gd.get_size_of_global_array()[:2].prod()
        for i, g in enumerate(g_z):
            if gd.beg_c[2] <= g < gd.end_c[2]:
                avg_z[i] = a_g[:, :, g - gd.beg_c[2]].sum() / nxy
        gd.comm.sum(avg_z)
        return avg_z

    def _solve_layer(self):
        # Start from the previous solution, if any.
        dphi_g = self.gd.zeros() if self._dphi is None else self._dphi
        self.ps.solver.solve(dphi_g, self._drho)
        return dphi_g

    def _read_potential(self, phi_g):
        # Potential is 0 at z=0 and its slope is the face flux.
        avg_z = self._plane_averages(phi_g, [*self.g_flux, self.g_extent])
        return 0.0, self.b_m @ avg_z[:-1], avg_z[-1]

    def _subtract_correction(self, vHt_g, scale, shift):
        self.vcorr_g = scale * self._dphi
        self.vcorr_g += shift
        vHt_g -= self.vcorr_g

    def _standard_shift(self, top_plateau, bottom_plateau):
        # Make plateaus symmetric about zero, like gpaw.dipole_correction.
        return 0.5 * (top_plateau + bottom_plateau)

    def _xy_averages(self):
        g_z = np.arange(1, self.gd.N_c[2])
        return (g_z * self.gd.h_cv[2, 2],
                self._plane_averages(self._drho, g_z),
                self._plane_averages(self._dphi, g_z))

    def _layer_calculation_needed(self, new_eps):
        eps_g = new_eps
        changed = (self._last_eps is None or
                   not np.array_equal(eps_g, self._last_eps))
        if self.gd.comm.sum_scalar(int(changed)) == 0:
            return False
        self._last_eps = eps_g.copy()
        return True

    def _check_region(self, rhot_g, eps_g):
        gd = self.gd
        h = gd.h_cv[2, 2]
        # Charge on the xy-averaged density, plane by plane (the layer's
        # step is laterally uniform).
        nxy = gd.get_size_of_global_array()[:2].prod()
        rho_z = self._plane_averages(rhot_g, range(1, self.g_extent + 1))
        charge = np.abs(rho_z).sum() * nxy * gd.dv
        # Local index i is the global plane beg_c[2] + i.
        neps = max(2 * self.nn + 1 - gd.beg_c[2], 0)
        eps_r = eps_g[:, :, :neps]
        eps_min = gd.comm.min_scalar(eps_r.min() if eps_r.size else np.inf)
        eps_max = gd.comm.max_scalar(eps_r.max() if eps_r.size else -np.inf)
        self.dipole_region_problem = _layer_region_problem(
            charge, f'the bottom {self.g_extent * h * Bohr:.2f} Å of the cell',
            eps_min, eps_max,
            f'the bottom {2 * self.nn * h * Bohr:.2f} Å of the cell')


class _PWLayer(_Layer):
    """Dipole layer for the plane-wave Poisson solver.

    Wraps a PWSolvationPoissonSolver. The cell must be declared
    non-periodic along z, although the plane waves are periodic.
    """

    def __init__(self, ps, center: float, width: float,
                 zero_vacuum: bool = False):
        super().__init__(ps, center, width, zero_vacuum)
        self.pw = self.ps.pw  # read by SlowPAWPoissonSolver
        self._check_cell(self.ps.grid.pbc_c, self.pw.cell_cv)
        Lz = self.pw.cell_cv[2, 2]
        Gz_G = self.pw.G_plus_k_Gv[:, 2]  # z-component of PW wave vector
        onaxis_G = self.pw.onaxis_mask()
        n_onaxis = self.pw.comm.sum_scalar(int(onaxis_G.sum()))
        if n_onaxis < 2:
            raise RuntimeError('Dipole layer has no on-axis plane-wave '
                               'components to correct.')
        # The layer in plane-wave space is:
        #   drho(G_z) = -2i G_z exp(-G_z^2 / (4 alpha)) exp(-i G_z center) / Lz
        self._drho = self.pw.zeros()
        self._drho.data[onaxis_G] = (
            -2j * Gz_G[onaxis_G]
            * np.exp(-Gz_G[onaxis_G]**2 / (4 * self.alpha))
            * np.exp(-1j * Gz_G[onaxis_G] * self.center) / Lz)

    def __str__(self) -> str:
        return f'{self.ps}  {self._description()}\n'

    def solve(self, vHt_g, rhot_g) -> float:
        self.ps.solve(vHt_g, rhot_g)
        eps_R = self.ps.grid.from_data(self.ps.dielectric.eps_gradeps[0])
        self._correct(vHt_g, rhot_g.gather(), eps_R.gather())
        return 0.5 * vHt_g.integrate(rhot_g)

    def _solve_layer(self):
        dphi_g = self.pw.zeros()
        try:
            self.ps.solve(dphi_g, self._drho, warm_start=False)
        except PoissonConvergenceError as error:
            msg = f'Dipole-layer solve: {error}'
            raise PoissonConvergenceError(msg) from error
        return dphi_g

    def _read_potential(self, phi_G: PWArray):
        value, slope = _xy_average_at(phi_G)
        return value, slope, _xy_average_at(phi_G, self.extent)[0]

    def _tail(self, eps0_R: UGArray | None) -> float:
        """Slope at z = 0 of the lower lobe that wraps around to the top of
        the cell."""
        # The lower tail of the dipole layer will always extend past the
        # cell boundary at z = 0 in a periodic cell, wrapping around to the
        # top of the cell. The layer is neutral as built (it has no G = 0
        # component), so unlike in fd no rescaling is needed; one step
        # corrects for the wrap: the layer's slope is read at z = 0, which
        # lies inside the layer, so it includes the field of the lobe charge
        # below z = 0, q = -2 sqrt(alpha / pi) exp(-alpha center^2) per unit
        # area. By Gauss's law that field is 4 pi |q| / eps, with eps
        # uniform at z = 0 (see _check_region()). Subtracting it leaves the
        # layer's far-field slope, to be matched against vHt's slope at z = 0.
        tail = nan
        if self.ps.grid.comm.rank == 0:
            assert eps0_R is not None  # for mypy
            tail = (8 * np.sqrt(np.pi * self.alpha) *
                    np.exp(-self.alpha * self.center**2) /
                    eps0_R.data[:, :, 0].mean())
        return broadcast_float(tail, self.ps.grid.comm)

    def _subtract_correction(self, vHt_g, scale, shift):
        vHt_g.data -= self._dphi.data * scale
        # A constant shift changes only the G=0 coefficient (on rank 0).
        if self.pw.comm.rank == 0:
            vHt_g.data[0] -= shift

    def _xy_averages(self):
        drho0_g = self._drho.gather()
        dphi0_g = self._dphi.gather()
        averages = None
        if self.ps.grid.comm.rank == 0:
            averages = [f0_g.ifft(grid=self.ps.grid0).data.mean(axis=(0, 1))
                        for f0_g in [drho0_g, dphi0_g]]
        drho_z, dphi_z = broadcast(averages, comm=self.ps.grid.comm)
        L = self.ps.grid0.cell_cv[2, 2]
        return np.arange(len(drho_z)) * L / len(drho_z), drho_z, dphi_z

    def _layer_calculation_needed(self, new_eps):
        eps0_R = new_eps
        # eps0_R is gathered on rank 0 only.
        same = False
        if self.ps.grid.comm.rank == 0:
            same = np.array_equal(eps0_R.data, self._last_eps)
            if not same:
                self._last_eps = eps0_R.data.copy()
        return not broadcast_float(same, self.ps.grid.comm)

    def _check_region(self, rhot0_g: PWArray, eps0_R: UGArray) -> None:
        # The layer also occupies the top of the cell, where its lower lobe
        # wraps around; eps must be uniform within that distance (at least
        # a few grid spacings) of z = 0 on both sides, as _tail() assumes.
        if self.pw.comm.rank != 0:
            return
        width = self.width
        top = self.extent
        wrap = max(4 * width - self.center, 0.0)
        L = self.ps.grid0.cell_cv[2, 2]
        nz = eps0_R.data.shape[2]
        h = L / nz
        z_z = np.arange(nz) * h
        charge_z = (z_z <= top) | (z_z >= L - wrap)
        reach = max(wrap, 3 * h)
        eps_z = (z_z <= reach) | (z_z >= L - reach)
        # Charge on the xy-averaged density (the layer's step is laterally
        # uniform, and plane-wave ringing averages out laterally).
        rhot_R = rhot0_g.ifft(grid=self.ps.grid0)
        nxy = rhot_R.data.shape[0] * rhot_R.data.shape[1]
        rho_z = rhot_R.data.mean(axis=(0, 1))
        charge = np.abs(rho_z[charge_z]).sum() * nxy * self.ps.grid0.dv
        eps_r = eps0_R.data[:, :, eps_z]
        where = f'the bottom {top * Bohr:.2f} Å'
        if wrap > 0:
            where += f' (and top {wrap * Bohr:.2f} Å)'
        self.dipole_region_problem = _layer_region_problem(
            charge, where + ' of the cell', eps_r.min(), eps_r.max(),
            f'{reach * Bohr:.2f} Å of the bottom/top face of the cell')


def _xy_average_at(f_G: PWArray, z: float = 0.0) -> np.ndarray:
    """Calculate the xy-averaged value and z-derivative at height z (bohr,
    from the bottom of the box)."""
    pw = f_G.desc
    mask_G = pw.onaxis_mask()
    Gz_q = pw.G_plus_k_Gv[mask_G, 2]
    f_q = f_G.data[mask_G]
    if z:
        f_q = f_q * np.exp(1j * Gz_q * z)
    value = f_q.real.sum() * 2
    # f(z) = sum_G f_G exp(iGz); using f_{-G} = conj(f_G) (f(z) real),
    # f'(z) = sum_G iG f_G exp(iGz) = -2 sum_{Gz>0} Gz Im(f_G exp(iGz)).
    derivative = -2 * Gz_q @ f_q.imag
    if pw.comm.rank == 0:
        # The G=0 term is counted once, not twice (its phase factor is 1).
        value -= f_q[0].real
    result = np.array([value, derivative])
    pw.comm.sum(result)
    return result


def _layer_region_problem(charge, charge_where, eps_min, eps_max,
                          eps_where, charge_tol=1e-3, eps_tol=1e-2):
    """Check for geometric problems with the dipole layer.

    The layer needs the region it occupies to be free of charge, and the
    dielectric to be uniform where the field is read, next to the bottom
    face of the cell. (Elsewhere in the layer the dielectric may vary: the
    layer's potential is solved with the full dielectric.) Returns a
    message for the first check that fails, or None.

    Parameters
    ==========
    charge: float
        Integral over the region charge_where of the absolute value of the
        xy-averaged charge density (electrons).
    charge_where: str
        Description of the layer's region for the message, e.g. 'the
        bottom 3.00 Å of the cell'.
    eps_min, eps_max: float
        Extremes of the dielectric in the region eps_where.
    eps_where: str
        Description, for the message, of the region where the field is
        read.
    charge_tol: float
        Largest charge (electrons) allowed in the layer's region.
    eps_tol: float
        Largest relative spread of the dielectric, (eps_max - eps_min) /
        eps_min, allowed where the field is read.
    """
    if charge > charge_tol:
        return (f'{charge:.1e} electrons of charge lie within '
                f'{charge_where}, which the dipole layer occupies; the '
                'dipole-layer correction will be inaccurate. Possible '
                'causes: (1) atoms are too close to the layer; (2) a '
                'counter charge, such as the SJM jellium, is too close to '
                'the layer; (3) in plane-wave mode, your atoms or jellium '
                'are "Gibbs ringing" into the dipole layer, which can be '
                'caused by too low of a cutoff energy or too sharp of a '
                'jellium edge.')
    if eps_max - eps_min > eps_tol * eps_min:
        return (f'The dielectric varies from {eps_min:.3f} to '
                f'{eps_max:.3f} within {eps_where}, where the dipole-layer '
                'correction reads the field and assumes it is uniform. '
                'Keep the solvent cavity away from the bottom of the '
                'cell.')
    return None
