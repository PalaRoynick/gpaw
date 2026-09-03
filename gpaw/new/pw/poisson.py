from functools import cached_property
from math import pi
from typing import Any

import numpy as np
from ase.units import Bohr, Ha
from scipy.special import erf

from gpaw import PoissonConvergenceError
from gpaw.core import PWArray, PWDesc, UGDesc
from gpaw.mpi import broadcast, broadcast_float
from gpaw.new.poisson import PoissonSolver


def make_poisson_solver(pw: PWDesc,
                        grid: UGDesc,
                        charge: float,
                        strength: float = 1.0,
                        dipolelayer: bool = False,
                        **kwargs) -> PoissonSolver:
    if charge != 0.0 and not grid.pbc_c.any():
        return ChargedPWPoissonSolver(pw, grid, charge, strength, **kwargs)

    ps = PWPoissonSolver(pw, charge, strength)

    if dipolelayer:
        return DipoleLayerPWPoissonSolver(ps, grid, **kwargs)

    assert not kwargs

    return ps


class PWPoissonSolver(PoissonSolver):
    def __init__(self,
                 pw: PWDesc,
                 charge: float = 0.0,
                 strength: float = 1.0):
        self.pw = pw
        self.charge = charge
        self.strength = strength

        self.ekin_g = pw.ekin_G.copy()
        if pw.comm.rank == 0:
            # Avoid division by zero:
            self.ekin_g[0] = 1.0

    def __str__(self) -> str:
        txt = ('poisson solver:\n'
               f'  ecut: {self.pw.ecut * Ha}  # eV\n')
        if self.strength != 1.0:
            txt += f'  strength: {self.strength}\n'
        if self.charge != 0.0:
            txt += f'  uniform background charge: {self.charge}  # electrons\n'
        return txt

    def solve(self,
              vHt_g: PWArray,
              rhot_g: PWArray) -> float:
        """Solve Poisson equation.

        Places result in vHt_g ndarray.
        """
        epot = self._solve(vHt_g, rhot_g)
        return epot

    def _solve(self,
               vHt_g,
               rhot_g) -> float:
        vHt_g.data[:] = 2 * pi * self.strength * rhot_g.data
        if self.pw.comm.rank == 0:
            # Use uniform background charge in case we have a charged system:
            vHt_g.data[0] = 0.0
        if not isinstance(self.ekin_g, vHt_g.xp.ndarray):
            self.ekin_g = vHt_g.xp.array(self.ekin_g)
        vHt_g.data /= self.ekin_g
        epot = 0.5 * vHt_g.integrate(rhot_g)
        return epot


class ChargedPWPoissonSolver(PWPoissonSolver):
    def __init__(self,
                 pw: PWDesc,
                 grid: UGDesc,
                 charge: float,
                 strength: float = 1.0,
                 alpha: float = None,
                 eps: float = 1e-5):
        """Reciprocal-space Poisson solver for charged molecules.

        * Add a compensating Gaussian-shaped charge to the density
          in order to make the total charge neutral (placed in the
          middle of the unit cell

        * Solve Poisson equation.

        * Correct potential so that it has the correct 1/r
          asymptotic behavior

        * Correct energy to remove the artificial interaction with
          the compensation charge

        Parameters
        ----------
        pw: PWDesc
        grid: UGDesc
        charge: float
        strength: float
        alpha: float
        eps: float

        Attributes
        ----------
        alpha : float
        charge_g : np.ndarray
            Gaussian-shaped charge in reciprocal space
        potential_g : PWArray
             Potential in reciprocal space created by charge_g
        """
        super().__init__(pw, charge, strength)

        if alpha is None:
            # Shortest distance from center to edge of cell:
            rcut = 0.5 / (pw.icell**2).sum(axis=1).max()**0.5

            # Make sure e^(-alpha*rcut^2)=eps:
            alpha = -rcut**-2 * np.log(eps)

        self.alpha = alpha

        center_v = pw.cell_cv.sum(axis=0) / 2
        G2_g = 2 * pw.ekin_G
        G_gv = pw.G_plus_k_Gv
        self.charge_g = np.exp(-1 / (4 * alpha) * G2_g +
                               1j * (G_gv @ center_v))
        self.charge_g *= charge / pw.dv

        R_Rv = grid.xyz()
        d_R = ((R_Rv - center_v)**2).sum(axis=3)**0.5
        potential_R = grid.empty()

        # avoid division by 0
        zero_indx = d_R == 0
        d_R[zero_indx] = 1
        potential_R.data[:] = charge * erf(alpha**0.5 * d_R) / d_R
        # at zero we should have:
        # erf(alpha**0.5 * d_R) / d_R = alpha**0.5 * 2 / sqrt(pi)
        potential_R.data[zero_indx] = charge * alpha**0.5 * 2 / np.sqrt(pi)
        self.potential_g = potential_R.fft(pw=pw)

    def __str__(self) -> str:
        txt, x, _ = super().__str__().rsplit('\n', 2)
        assert x.startswith('  uniform background charge:')
        txt += (
            '\n  # using Gaussian-shaped compensation charge: e^(-alpha r^2)\n'
            f'  alpha: {self.alpha}   # bohr^-2')
        return txt

    def _solve(self,
               vHt_g,
               rhot_g) -> float:
        xp = vHt_g.xp
        if self.potential_g.xp is not xp:
            self.ekin_g = xp.array(self.ekin_g)
            self.charge_g = xp.array(self.charge_g)
            self.potential_g = self.potential_g.to_xp(xp)

        neutral_g = rhot_g.copy()
        neutral_g.data += self.charge_g

        if neutral_g.desc.comm.rank == 0:
            error = neutral_g.data[0]  # * self.pd.gd.dv
            assert error.imag == 0.0, error
            assert abs(float(error.real)) < 0.00001, error
            neutral_g.data[0] = 0.0

        vHt_g.data[:] = 2 * pi * neutral_g.data
        vHt_g.data /= self.ekin_g
        epot = 0.5 * vHt_g.integrate(neutral_g)
        epot -= self.potential_g.integrate(rhot_g)
        epot -= self.charge**2 * (self.alpha / 2 / pi)**0.5
        vHt_g.data -= self.potential_g.data
        return epot


class DipoleLayerPWPoissonSolver(PoissonSolver):
    def __init__(self,
                 ps: PWPoissonSolver,
                 grid: UGDesc,
                 width: float = 1.0,  # Ångström
                 zero_vacuum=False):
        self.ps = ps
        self.grid = grid
        self.width = width / Bohr
        self.zero_vacuum = zero_vacuum
        assert grid.pbc_c.sum() == 2
        self.axis = np.where(~grid.pbc_c)[0][0]
        self.correction = np.nan
        self.pw = ps.pw

    def solve(self,
              vHt_g: PWArray,
              rhot_g: PWArray) -> float:
        epot = self.ps.solve(vHt_g, rhot_g)
        dip_v = -rhot_g.moment()
        c = self.axis
        L = self.grid.cell_cv[c, c]
        self.correction = 2 * np.pi * dip_v[c] * L / self.grid.volume
        vHt_g.data -= 2 * self.correction * self.sawtooth_g.data
        if self.zero_vacuum:
            v0 = vHt_g.boundary_value(self.axis)
            if vHt_g.desc.comm.rank == 0:
                vHt_g.data[0] += self.correction - v0
        return epot + 2 * np.pi * dip_v[c]**2 / self.grid.volume

    def dipole_layer_correction(self) -> float:
        return self.correction

    @cached_property
    def sawtooth_g(self) -> PWArray:
        grid = self.grid
        if grid.comm.rank == 0:
            c = self.axis
            L = grid.cell_cv[c, c]
            w = self.width / 2
            assert w < L / 2, (w, L, c)
            gc = int(w / L * grid.size_c[c])
            x = np.linspace(0, L, grid.size_c[c], endpoint=False)
            sawtooth = x / L - 0.5
            a = 1 / L - 0.75 / w
            b = 0.25 / w**3
            sawtooth[:gc] = x[:gc] * (a + b * x[:gc]**2)
            sawtooth[-gc:] = -sawtooth[gc:0:-1]
            sawtooth_r = grid.new(comm=None).empty()
            shape = [1, 1, 1]
            shape[c] = -1
            sawtooth_r.data[:] = sawtooth.reshape(shape)
            sawtooth_g = sawtooth_r.fft(pw=self.ps.pw.new(comm=None)).data
        else:
            sawtooth_g = None

        result_g = self.ps.pw.empty()
        result_g.scatter_from(sawtooth_g)
        return result_g


class ConjugateGradientPoissonSolver(PWPoissonSolver):
    """Poisson solver using conjugate gradient method in reciprocal space.
    """

    def __init__(self,
                 pw: PWDesc,
                 grid,
                 dielectric,
                 charge: float = 0.0,
                 strength: float = 1.0,
                 rtol=1e-6,
                 energytol=1e-6,
                 maxiter=500,
                 dipolelayer=False,
                 zero_vacuum=False):
        """Initialize the conjugate gradient Poisson solver.

        Parameters:
        -----------
        pw : PWDesc
            Plane wave descriptor
        charge : float, optional
            Total charge of the system
        strength : float, optional
            Scaling factor for the potential
        rtol : float, optional
            Relative convergence threshold for conjugate gradient algorithm
        energytol : float, optional
            Tolerance for the estimated absolute accuracy of the electrostatic
            energy, eV. Disable by setting to `float('inf')`.
        maxiter : int, optional
            Maximum number of iterations for the conjugate gradient
            algorithm before raising ``PoissonConvergenceError``.
        dipolelayer : bool, optional
            Appply a dipole layer to correct for z-oriented dipole moments,
            resulting in two zero-slope regions of potential in the vacuum.
        zero_vacuum : bool, optional
            Zero the potential at the z=0 vacuum position; only used if a
            dipolelayer is employed.
        """
        super().__init__(pw, charge, strength)
        self.dielectric = dielectric
        self.grid = grid
        self.pw0 = pw.new(comm=None)  # serial copy
        self.grid0 = grid.new(comm=None)  # serial copy
        if pw.comm.rank == 0:
            self.ekin_g = self.pw0.ekin_G.copy()
            self.ekin_g[0] = 1.0

        self.rtol = rtol
        self.energytol = energytol
        self.maxiter = maxiter

        self.dipolelayer = dipolelayer
        self.zero_vacuum = zero_vacuum
        if zero_vacuum and not self.dipolelayer:
            raise NotImplementedError('zero_vacuum not available if '
                                      'dipolelayer is False.')
        # For caching the dipole-layer solve.
        self._dipole_cache: dict[str, Any] = {}

        # For warm-starting the PCG solve with the previous solution.
        self._warm_cache: dict[str, Any] = {}

    def __str__(self) -> str:
        txt = ('conjugate gradient poisson solver:\n'
               f'  ecut: {self.pw.ecut * Ha}  # eV\n'
               f'  rtol: {self.rtol}\n'
               f'  energytol: {self.energytol}  # eV\n'
               f'  maxiter: {self.maxiter}\n')
        if self.strength != 1.0:
            txt += f'  strength: {self.strength}\n'
        if self.charge != 0.0:
            txt += f'  uniform background charge: {self.charge}  # electrons\n'
        return txt

    def _solve(self,
               vHt_g,
               rhot_g,
               dipole_solve=False) -> float:
        rhot0_g = rhot_g.gather()
        vHt0_g = None
        eps_R = self.grid.from_data(self.dielectric.eps_gradeps[0])
        self.eps0_R = eps_R.gather()  # used by self._operator

        result = None  # will hold convergence information

        # Not implemented in parallel (yet).
        if self.pw.comm.rank == 0:
            vHt0_g = self.pw0.zeros()
            # Prepare the arguments for the CG. The initial guess x0 is re-used
            # from the previous SCF step if eps hasn't changed.
            b_G = 4 * pi * self.strength * rhot0_g.data
            b_G[0] = 0.
            Minv_g = 0.5 / self.ekin_g  # pre-conditioner
            x0 = None
            cached = self._warm_cache.get('eps')
            if np.array_equal(self.eps0_R.data, cached):  # type: ignore
                x0 = self._warm_cache['x0']
            if dipole_solve:
                x0 = None

            # Solve it.
            vHt0_g.data[:], result = self._pcg(b_G, Minv_g, x0)

        result = broadcast(result, comm=self.pw.comm)
        if not result['converged']:
            resid_ratio = result['resid_ratio']
            energy_err = result['energy_err']
            label = 'dipole-layer' if dipole_solve else 'main'
            unmet = []
            if resid_ratio > self.rtol:
                unmet.append(f'|r|/|b| = {resid_ratio:.3e} > rtol = '
                             f'{self.rtol:.3e}')
            if energy_err > self.energytol:
                unmet.append(f'|<r|x>|/2 = {energy_err:.3e} eV > energytol = '
                             f'{self.energytol:.3e} eV')
            raise PoissonConvergenceError(
                f'Conjugate gradient ({label} solve) did not converge in '
                f'{self.maxiter} iterations: ' + '; '.join(unmet))

        # Stash the results for next round's initial guess.
        if self.pw.comm.rank == 0 and not dipole_solve:
            self._warm_cache['x0'] = vHt0_g.data.copy()  # type: ignore
            self._warm_cache['eps'] = self.eps0_R.data.copy()

        vHt_g.scatter_from(vHt0_g)
        if self.dipolelayer and not dipole_solve:
            self._correct_dipole(vHt_g)
        epot = 0.5 * vHt_g.integrate(rhot_g)
        return epot

    def _operator(self, phi_G):
        """Apply the generalized Poisson operator in reciprocal space.

        Parameters:
        -----------
        phi_q : ndarray
            Input potential in reciprocal space

        Returns:
        --------
        ndarray
            Result of operator application
        """
        pw = self.pw0
        grid = self.grid0
        G_vG = pw.G_plus_k_Gv.T

        ophi_G = np.zeros_like(phi_G)
        for G_G in G_vG:
            grad_G = pw.from_data(G_G * phi_G * 1j)
            grad_R = grad_G.ifft(grid=grid)
            grad_R.data *= self.eps0_R.data
            ophi_G -= grad_R.fft(pw=pw).data * G_G * 1j

        return ophi_G

    def _correct_dipole(self, vHt_g: PWArray) -> None:
        """Apply the dipole-layer slope correction, and (if requested)
        the vacuum-zeroing shift, to an already-solved potential field (vHt_g),
        in place. Uses a gaussian-derivative, epsilon(r)-aware dipole layer.
        Caches the dipole solution and re-uses it if the epsilon shape hasn't
        changed.
        """
        _, v_deriv = xy_average_at_boundary(vHt_g)
        # The vacuum is zeroed at the top of the cell, where SJM references
        # its electrode potential -- the same convention as the FD solver
        # (see gpaw.new.electrocatalysis.sjm.SJMPoissonSolver).  In the
        # plane-wave representation z=0 and z=L are the same point, so "the
        # top" is the last point of the real-space grid, z = L - h.
        z_top = self.grid.cell_cv[2, 2] * (1.0 - 1.0 / self.grid.size_c[2])
        v_top, _ = xy_average_at(vHt_g, z_top)

        # Check if a valid dipole layer solution is cached.
        matches_cache = False
        if self.grid.comm.rank == 0:
            cached = self._dipole_cache.get('eps0')
            matches_cache = np.array_equal(self.eps0_R.data,
                                           cached)  # type: ignore
        matches_cache = broadcast_float(matches_cache, self.grid.comm)

        if matches_cache:
            dphi_g = self._dipole_cache['dphi_g']
            v_deriv_dip = self._dipole_cache['v_deriv_dip']
            v_top_dip = self._dipole_cache['v_top_dip']
        else:
            # Create a gaussian-derivative dipole layer and solve.
            drho_g, alpha, z0 = create_dipole_layer(self.pw)
            dphi_g = self.pw.zeros()
            self._solve(dphi_g, drho_g, dipole_solve=True)
            _, v_deriv_dip = xy_average_at_boundary(dphi_g)
            v_top_dip, _ = xy_average_at(dphi_g, z_top)
            # The slope (v_deriv_dip) is slightly contaminated by the dipole
            # layer itself; we analytically correct for this below.
            tail = np.nan
            if self.grid.comm.rank == 0:
                tail = (8 * np.sqrt(np.pi * alpha) * np.exp(-alpha * z0**2) /
                        self.eps0_R.data[:, :, 0].mean())
            tail = broadcast_float(tail, self.grid.comm)
            v_deriv_dip -= tail
            # Cache to avoid duplicate solutions.
            self._dipole_cache['dphi_g'] = dphi_g
            self._dipole_cache['v_deriv_dip'] = v_deriv_dip
            self._dipole_cache['v_top_dip'] = v_top_dip
            if self.grid.comm.rank == 0:
                self._dipole_cache['eps0'] = self.eps0_R.data.copy()

        # Correct the slope.
        vHt_g.data -= dphi_g.data * (v_deriv / v_deriv_dip)
        if self.zero_vacuum:
            # The G=0 coefficient lives only on rank 0.
            if self.pw.comm.rank == 0:
                vHt_g.data[0] -= v_top - (v_deriv / v_deriv_dip) * v_top_dip

    def _pcg(self, b_G, Minv_g, x0=None):
        """Preconditioned conjugate gradient in reciprocal space.

        Uses ``PWArray.integrate()`` as the inner product. The generalized
        Poisson operator is self-adjoint and positive definite, consistent with
        conjugate gradient's requirements.

        Parameters
        ----------
        b_G : ndarray
            Right-hand side coefficients.
        Minv_g : ndarray
            Diagonal elements of preconditioner, in reciprocal space.
        x0 : ndarray, optional
            Initial guess (warm start). ``None`` means start from zero.

        Notes
        -----

        Converges when both the residual is less than self.rtol and the energy
        is less than self.energytol. Note the energy tolerance test is
        leading-order only.

        Returns
        -------
        x_G, result : ndarray, dict
            ``result['converged']`` is ``False`` if ``maxiter`` was hit
            without meeting the combined criterion, in which case
            ``result['resid_ratio']`` (``|r|/|b|``) and
            ``result['energy_err']`` (``|<r|x>|/2`` in eV) are the true,
            freshly recomputed diagnostics for the abort message.
            ``result['niter']`` is the iteration count in either case.
        """
        def inner_product(a_G, c_G) -> float:
            p = self.pw0.from_data(a_G).integrate(self.pw0.from_data(c_G)).real
            return p

        def norm(a_G):
            return np.sqrt(inner_product(a_G, a_G))

        def is_converged(r_G, x_G):
            """Checks both residual and energy."""
            if norm(r_G) > threshold:
                return False
            energy = 0.5 * inner_product(r_G, x_G) / (4 * pi * self.strength)
            return abs(energy) * Ha <= self.energytol

        threshold = self.rtol * norm(b_G)

        if x0 is None:
            x_G = np.zeros_like(b_G)
            r_G = b_G.copy()
        else:
            x_G = np.array(x0, copy=True)
            r_G = b_G - self._operator(x_G)

        # The main PCG loop.
        niter = 0
        converged = is_converged(r_G, x_G)
        rz_old = 1.
        p_G = np.zeros_like(r_G)
        while not converged and niter < self.maxiter:
            niter += 1
            z_G = Minv_g * r_G  # pre-conditioned residual
            rz_new = inner_product(r_G, z_G)  # pre-conditioned residual norm
            beta = rz_new / rz_old  # blending coefficient
            p_G = z_G + beta * p_G  # search direction
            rz_old = rz_new  # pre-conditioned residual norm
            Ap_G = self._operator(p_G)
            alpha = rz_old / inner_product(p_G, Ap_G)  # step size
            x_G = x_G + alpha * p_G  # potential
            r_G = r_G - alpha * Ap_G  # residual
            if is_converged(r_G, x_G):
                # r_G can have some drift in it; calculate fresh to be sure.
                r_G = b_G - self._operator(x_G)  # residual
                converged = is_converged(r_G, x_G)

        x_G[0] = 0.0  # average potential is arbitrary; set to 0

        if converged:
            return x_G, {'converged': True, 'niter': niter}

        # We apparently did not converge, but have only checked the CG-updated
        # residual. Check with true residual before failing.
        r_G = b_G - self._operator(x_G)
        if is_converged(r_G, x_G):
            return x_G, {'converged': True, 'niter': niter}

        # Give failure diagnostics.
        bn = norm(b_G)
        rn = norm(r_G)
        resid_ratio = rn / bn if bn else rn
        energy_err = abs(0.5 * inner_product(r_G, x_G) /
                         (4 * pi * self.strength)) * Ha
        return x_G, {'converged': False, 'niter': niter,
                     'resid_ratio': resid_ratio, 'energy_err': energy_err}


def create_dipole_layer(pw: PWDesc, z0: float = 2.0):
    """Gaussian-derivative dipole layer, built in plane-wave space.

    Creates a gaussian derivative dipole layer to be used in dipole
    correction schemes. In real space, this is
    rho(z) = (4 alpha^(3/2)/sqrt(pi)) * (z-z0) * exp(-alpha * (z-z0)^2)
    where alpha = 6 / z0^2
    In plane wave, the Fourier-transformed form is:
    rho(G_z) = -2i * G_z * exp(-G_z^2/(4*alpha)) * exp(-i * G_z * z0)
    PWArray's plane-wave coefficients are normalized by the cell volume
    (see UGDesc.fft()'s docstring); for an on-axis mode (Gx=Gy=0) that
    reduces the above continuous transform by a factor of the cell's
    z-length, Lz.
    """
    alpha = 6. / z0**2
    Lz = pw.cell_cv[2, 2]
    Gz_G = pw.G_plus_k_Gv[:, 2]  # z-component of PW wave vector
    onaxis_G = (pw.G_plus_k_Gv[:, 0] == 0) & (pw.G_plus_k_Gv[:, 1] == 0)
    drho_g = pw.zeros()
    drho_g.data[onaxis_G] = (
        -2j * Gz_G[onaxis_G]
        * np.exp(-Gz_G[onaxis_G]**2 / (4 * alpha))
        * np.exp(-1j * Gz_G[onaxis_G] * z0) / Lz)
    return drho_g, alpha, z0


def xy_average_at(f_G: PWArray, z: float = 0.0) -> np.ndarray:
    """Calculate the xy-averaged value and z-derivative at height ``z``.

    ``z`` is measured in Bohr from the lower boundary of the box.
    """
    pw = f_G.desc
    m0_G, m1_G = pw.indices_cG[:2, pw.ng1:pw.ng2] == 0
    mask_G = m0_G & m1_G
    Gz_q = pw.G_plus_k_Gv[mask_G, 2]
    f_q = f_G.data[mask_G]
    if z:
        f_q = f_q * np.exp(1j * Gz_q * z)
    value = f_q.real.sum() * 2
    # f(z) = sum_G f_G exp(iGz); using f_{-G} = conj(f_G) (f(z) real),
    # f'(z) = sum_G iG f_G exp(iGz) = -2 sum_{Gz>0} Gz Im(f_Gz exp(iGz z)).
    derivative = -2 * Gz_q @ f_q.imag
    if pw.comm.rank == 0:
        # The G=0 term is counted once, not twice (its phase factor is 1):
        value -= f_q[0].real
    result = np.array([value, derivative])
    pw.comm.sum(result)
    return result


def xy_average_at_boundary(f_G: PWArray) -> np.ndarray:
    """Calculate average value and derivative at boundary of box."""
    return xy_average_at(f_G, 0.0)
