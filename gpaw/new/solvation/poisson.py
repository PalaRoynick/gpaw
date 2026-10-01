"""Poisson solvers for implicit solvation (a position-dependent dielectric).

For a dipole-corrected version, see ``Solvation(dipolelayer=...)``
and ``GaussianDerivativeDipole`` in ``gpaw.new.solvation``.
"""
from math import nan, pi
from typing import Any

import numpy as np
from ase.units import Ha

from gpaw import PoissonConvergenceError
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.mpi import broadcast
from gpaw.new.poisson import PoissonSolver
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.solvation.dielectric import Dielectric
from gpaw.solvation.poisson import WeightedFDPoissonSolver


def SolvationPoissonSolver(pw, grid, dielectric, **kwargs) -> PoissonSolver:
    """Poisson solver for continuum solvent systems. Returns the PW or FD
    version as appropriate; kwargs are passed on to it."""
    if isinstance(pw, PWDesc):
        return PWSolvationPoissonSolver(pw, grid, dielectric, **kwargs)
    else:
        return FDSolvationPoissonSolver(grid, dielectric, **kwargs)


class PWSolvationPoissonSolver(PWPoissonSolver):
    """Poisson solver using conjugate gradient method in reciprocal space.
    """

    def __init__(self,
                 pw: PWDesc,
                 grid: UGDesc,
                 dielectric: Dielectric,
                 rtol: float = 1e-6,
                 energytol: float = 1e-6,
                 maxiter: int = 500):
        """Initialize the conjugate gradient Poisson solver.

        Parameters
        ----------
        pw : PWDesc
            Plane wave descriptor
        grid : UGDesc
            Real-space grid on which the dielectric is defined.
        dielectric : Dielectric
            Dielectric object; eps is read from dielectric.eps_gradeps[0] at
            every solve.
        rtol : float, optional
            Relative convergence threshold for conjugate gradient algorithm
        energytol : float, optional
            Tolerance for the estimated absolute accuracy of the electrostatic
            energy, eV. Disable by setting to ``float('inf')``.
        maxiter : int, optional
            Maximum number of iterations for the conjugate gradient
            algorithm before raising ``PoissonConvergenceError``.
        """
        super().__init__(pw=pw, charge=0.)
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

        # For warm-starting the PCG solve with the previous solution.
        self._warm_cache: dict[str, Any] = {}

    def __str__(self) -> str:
        txt = ('Conjugate gradient Poisson solver:\n'
               f'  ecut: {self.pw.ecut * Ha}  # eV\n'
               f'  rtol: {self.rtol}\n'
               f'  energytol: {self.energytol}  # eV\n'
               f'  maxiter: {self.maxiter}\n')
        return txt

    def solve(self,
              vHt_g: PWArray,
              rhot_g: PWArray,
              warm_start: bool = True) -> float:
        """Solve the Poisson equation, placing the result in vHt_g.

        With warm_start, the conjugate gradient starts from the previous
        solution if the dielectric is unchanged, and this solution is kept
        for the next call.  Turn it off for a charge unrelated to the SCF
        density (such as a dipole layer's).
        """
        return self._solve(vHt_g, rhot_g, warm_start)

    def _solve(self,
               vHt_g,
               rhot_g,
               warm_start=True) -> float:
        rhot0_g = rhot_g.gather()
        vHt0_g = None
        eps_R = self.grid.from_data(self.dielectric.eps_gradeps[0])
        self.eps0_R = eps_R.gather()  # used by self._operator

        result = None  # will hold convergence information

        # Not implemented in parallel (yet).
        if self.pw.comm.rank == 0:
            vHt0_g = self.pw0.zeros()
            # Prepare the arguments for the CG. For warm starts, the x0 guess
            # comes from the previous SCF step only if eps hasn't changed.
            b_G = 4 * pi * rhot0_g.data
            b_G[0] = 0.
            Minv_g = 0.5 / self.ekin_g  # pre-conditioner
            x0 = None
            cached = self._warm_cache.get('eps')
            if warm_start and np.array_equal(self.eps0_R.data,
                                             cached):  # type: ignore
                x0 = self._warm_cache['x0']

            # Solve it.
            vHt0_g.data[:], result = self._pcg(b_G, Minv_g, x0)

        result = broadcast(result, comm=self.pw.comm)
        if not result['converged']:
            resid_ratio = result['resid_ratio']
            energy_err = result['energy_err']
            unmet = []
            if resid_ratio > self.rtol:
                unmet.append(f'|r|/|b| = {resid_ratio:.3e} > rtol = '
                             f'{self.rtol:.3e}')
            if energy_err > self.energytol:
                unmet.append(f'|<r|x>|/2 = {energy_err:.3e} eV > energytol = '
                             f'{self.energytol:.3e} eV')
            raise PoissonConvergenceError(
                f'Conjugate gradient did not converge in '
                f'{self.maxiter} iterations: ' + '; '.join(unmet))

        # Stash the results for next round's initial guess.
        if self.pw.comm.rank == 0 and warm_start:
            self._warm_cache['x0'] = vHt0_g.data.copy()  # type: ignore
            self._warm_cache['eps'] = self.eps0_R.data.copy()

        vHt_g.scatter_from(vHt0_g)
        epot = 0.5 * vHt_g.integrate(rhot_g)
        return epot

    def _operator(self, phi_G):
        """Apply the generalized Poisson operator in reciprocal space.

        Parameters
        ----------
        phi_G : ndarray
            Input potential in reciprocal space

        Returns
        -------
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
            energy = 0.5 * inner_product(r_G, x_G) / (4 * pi)
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
                         (4 * pi)) * Ha
        return x_G, {'converged': False, 'niter': niter,
                     'resid_ratio': resid_ratio, 'energy_err': energy_err}


class FDSolvationPoissonSolver(PoissonSolver):
    """Poisson solver for implicit solvents using multigrid method, real space.

    Wraps gpaw.solvation.poisson.WeightedFDPoissonSolver (shared with old GPAW)
    for new GPAW.
    """

    def __init__(self,
                 grid: UGDesc,
                 dielectric: Dielectric,
                 **kwargs):
        """Initialize the Poisson solver. kwargs are passed on to
        ``WeightedFDPoissonSolver``.

        Parameters
        ----------
        grid : UGDesc
            Real-space grid on which the dielectric is defined.
        dielectric : Dielectric
            Dielectric object; eps and its gradient are read from
            dielectric.eps_gradeps at every solve.
        """
        self.dielectric = dielectric
        self.solver = WeightedFDPoissonSolver(**kwargs)
        self.solver.set_dielectric(dielectric)
        self.solver.set_grid_descriptor(grid._gd)

    def __str__(self) -> str:
        return self.solver.get_description().rstrip()

    def solve(self, vHt_r: UGArray, rhot_r: UGArray) -> float:
        """Solve the Poisson equation, placing the result in vHt_r.

        The multigrid iteration starts from the potential already in
        vHt_r, so passing in the previous solution gives a warm start.
        Returns nan: the fd potential calculator computes the
        electrostatic energy itself.
        """
        self.solver.solve(vHt_r.data, rhot_r.data)
        return nan
