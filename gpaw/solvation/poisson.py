import warnings

import numpy as np
from scipy.special import erf
import scipy.fft

from gpaw.poisson import BasePoissonSolver, PoissonSolver, FDPoissonSolver
from gpaw.fd_operators import Laplace, Gradient
from gpaw.wfd_operators import WeightedFDOperator
from gpaw.utilities.gauss import Gaussian
from gpaw import PoissonConvergenceError
from gpaw.utilities.timing import NullTimer
from gpaw.templogger import logg

from gpaw.pw.descriptor import PWDescriptor
from gpaw.typing import Array1D
from gpaw.pw.density import ReciprocalSpaceDensity
from gpaw.density import Density
import matplotlib.pyplot as plt


class SolvationPoissonSolver(FDPoissonSolver):
    """Base class for Poisson solvers with spatially varying dielectric.

    The Poisson equation
    div(epsilon(r) grad phi(r)) = -4 pi rho(r)
    is solved.
    """

    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 remove_moment=None, use_charge_center=False):
        if remove_moment is not None:
            raise NotImplementedError(
                'Removing arbitrary multipole moments '
                'is not implemented for SolvationPoissonSolver!')
        if nn == 'M':
            raise NotImplementedError(
                'Mehrstellen stencil is not implemented '
                'for SolvationPoissonSolver!')
        FDPoissonSolver.__init__(self, nn, relax, eps, maxiter, remove_moment,
                                 use_charge_center=use_charge_center)

    def set_dielectric(self, dielectric):
        """Set the dielectric.

        Arguments:
        dielectric -- A Dielectric instance.
        """
        self.dielectric = dielectric

    def load_gauss(self, center=None):
        """Load compensating charge distribution for charged systems.

        See Appendix B of
        A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
        """
        # XXX Check if update is needed (dielectric changed)?
        epsr, dx_epsr, dy_epsr, dz_epsr = self.dielectric.eps_gradeps
        gauss = Gaussian(self.gd, center=center)
        rho_g = gauss.get_gauss(0)
        phi_g = gauss.get_gauss_pot(0)
        x, y, z = gauss.xyz
        fac = 2. * np.sqrt(gauss.a) * np.exp(-gauss.a * gauss.r2)
        fac /= np.sqrt(np.pi) * gauss.r2
        fac -= erf(np.sqrt(gauss.a) * gauss.r) / (gauss.r2 * gauss.r)
        fac *= 2.0 * 1.7724538509055159
        dx_phi_g = fac * x
        dy_phi_g = fac * y
        dz_phi_g = fac * z
        sp = dx_phi_g * dx_epsr + dy_phi_g * dy_epsr + dz_phi_g * dz_epsr
        rho = epsr * rho_g - 1. / (4. * np.pi) * sp
        invnorm = np.sqrt(4. * np.pi) / self.gd.integrate(rho)
        self.phi_gauss = phi_g * invnorm
        self.rho_gauss = rho * invnorm


class WeightedFDPoissonSolver(SolvationPoissonSolver):
    """Weighted finite difference Poisson solver with dielectric.

    Following V. M. Sanchez, M. Sued and D. A. Scherlis,
    J. Chem. Phys. 131, 174108 (2009).
    """

    def create_laplace(self, gd, scale=1.0, n=1, dtype=float):
        operators = [Laplace(gd, scale, n, dtype)]
        operators += [Gradient(gd, j, scale, n, dtype) for j in (0, 1, 2)]
        return WeightedFDOperator(operators)

    def solve(self, phi, rho, charge=None,
              maxcharge=1e-6,
              zero_initial_phi=False, timer=None):
        self._init()

        # XXX/cb: band-aid! Currently fails.
        # if self.gd.pbc_c.all():
        #     actual_charge = self.gd.integrate(rho)
        #     if abs(actual_charge) > maxcharge:
        #         raise NotImplementedError(
        #             'charged periodic systems are not implemented')
        self.restrict_op_weights()
        ret = FDPoissonSolver.solve(self, phi, rho, charge, maxcharge,
                                    zero_initial_phi)
        return ret

    def restrict_op_weights(self):
        """Restric operator weights to coarse grids."""
        self._init()
        logg('diel eps_gradeps')
        logg(self.dielectric.eps_gradeps, header=False)
        weights = [self.dielectric.eps_gradeps] + self.op_coarse_weights
        for i, res in enumerate(self.restrictors):
            for j in range(4):
                res.apply(weights[i][j], weights[i + 1][j])
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()

    def get_description(self):
        description = SolvationPoissonSolver.get_description(self)
        return description.replace(
            'solver with',
            'weighted FD solver with dielectric and')

    def _init(self):
        if self._initialized:
            return
        self.operators[0].set_weights(self.dielectric.eps_gradeps)
        self.op_coarse_weights = []
        for operator in self.operators[1:]:
            weights = [gd.empty() for gd in (operator.gd, ) * 4]
            self.op_coarse_weights.append(weights)
            operator.set_weights(weights)
        return SolvationPoissonSolver._init(self)


class PolarizationPoissonSolver(BasePoissonSolver):
    """Poisson solver with dielectric.

    Calculates the polarization charges first using only the
    vacuum poisson equation, then solves the vacuum equation
    with polarization charges.
    """

    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 remove_moment=None, use_charge_center=False,
                 gas_phase_poisson='fast'):
        self.nn = nn
        self.eps = eps
        self.maxiter = maxiter
        self.phi_tilde = None

        self.gas_phase_poisson = PoissonSolver(
            name=gas_phase_poisson, nn=nn, eps=eps,
            remove_moment=remove_moment,
            use_charge_center=use_charge_center)

    def set_dielectric(self, dielectric):
        """Set the dielectric.

        Arguments:
        dielectric -- A Dielectric instance.
        """
        self.dielectric = dielectric

    def get_description(self):
        return ('PolarizationPoissonSolver based on '
                + self.gas_phase_poisson.get_description())

    def set_grid_descriptor(self, gd):
        self.gd = gd
        self.gas_phase_poisson.set_grid_descriptor(gd)

    def solve(self, phi, rho, charge=None,
              maxcharge=1e-6,
              zero_initial_phi=False, timer=NullTimer()):
        # get initial meaningful phi -> only do this if necessary
        if zero_initial_phi:
            niter = self.gas_phase_poisson.solve(
                phi, rho, charge=None, maxcharge=maxcharge,
                zero_initial_phi=zero_initial_phi, timer=timer)
        else:
            niter = 0

        phi_old = phi.copy()
        while niter < self.maxiter:
            rho_mod = self.rho_with_polarization_charge(phi, rho)
            niter += self.gas_phase_poisson.solve(
                phi, rho_mod, charge=None, maxcharge=maxcharge,
                zero_initial_phi=zero_initial_phi, timer=timer)
            residual = phi - phi_old
            error = self.gd.comm.sum_scalar(
                np.dot(residual.ravel(), residual.ravel())) * self.gd.dv
            if error < self.eps:
                return niter
            phi_old = phi.copy()

        raise PoissonConvergenceError(
            'PolarizationPoisson solver did not converge in '
            + f'{niter} iterations!')

    def rho_with_polarization_charge(self, phi, rho):
        epsr, dx_epsr, dy_epsr, dz_epsr = self.dielectric.eps_gradeps
        dx_phi = self.gd.empty()
        dy_phi = self.gd.empty()
        dz_phi = self.gd.empty()
        Gradient(self.gd, 0, 1.0, self.nn).apply(phi, dx_phi)
        Gradient(self.gd, 1, 1.0, self.nn).apply(phi, dy_phi)
        Gradient(self.gd, 2, 1.0, self.nn).apply(phi, dz_phi)

        scalar_product = (
            dx_epsr * dx_phi +
            dy_epsr * dy_phi +
            dz_epsr * dz_phi)

        return (rho + scalar_product / (4. * np.pi)) / epsr

    def estimate_memory(self, mem):
        # XXX estimate your own contribution
        return self.gas_phase_poisson.estimate_memory(mem)

    def todict(self):
        my_dict = self.gas_phase_poisson.todict()
        my_dict['name'] = f"polarization-{my_dict['name']}"
        return my_dict


class WrappedWeightedFDPoissonSolver(WeightedFDPoissonSolver):
    """Wrapper class for WeightedFDPoissonSolver that handles reciprocal space density.
    
    This class provides an interface between reciprocal space densities and the
    real-space weighted finite difference Poisson solver. It handles the 
    necessary Fourier transforms between real and reciprocal space to make this 
    happen.
    
    This is created just to test a potential planewave hamiltonian for 
    solvation; it allows us to benchmark finite difference and conjugate 
    gradient in the reciprocal space context.
    """

    def __init__(self, pd3, nn=3, relax='J', eps=1e-6, maxiter=10000,
                 remove_moment=None, use_charge_center=False):
        """Set up this solver instance. Primarily relies on the underlying 
        WeightedFD initialization, minus setting up the planewaver descriptor 
        for any necessary Fourier transforms.
        
        Parameters:
        -----------
        pd3 : PWDescriptor
            Plane wave descriptor for 3D transforms
        nn : int, optional
            Number of neighbors for finite difference stencil
        relax : str, optional
            Relaxation method ('J' for Jacobi)
        eps : float, optional
            Convergence threshold
        maxiter : int, optional
            Maximum number of iterations
        remove_moment : bool, optional
            Whether to remove multipole moments
        use_charge_center : bool, optional
            Whether to use charge center
        """
        self.pd3 = pd3
        WeightedFDPoissonSolver.__init__(self, nn=nn, relax=relax, eps=eps,
                                         maxiter=maxiter,
                                         remove_moment=remove_moment,
                                         use_charge_center=use_charge_center)

    def solve(self, phi, rho, charge=None, maxcharge=1e-6,
              zero_initial_phi=False, timer=None):
        """Solve the Poisson equation using WeightedFD, wrapped in Fourier 
        transforms.
        
        Parameters:
        -----------
        phi : ndarray
            Initial guess and output potential
        rho : ndarray
            Input charge density
        charge : float, optional
            Total charge constraint
        maxcharge : float, optional
            Maximum allowed charge for periodic systems
        zero_initial_phi : bool, optional
            Whether to start from zero potential
        timer : Timer, optional
            Timer for profiling
            
        Returns:
        --------
        phi : ndarray
            Solution potential
        """
        
        # move densities and potentials to the realspace grid
        rho_r = self.pd3.ifft(rho)
        phi_r = self.pd3.ifft(phi)
        phi_r = self.pd3.gd.zeros()

        # solve using the parent class
        WeightedFDPoissonSolver.solve(self, phi_r, rho_r, charge,
                                      maxcharge, zero_initial_phi)

        # move the solution back to reciprocal space
        phi_p = self.pd3.fft(phi_r)
        phi[:] = phi_p[:]

        return phi


class WrappedPolarizationPoissonSolver(PolarizationPoissonSolver):
    """Wrapper class for WeightedPolarizationPoissonSolver that handles reciprocal space density.
    
    This class provides an interface between reciprocal space densities and the
    real-space self-consistency Poisson solver as in Fisicaro, et. al. It 
    handles the necessary Fourier transforms between real and reciprocal space 
    to make this happen.
    
    This is created just to test a potential planewave hamiltonian for 
    solvation; it allows us to benchmark finite difference and conjugate 
    gradient, etc. in the reciprocal space context.
    """
    
    def __init__(self, pd3, nn=3, relax='J', eps=1e-6, maxiter=10000,
                 remove_moment=None, use_charge_center=False):
        """Set up this solver instance. Primarily relies on the underlying 
        initialization, minus setting up the planewave descriptor for any 
        necessary Fourier transforms.
        
        Parameters:
        -----------
        pd3 : PWDescriptor
            Plane wave descriptor for 3D transforms
        nn : int, optional
            Number of neighbors for finite difference stencil
        relax : str, optional
            Relaxation method ('J' for Jacobi)
        eps : float, optional
            Convergence threshold
        maxiter : int, optional
            Maximum number of iterations
        remove_moment : bool, optional
            Whether to remove multipole moments
        use_charge_center : bool, optional
            Whether to use charge center
        """
        self.pd3 = pd3
        PolarizationPoissonSolver.__init__(self, nn=nn, relax=relax, eps=eps,
                                           maxiter=maxiter,
                                           remove_moment=remove_moment,
                                           use_charge_center=use_charge_center)

    def solve(self, phi, rho, charge=None, maxcharge=1e-6,
              zero_initial_phi=False, timer=None):

        # move densities and potentials to the realspace grid
        rho_r = self.pd3.ifft(rho)
        phi_r = self.pd3.ifft(phi)
        phi_r = self.pd3.gd.zeros()

        # solve using parent class
        PolarizationPoissonSolver.solve(self, phi_r, rho_r, charge,
                                        maxcharge, zero_initial_phi)

        # back to planewave
        phi_p = self.pd3.fft(phi_r)
        phi[:] = phi_p[:]

        return phi


class ConjugateGradientSolver():
    """Generic conjugate gradient solver for linear systems (roughly equivalent 
    to e.g., a numpy implementation).
    
    The beauty of conjugate gradient is that we can use an operator oracle 
    rather than the entire matrix A (in a system Ax = b). We can leverage this 
    to express the generalized planewave Poisson operator compactly, which has 
    massive intermediates as a matrix.
    """

    def __init__(self,
                 A_fn,
                 b,
                 preconditioner=lambda x: x,
                 init=None,
                 nits_max=15,
                 threshold=1e-3):
        """Initialize the conjugate gradient solver.
        
        Parameters:
        -----------
        A_fn : callable
            Function that applies the operator A to a vector
        b : ndarray
            Right-hand side vector
        preconditioner : callable, optional
            Function that applies preconditioning
        init : ndarray, optional
            Initial guess for solution
        nits_max : int, optional
            Maximum number of iterations
        threshold : float, optional
            Convergence threshold
        """
        
        # system parameters
        self.A = A_fn
        self.b = b
        self.preconditioner = preconditioner
        if init is not None:
            self.x = init
        else:
            self.x = np.zeros_like(self.b)

        # initial residual/gradient
        self.r = self.b - self.A(self.x)
        self.z = self.preconditioner(self.r)
        self.p = self.z.copy()

        # to be computed during iterations
        self.alpha = None
        self.beta = None

        # hyperparameters
        self.nits_max = nits_max
        self.threshold = threshold
        self.r_norms = []

    def update_alpha(self):
        """Update the step size alpha using current residual and conjugate 
        direction.
        """
        self.alpha = (self.rz) / np.vdot(self.p, self.Ap)

    def update_beta(self):
        """Update the direction update parameter beta based on residual growth.
        """
        new_rz = np.vdot(self.r, self.z)
        self.beta = (new_rz) / (self.rz)
        self.rz = new_rz

    def solve(self):
        """Execute the conjugate gradient algorithm.
        
        Returns:
        --------
        x : ndarray
            Solution vector
        """
        i = 0

        self.rz = np.vdot(self.r, self.z)
        update_norm = float('inf')

        while (np.linalg.norm(self.r) / np.linalg.norm(self.b) > self.threshold
               and update_norm > self.threshold
               and i < self.nits_max):
            #    and (len(self.r_norms) < 2 or self.r_norms[-1]
            # < self.r_norms[-2])):
            self.Ap = self.A(self.p)

            # update guess
            self.update_alpha()
            dx = self.alpha * self.p
            self.x = self.x + dx
            update_norm = np.linalg.norm(
                dx) / max(np.linalg.norm(self.x), 1e-10)

            # explicitly recompute residual
            self.r = self.r - self.alpha * self.Ap
            self.z = self.preconditioner(self.r)

            # choose next direction conjugate to r_0
            self.update_beta()
            self.p = self.z + self.beta * self.p
            self.r_norms.append(
                np.linalg.norm(self.r) / np.linalg.norm(self.b))
            i += 1
        return self.x


class ConjugateGradientPoissonSolver():
    """Poisson solver using conjugate gradient method in reciprocal space.
    
    This solver works directly in reciprocal space and is particularly efficient
    for systems with periodic boundary conditions.
    """
    
    def __init__(self,
                 pd: PWDescriptor,
                 charge: float = 0.0,
                 strength: float = 1.0):
        """Initialize the conjugate gradient Poisson solver.
        
        Parameters:
        -----------
        pd : PWDescriptor
            Plane wave descriptor
        charge : float, optional
            Total charge of the system
        strength : float, optional
            Scaling factor for the potential
        """
        
        self.pd = pd
        self.charge = charge
        self.strength = strength

        self.G2_q = pd.G2_qG[0].copy()
        if pd.gd.comm.rank == 0:
            # Avoid division by zero:
            self.G2_q[0] = 1.0

    def set_dielectric(self, dielectric):
        self.dielectric = dielectric

    def __str__(self):
        return f'Uniform background charge: {self.charge:.3f} electrons'

    def get_description(self):
        return 'Conjugate Gradient Poisson Solver'

    def estimate_memory(self, mem):
        pass

    def operator(self, phi_q):
        """Apply the generalized Poisson operator in reciprocal space. This refers to the "A_fn" we'll use in the conjugate gradient algorithm.
        
        Parameters:
        -----------
        phi_q : ndarray
            Input potential in reciprocal space
            
        Returns:
        --------
        ndarray
            Result of operator application
        """
        
        G_Qv = self.pd.get_reciprocal_vectors()
        Gx, Gy, Gz = G_Qv.T

        # gradient of phi
        grad_x, grad_y, grad_z = [x * phi_q for x in [Gx, Gy, Gz]]

        # move to realspace
        gx_real, gy_real, gz_real = [self.pd.ifft(x)
                                     for x in [grad_x, grad_y, grad_z]]

        # apply dielectric
        epsg_x_r, epsg_y_r, epsg_z_r = [x * self.dielectric.eps_gradeps[0]
                                        for x in [gx_real, gy_real, gz_real]]

        # take divergence
        epsg_x, epsg_y, epsg_z = [self.pd.fft(x)
                                  for x in [epsg_x_r, epsg_y_r, epsg_z_r]]

        return np.sum(
            [Gx * epsg_x, Gy * epsg_y, Gz * epsg_z], axis=0)

    def solve(self,
              vHt_q: Array1D,
              dens: ReciprocalSpaceDensity) -> float:
        """Solve Poisson equeation.

        Places result in vHt_q ndarray.
        """
        assert dens.rhot_q is not None
        epot = self._solve(vHt_q, dens.rhot_q)
        return epot

    def _solve(self,
               vHt_q: Array1D,
               rhot_q: Array1D) -> float:

        vHt_q[:] = 4 * np.pi * self.strength * rhot_q
        if self.pd.gd.comm.rank == 0:
            # use uniform backgroud charge in case we have a charged system:
            vHt_q[0] = 0.0

        def precon(vec):
            return vec / self.G2_q

        self.cg_solv = ConjugateGradientSolver(self.operator, vHt_q,
                                               preconditioner=precon)
        vHt_q[:] = self.cg_solv.solve()

        epot = 0.5 * self.pd.integrate(vHt_q, rhot_q)
        return epot
    
    
class RealConjugateGradientPoissonSolver(BasePoissonSolver):
    """Poisson solver using conjugate gradient method in real space.
    
    This solver works directly in real space with arbitrarily mixed Dirichlet/
    periodic boundary conditions
    """
    
    def __init__(self, nn=3, relax='J', eps=1e-3, maxiter=15,
                 remove_moment=None, use_charge_center=False):
        self.nn = nn
        self.eps = eps
        self.maxiter = maxiter

    def get_description(self):
        return 'Real Space Conjugate Gradient Poisson Solver'

    def estimate_memory(self, mem):
        pass
    
    def set_dielectric(self, dielectric):
        """Set the dielectric.

        Arguments:
        dielectric -- A Dielectric instance.
        """
        self.dielectric = dielectric
        
    def set_grid_descriptor(self, gd):
        self.gd = gd
        self.pbc = gd.pbc_c
        self.L = np.array([np.linalg.norm(gd.h_cv[:, i]) * gd.N_c[i] for i in range(3)])
        laplace_coefs = self.build_laplace()
        self.laplace_coefs = np.where(abs(laplace_coefs) < 1e-10, 1.0, laplace_coefs)
        
    def stransform(self, f):
        result = f.copy()
        ndims = f.ndim
        for ax in range(ndims):
            if self.pbc[ax]:
                result = scipy.fft.fft(result, axis=ax, norm='ortho')
            else:
                result = scipy.fft.dst(result, type=2, axis=ax, norm='ortho')
        return result

    def istransform(self, F):
        result = F.copy()
        ndims = F.ndim
        for ax in range(ndims):
            if self.pbc[ax]:
                result = scipy.fft.ifft(result, axis=ax, norm='ortho')
            else:
                result = scipy.fft.dst(result, type=3, axis=ax, norm='ortho')
        return result

    def ctransform(self, f, grad_dim):
        result = f.copy()
        ndims = f.ndim
        for ax in range(ndims):
            if self.pbc[ax]:
                result = scipy.fft.fft(result, axis=ax, norm='ortho')
            else:
                if ax == grad_dim:
                    result = scipy.fft.dct(result, type=2, axis=ax, norm='ortho')
                else:
                    result = scipy.fft.dst(result, type=2, axis=ax, norm='ortho')
        return result

    def ictransform(self, F, grad_dim):
        result = F.copy()
        ndims = F.ndim
        for ax in range(ndims):
            if self.pbc[ax]:
                result = scipy.fft.ifft(result, axis=ax, norm='ortho')
            else:
                if ax == grad_dim:
                    result = scipy.fft.dct(result, type=3, axis=ax, norm='ortho')
                else:
                    result = scipy.fft.dst(result, type=3, axis=ax, norm='ortho')
        return result
    
    def build_laplace(self):
        full_shape = self.gd.N_c
        effective_shape = []
        for ax in range(len(full_shape)):
            if self.pbc[ax]:
                effective_shape.append(full_shape[ax])
            else:
                effective_shape.append(full_shape[ax] - 1)
        effective_shape = tuple(effective_shape)
        ndims = len(effective_shape)
        coefs = np.zeros(effective_shape, dtype=float)
        for ax in range(ndims):
            n = effective_shape[ax]
            L_axis = self.L[ax]
            shape_ax = [1] * ndims
            shape_ax[ax] = n
            if self.pbc[ax]:
                k = 2 * np.pi * np.fft.fftfreq(n, d=L_axis/n)
            else:
                k = (np.arange(n) + 1) * np.pi / L_axis
            coefs += (k.reshape(shape_ax)**2)
        return coefs

    def operator(self, phi_tilde):
        ndims = phi_tilde.ndim
        grad_components = []
        for ax in range(ndims):
            n = phi_tilde.shape[ax]
            L_axis = self.L[ax]
            shape = [1] * ndims
            shape[ax] = n
            if self.pbc[ax]:
                k = 2 * np.pi * np.fft.fftfreq(n, d=L_axis / n)
                multiplier = (k).reshape(shape)
            else:
                k = (np.arange(n) + 1) * np.pi / L_axis
                multiplier = k.reshape(shape)
            grad_components.append(phi_tilde * multiplier)
            
        grad_real = [self.ictransform(comp, i) for i, comp in enumerate(grad_components)]
        
        grad_real_mod = [g * self.dielectric.eps_gradeps[0] for g in grad_real]
        
        grad_tilde_mod = [self.ctransform(g, i) for i, g in enumerate(grad_real_mod)]
        divergence_tilde = 0
        
        for ax in range(ndims):
            n = phi_tilde.shape[ax]
            L_axis = self.L[ax]
            shape = [1] * ndims
            shape[ax] = n
            if self.pbc[ax]:
                k = 2 * np.pi * np.fft.fftfreq(n, d=L_axis / n)
                multiplier = (k).reshape(shape)
            else:
                k = (np.arange(n) + 1) * np.pi / L_axis
                multiplier = k.reshape(shape)
            divergence_tilde += grad_tilde_mod[ax] * multiplier
            
        return divergence_tilde

    def solve(self,
              phi: Array1D,
              rho: Density) -> float:
        """Solve Poisson equeation.

        Places result in vHt_q ndarray.
        """
        assert rho is not None
        epot = self._solve(phi, rho)
        return epot

    def _solve(self,
               phi: Array1D,
               rho: Array1D) -> float:

        phi[:] = 4 * np.pi * rho
        
        phi_tilde = self.stransform(phi)

        def precon(vec):
            return vec / self.laplace_coefs

        self.cg_solv = ConjugateGradientSolver(self.operator, phi_tilde,
                                               preconditioner=precon, nits_max=15)
        
        phi[:] = self.istransform(self.cg_solv.solve())

        epot = 0.5 * self.gd.integrate(phi, rho)
        return epot


class WrappedConjugateGradientPoissonSolver(ConjugateGradientPoissonSolver):
    def __init__(self, pd3: PWDescriptor, charge: float = 0.0,
                 strength: float = 1.0, nn=3, relax='J', eps=1e-6):
        self.pd3 = pd3
        self.nn = nn
        self.relax = relax
        self.eps = eps
        ConjugateGradientPoissonSolver.__init__(self, pd3, charge, strength)

    def solve(self, phi, rho, charge=None, maxcharge=1e-6,
              zero_initial_phi=False, timer=None):

        rho_q = self.pd3.fft(rho)

        phi_q = self.pd3.ifft(phi)

        ConjugateGradientPoissonSolver.solve(self, phi_q, rho_q, charge,
                                             maxcharge, zero_initial_phi)

        phi_p = self.pd3.fft(phi_q)

        phi[:] = phi_p[:]

        return phi

    def set_grid_descriptor(self, gd):
        self.gd = gd


class ADM12PoissonSolver(SolvationPoissonSolver):
    """Poisson solver with dielectric.

    Following O. Andreussi, I. Dabo, and N. Marzari,
    J. Chem. Phys. 136, 064102 (2012).

    Warning: Not intended for production use, as it is not tested
             thouroughly!

    XXX TODO : * Correction for charged systems???
               * Check: Can the polarization charge introduce a monopole?
               * Convergence problems depending on eta. Apparently this
                 method works best with FFT as in the original Paper.
               * Optimize numerics.
    """

    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 remove_moment=None, eta=.6, use_charge_center=False):
        """Constructor for ADM12PoissonSolver.

        Additional arguments not present in SolvationPoissonSolver:
        eta -- linear mixing parameter
        """
        adm12_warning = UserWarning(
            'ADM12PoissonSolver is not tested thoroughly'
            ' and therefore not recommended for production code!')
        warnings.warn(adm12_warning)
        self.eta = eta
        SolvationPoissonSolver.__init__(
            self, nn, relax, eps, maxiter, remove_moment,
            use_charge_center=use_charge_center)

    def set_grid_descriptor(self, gd):
        SolvationPoissonSolver.set_grid_descriptor(self, gd)
        self.gradx = Gradient(gd, 0, 1.0, self.nn)
        self.grady = Gradient(gd, 1, 1.0, self.nn)
        self.gradz = Gradient(gd, 2, 1.0, self.nn)

    def get_description(self):
        if len(self.operators) == 0:
            return 'uninitialized ADM12PoissonSolver'
        else:
            description = SolvationPoissonSolver.get_description(self)
            return description.replace(
                'solver with',
                'ADM12 solver with dielectric and')

    def _init(self):
        if self._initialized:
            return
        self.rho_iter = self.gd.zeros()
        self.d_phi = self.gd.empty()
        return SolvationPoissonSolver._init(self)

    def solve(self, phi, rho, charge=None,
              maxcharge=1e-6,
              zero_initial_phi=False, timer=None):
        self._init()
        if self.gd.pbc_c.all():
            actual_charge = self.gd.integrate(rho)
            if abs(actual_charge) > maxcharge:
                raise NotImplementedError(
                    'charged periodic systems are not implemented')
        return FDPoissonSolver.solve(
            self, phi, rho, charge, maxcharge, zero_initial_phi,
            timer=timer)

    def solve_neutral(self, phi, rho, timer=None):
        self._init()
        self.rho = rho
        return SolvationPoissonSolver.solve_neutral(self, phi, rho)

    def iterate2(self, step, level=0):
        self._init()
        if level == 0:
            epsr, dx_epsr, dy_epsr, dz_epsr = self.dielectric.eps_gradeps
            self.gradx.apply(self.phis[0], self.d_phi)
            sp = dx_epsr * self.d_phi
            self.grady.apply(self.phis[0], self.d_phi)
            sp += dy_epsr * self.d_phi
            self.gradz.apply(self.phis[0], self.d_phi)
            sp += dz_epsr * self.d_phi
            self.rho_iter = self.eta / (4. * np.pi) * sp + \
                (1. - self.eta) * self.rho_iter
            self.rhos[0][:] = (self.rho_iter + self.rho) / epsr
        return SolvationPoissonSolver.iterate2(self, step, level)
