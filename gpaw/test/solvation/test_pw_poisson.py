import warnings

from ase.parallel import parprint
from ase.units import Bohr
import numpy as np

from gpaw.solvation.poisson import (WeightedFDPoissonSolver,
                                    ADM12PoissonSolver,
                                    PolarizationPoissonSolver,
                                    ConjugateGradientPoissonSolver)
from gpaw.pw.poisson import ReciprocalSpacePoissonSolver
from gpaw.solvation.dielectric import Dielectric
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.utilities.gauss import Gaussian
from gpaw.fd_operators import Gradient
import pytest
import time

nn = 3
accuracy = 2e-10
BOX = 12.0

def gradient(gd, x, nn):
    """
    Compute the gradient of a function.
    
    Parameters:
    ----------
    gd : GridDescriptor
        Grid descriptor
    x : np.ndarray
        Function
    nn : int
        Number of grid points

    Returns:
    --------
    out : np.ndarray
        Gradient
    """
    out = gd.empty(3)
    for i in (0, 1, 2):
        Gradient(gd, i, 1.0, nn).apply(x, out[i])
    return out

def make_gd(h, box, pbc):
    """
    Make a GridDescriptor for testing.

    Parameters:
    ----------
    h : float
        Grid spacing
    box : float
        Box size
    pbc : bool
        Boundary conditions

    Returns:
    --------
    gd : GridDescriptor
        Grid descriptor
    """
    
    diag = np.array([box] * 3)
    cell = np.diag(diag)
    grid_shape = tuple((diag / h * 2).astype(int))
    return GridDescriptor(grid_shape, cell / Bohr, pbc)


def make_pwd(ecut, box, h, pbc) -> PWDescriptor:
    """
    Make a PWDescriptor for testing.

    Parameters:
    ----------
    ecut : float
        Cut-off energy
    box : float
        Box size
    h : float
        Grid spacing
    pbc : bool
        Boundary conditions

    Returns:
    --------
    pwd : PWDescriptor
        Plane wave descriptor
    """
    
    diag = np.array([box] * 3)
    cell = np.diag(diag)
    grid_shape = tuple((diag / h * 2).astype(int))
    gd = GridDescriptor(grid_shape, cell / Bohr, pbc)

    return PWDescriptor(
        ecut=ecut,
        gd=gd
    )

class MockDielectric(Dielectric):
    """
    Mock dielectric class for testing.
    
    Parameters:
    ----------
    epsinf : float
        Permittivity
    nn : int
        Number of grid points
    """
    
    def __init__(self, epsinf, nn):
        Dielectric.__init__(self, epsinf)
        self.nn = nn

    def update(self, cavity):
        grad_eps_vg = gradient(self.gd, self.eps_gradeps[0] - self.epsinf,
                               self.nn)
        for i in (0, 1, 2):
            self.eps_gradeps[1 + i][...] = grad_eps_vg[i]


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

def generate_two_gaussians(pd, box):
    """
    Generate two Gaussian charges with different shifts.
    
    Parameters:
    -----------
    pd : PWDescriptor
        Plane wave descriptor for 3D transforms
    box : float
        Box size

    Returns:
    ---------
    rho : np.ndarray
        Charge density for testing
    """
    qs = (-1., 1.)
    shifts = (-1., 1.)
    rho = pd.gd.zeros()
    for q, shift in zip(qs, shifts):
        gauss_norm = q / np.sqrt(4 * np.pi)
        gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        rho += gauss_norm * gauss.get_gauss(0)

    return rho


def generate_four_gaussians(pd, box, seed=None, min_distance=None):
    """
    Generate four Gaussian charges with different shifts.
    
    Parameters:
    -----------
    pd : PWDescriptor
        Plane wave descriptor for 3D transforms
    box : float
        Box size
    seed : int, optional
        Seed for random number generator
    min_distance : float, optional
        Minimum distance between charges

    Returns:
    ---------
    rho : np.ndarray
        Charge density for testing
    """
    qs = (-1., 1., -1., 1.)
    min_distance = min_distance or box / 8

    rng = np.random.default_rng(seed)
    shifts = []

    while len(shifts) < 4:
        candidate = rng.uniform(box / 4, 3 * box / 4, size=3)
        if all(np.linalg.norm(candidate - np.array(shift)) >= min_distance for shift in shifts):
            shifts.append(candidate)

    rho = pd.gd.zeros()
    for q, shift in zip(qs, shifts):
        gauss_norm = q / np.sqrt(4 * np.pi)
        gauss = Gaussian(pd.gd, center=np.array(shift) / Bohr)
        rho += gauss_norm * gauss.get_gauss(0)

    return rho


def gauss_dielectric(gd, box, epsshift=0.):
    """
    Generate a dielectric function with a Gaussian peak; example of some nice 
    but variable dielectric.

    Parameters:
    ----------
    gd : GridDescriptor
        Grid descriptor
    box : float
        Box size
    epsshift : float, optional
        Shift of the Gaussian peak

    Returns:
    --------
    eps : np.ndarray
        Dielectric function for testing
    """
    
    epsinf = 80.0
    gauss = Gaussian(gd, center=(box / 2. + epsshift) * np.ones(3) / Bohr)
    eps = gauss.get_gauss(0)
    eps = epsinf - eps / eps.max() * (epsinf - 1.0)
    return eps


def solve(ps, pd, eps, rho, pw, nn):
    """
    Solves the Poisson equation using a given Poisson solver, grid, and charge 
    density input.

    Parameters:
    ----------
    ps : PoissonSolver
        Poisson solver
    pd : PwDescriptor
        Pw descriptor
    eps : np.ndarray
        Dielectric function
    rho : np.ndarray
        Charge density
    pw : bool
        Whether or not the solver is designed for planewave densities
    nn : int
        Number of nodes

    Returns:
    --------
    phi : np.ndarray
        Solution of the Poisson equation
    """
    
    phi = pd.zeros()
    dielectric = MockDielectric(epsinf=eps.max(), nn=nn)
    dielectric.set_grid_descriptor(pd.gd)
    dielectric.allocate()
    dielectric.eps_gradeps[0][...] = eps
    dielectric.update(None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        args = [pd]
        kwargs = {}
        solver = ps(*args, **kwargs)
    if not pw:
        solver.set_grid_descriptor(pd.gd)
    solver.set_dielectric(dielectric)
    if pw:
        solver._solve(phi, rho)
    else:
        solver.solve(phi, rho)
    return phi


@pytest.mark.parametrize("density_function", [
    generate_two_gaussians,
    lambda pd, box: generate_four_gaussians(pd, box, seed=0),
    lambda pd, box: generate_four_gaussians(pd, box, seed=1),
    lambda pd, box: generate_four_gaussians(pd, box, seed=2),
])
def test_planewave_const_permittivity(density_function):
    """Test the planewave Poisson solver against realspace versions for 
    constant permittivity."""
    cg_times = []
    fd_times = []
    boxes = [BOX]

    for box in boxes:
        pd = make_pwd(ecut=20.0, box=box, h=0.4, pbc=[True, True, True])

        psolvers = [(ConjugateGradientPoissonSolver, True, 'CG Solver'),
                    (WrappedWeightedFDPoissonSolver, False, 'Weighted FD Solver'),
                    (WrappedPolarizationPoissonSolver, False, 'Polarization Solver')]

        epsinf = 80.
        eps = pd.gd.zeros()
        eps.fill(epsinf)
        rho = density_function(pd, box)

        # compute the potential we expect given the charge density, WITHOUT
        # the variable dielectric (basically analytical for constant perm.)
        # serves as a reference
        phi_expected = pd.zeros()
        recip_psolver = ReciprocalSpacePoissonSolver(pd)
        recip_psolver._solve(phi_expected, pd.fft(rho))
        phi_expected = pd.ifft(phi_expected) / epsinf

        rho_q = pd.fft(rho[0])
        rho_r = pd.ifft(rho_q)
        rho_q_test = pd.fft(rho_r)
        assert np.abs(rho_q - rho_q_test).max() < 1e-6

        for ps, pw, nm in psolvers:
            rho_q = pd.fft(rho)
            start = time.time()
            phi = solve(ps, pd, eps, rho_q, pw, nn)
            end = time.time()
            parprint(f'{nm} Timing: {end - start}')
            phi_r = pd.ifft(phi)

            assert phi_r == pytest.approx(phi_expected, abs=1e-3)


@pytest.mark.parametrize("dielectric_function", [
    gauss_dielectric
])
@pytest.mark.parametrize("density_function", [
    generate_two_gaussians,
    lambda pd, box: generate_four_gaussians(pd, box, seed=0),
    lambda pd, box: generate_four_gaussians(pd, box, seed=1),
    lambda pd, box: generate_four_gaussians(pd, box, seed=2),
])
def test_planewave_var_permittivity(dielectric_function, density_function):
    """Test the planewave Poisson solver against realspace versions for 
    variable permittivity."""
    cg_times = []
    fd_times = []
    boxes = [BOX]

    for box in boxes:

        pd = make_pwd(ecut=None, box=box, h=0.4, pbc=[True, True, True])
        psolvers = [(ConjugateGradientPoissonSolver, True, 'CG Solver'),
                    (WrappedWeightedFDPoissonSolver, False, 'Weighted FD Solver'),
                    (WrappedPolarizationPoissonSolver, False, 'Polarization Solver')]

        epsinf = 80.
        eps = dielectric_function(pd.gd, box)

        rho = density_function(pd, box)

        # compute the potential we expect given the charge density, WITHOUT
        # the variable dielectric
        phi_expected = pd.zeros()
        recip_psolver = ReciprocalSpacePoissonSolver(pd)
        recip_psolver._solve(phi_expected, pd.fft(rho))
        phi_expected = pd.ifft(phi_expected) / epsinf

        # compute the charge density that would have given us the potential we
        # just computed, if we had included the variable dielectric
        # this makes the potential we just computed the correct solution
        # for the planewave Poisson solver under the variable dielectric
        cg_dummy = ConjugateGradientPoissonSolver(pd)
        dielectric = MockDielectric(epsinf=eps.max(), nn=nn)
        dielectric.set_grid_descriptor(pd.gd)
        dielectric.allocate()
        dielectric.eps_gradeps[0][...] = eps
        dielectric.update(None)
        cg_dummy.set_dielectric(dielectric)
        
        # use the forward operator to copmute the charge density that implies
        # phi_expected under variable dielectric
        rho_q = cg_dummy.operator(pd.fft(phi_expected)) / 4 / np.pi

        rho_r = pd.ifft(rho_q)
        rho_q_test = pd.fft(rho_r)
        assert np.abs(rho_q - rho_q_test).max() < 1e-6

        for ps, pw, nm in psolvers:
            start = time.time()
            phi = solve(ps, pd, eps, rho_q, pw, nn)
            end = time.time()
            print(f'{nm} Timing: {end - start}')
            phi_r = pd.ifft(phi)
            assert phi_r == pytest.approx(phi_expected, abs=5e-3)