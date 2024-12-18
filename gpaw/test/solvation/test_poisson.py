import warnings

from ase.parallel import parprint
from ase.units import Bohr
from ase.io.cube import write_cube
from ase import Atoms
import numpy as np
from scipy.special import erf

from gpaw.solvation.poisson import (WeightedFDPoissonSolver,
                                    ADM12PoissonSolver,
                                    PolarizationPoissonSolver,
                                    WrappedWeightedFDPoissonSolver,
                                    WrappedPolarizationPoissonSolver,
                                    ConjugateGradientPoissonSolver)
from gpaw.pw.poisson import ReciprocalSpacePoissonSolver
from gpaw.solvation.dielectric import Dielectric
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.utilities.gauss import Gaussian
from gpaw.fd_operators import Gradient
import pytest
import matplotlib.pyplot as plt
import time

nn = 3
accuracy = 2e-10
BOX = 12.0


def gradient(gd, x, nn):
    out = gd.empty(3)
    for i in (0, 1, 2):
        Gradient(gd, i, 1.0, nn).apply(x, out[i])
    return out


def make_gd(h, box, pbc):
    diag = np.array([box] * 3)
    cell = np.diag(diag)
    grid_shape = tuple((diag / h * 2).astype(int))
    return GridDescriptor(grid_shape, cell / Bohr, pbc)


def make_pwd(ecut, box, h, pbc) -> PWDescriptor:
    diag = np.array([box] * 3)
    cell = np.diag(diag)
    grid_shape = tuple((diag / h * 2).astype(int))
    gd = GridDescriptor(grid_shape, cell / Bohr, pbc)

    return PWDescriptor(
        ecut=ecut,
        gd=gd
    )


class MockDielectric(Dielectric):
    def __init__(self, epsinf, nn):
        Dielectric.__init__(self, epsinf)
        self.nn = nn

    def update(self, cavity):
        grad_eps_vg = gradient(self.gd, self.eps_gradeps[0] - self.epsinf,
                               self.nn)
        for i in (0, 1, 2):
            self.eps_gradeps[1 + i][...] = grad_eps_vg[i]


def test_realspace_solvation_poisson():
    box = BOX
    gd = make_gd(h=.4, box=box, pbc=False)

    def solve(ps, eps, rho):
        phi = gd.zeros()
        dielectric = MockDielectric(epsinf=eps.max(), nn=nn)
        dielectric.set_grid_descriptor(gd)
        dielectric.allocate()
        dielectric.eps_gradeps[0][...] = eps
        dielectric.update(None)
        with warnings.catch_warnings():
            # ignore production code warning for alternative psolvers
            warnings.simplefilter("ignore")
            solver = ps(nn=nn, relax='J', eps=accuracy)
        solver.set_dielectric(dielectric)
        solver.set_grid_descriptor(gd)
        solver.solve(phi, rho)
        return phi

    psolvers = (WeightedFDPoissonSolver,
                ADM12PoissonSolver,
                PolarizationPoissonSolver)

    # test neutral system with constant permittivity
    parprint('neutral, constant permittivity')
    epsinf = 80.
    eps = gd.zeros()
    eps.fill(epsinf)
    qs = (-1., 1.)
    shifts = (-1., 1.)
    rho = gd.zeros()
    phi_expected = gd.zeros()
    for q, shift in zip(qs, shifts):
        gauss_norm = q / np.sqrt(4 * np.pi)
        gauss = Gaussian(gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        rho += gauss_norm * gauss.get_gauss(0)
        phi_expected += gauss_norm * gauss.get_gauss_pot(0) / epsinf

    for ps in psolvers:
        phi = solve(ps, eps, rho)
        parprint(ps, np.abs(phi - phi_expected).max())
        assert phi == pytest.approx(phi_expected, abs=1e-3)

    # test charged system with constant permittivity
    parprint('charged, constant permittivity')
    epsinf = 80.
    eps = gd.zeros()
    eps.fill(epsinf)
    q = -2.
    gauss_norm = q / np.sqrt(4 * np.pi)
    gauss = Gaussian(gd, center=(box / 2. + 1.) * np.ones(3) / Bohr)
    rho_gauss = gauss_norm * gauss.get_gauss(0)
    phi_gauss = gauss_norm * gauss.get_gauss_pot(0)
    phi_expected = phi_gauss / epsinf

    for ps in psolvers:
        phi = solve(ps, eps, rho_gauss)
        parprint(ps, np.abs(phi - phi_expected).max())
        assert phi == pytest.approx(phi_expected, abs=1e-3)

    # test non-constant permittivity
    msgs = ('neutral, non-constant permittivity',
            'charged, non-constant permittivity')
    qss = ((-1., 1.), (2., ))
    shiftss = ((-.4, .4), (-1., ))
    epsshifts = (.0, -1.)

    for msg, qs, shifts, epsshift in zip(msgs, qss, shiftss, epsshifts):
        parprint(msg)
        epsinf = 80.
        gauss = Gaussian(gd, center=(box / 2. + epsshift) * np.ones(3) / Bohr)
        eps = gauss.get_gauss(0)
        eps = epsinf - eps / eps.max() * (epsinf - 1.)

        rho = gd.zeros()
        phi_expected = gd.zeros()
        grad_eps = gradient(gd, eps - epsinf, nn)

        for q, shift in zip(qs, shifts):
            gauss = Gaussian(gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
            phi_tmp = gauss.get_gauss_pot(0)
            xyz = gauss.xyz
            fac = 2. * np.sqrt(gauss.a) * np.exp(-gauss.a * gauss.r2)
            fac /= np.sqrt(np.pi) * gauss.r2
            fac -= erf(np.sqrt(gauss.a) * gauss.r) / (gauss.r2 * gauss.r)
            fac *= 2.0 * 1.7724538509055159
            grad_phi = fac * xyz
            laplace_phi = -4. * np.pi * gauss.get_gauss(0)
            rho_tmp = -1. / (4. * np.pi) * (
                (grad_eps * grad_phi).sum(0) + eps * laplace_phi)
            norm = gd.integrate(rho_tmp)
            rho_tmp /= norm * q
            phi_tmp /= norm * q
            rho += rho_tmp
            phi_expected += phi_tmp

        for ps in psolvers:
            phi = solve(ps, eps, rho)
            parprint(ps, np.abs(phi - phi_expected).max())
            x_ax = np.array([x for x in range(int(box * 5) - 1)]) / 5
            plot_results(phi, phi_expected, x_ax, 'rspace solver', 'rspace')
            assert phi == pytest.approx(phi_expected, abs=1e-3)


def generate_two_gaussians(pd, box):
    qs = (-1., 1.)
    shifts = (-1., 1.)
    rho = pd.gd.zeros()
    for q, shift in zip(qs, shifts):
        gauss_norm = q / np.sqrt(4 * np.pi)
        gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        rho += gauss_norm * gauss.get_gauss(0)

    return rho


def generate_four_gaussians(pd, box, seed=None, min_distance=None):
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
    epsinf = 80.0
    gauss = Gaussian(gd, center=(box / 2. + epsshift) * np.ones(3) / Bohr)
    eps = gauss.get_gauss(0)
    eps = epsinf - eps / eps.max() * (epsinf - 1.0)
    return eps


def solve(ps, pd, eps, rho, pw, nn):
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
            if pw:
                cg_times.append(end - start)
            else:
                fd_times.append(end - start)
            phi_r = pd.ifft(phi)

            x_ax = np.array([x for x in range(int(box * 5))]) / 5
            plot_results(phi_r, phi_expected, x_ax, nm, tag='nonvar')

            assert phi_r == pytest.approx(phi_expected, abs=1e-3)

    # plt.scatter([x for x in boxes], cg_times, label='CG Solver')
    # plt.scatter([x for x in boxes], fd_times, label='Weighted FD Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Box Length ($\AA$)')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/length_timing.png')
    # plt.close()

    # plt.scatter([x for x in boxes], cg_times, label='CG Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Box Length ($\AA$)')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/cg_length_timing.png')
    # plt.close()

    # plt.scatter([(x * 5)**3 for x in boxes], cg_times, label='CG Solver')
    # plt.scatter([(x * 5)**3 for x in boxes],
    #             fd_times, label='Weighted FD Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Number of Grid Points')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/vol_timing.png')
    # plt.close()

    # plt.scatter([(x * 5)**3 for x in boxes], cg_times, label='CG Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Number of Grid Points')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/cg_vol_timing.png')
    # plt.close()


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

        with open('dens.cube', 'w') as f:
            write_cube(f, Atoms(), rho, [0, 0, 0])

        phi_expected = pd.zeros()
        recip_psolver = ReciprocalSpacePoissonSolver(pd)
        recip_psolver._solve(phi_expected, pd.fft(rho))
        phi_expected = pd.ifft(phi_expected) / epsinf

        cg_dummy = ConjugateGradientPoissonSolver(pd)
        dielectric = MockDielectric(epsinf=eps.max(), nn=nn)
        dielectric.set_grid_descriptor(pd.gd)
        dielectric.allocate()
        dielectric.eps_gradeps[0][...] = eps
        dielectric.update(None)
        cg_dummy.set_dielectric(dielectric)
        rho_q = cg_dummy.operator(pd.fft(phi_expected)) / 4 / np.pi

        rho_r = pd.ifft(rho_q)
        rho_q_test = pd.fft(rho_r)
        assert np.abs(rho_q - rho_q_test).max() < 1e-6

        for ps, pw, nm in psolvers:
            start = time.time()
            phi = solve(ps, pd, eps, rho_q, pw, nn)
            end = time.time()
            print(f'{nm} Timing: {end - start}')
            if pw:
                cg_times.append(end - start)
            else:
                fd_times.append(end - start)
            phi_r = pd.ifft(phi)

            x_ax = np.array([x for x in range(int(box * 5))]) / 5
            plot_results(phi_r, phi_expected, x_ax, nm)
            assert phi_r == pytest.approx(phi_expected, abs=5e-3)

    # plt.scatter([x for x in boxes], cg_times, label='CG Solver')
    # plt.scatter([x for x in boxes], fd_times, label='Weighted FD Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Box Length ($\AA$)')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/length_timing_var.png')
    # plt.close()

    # plt.scatter([x for x in boxes], cg_times, label='CG Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Box Length ($\AA$)')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/cg_length_timing_var.png')
    # plt.close()

    # plt.scatter([(x * 5)**3 for x in boxes], cg_times, label='CG Solver')
    # plt.scatter([(x * 5)**3 for x in boxes],
    #             fd_times, label='Weighted FD Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Number of Grid Points')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/vol_timing_var.png')
    # plt.close()

    # plt.scatter([(x * 5)**3 for x in boxes], cg_times, label='CG Solver')
    # plt.legend(loc='center left')
    # plt.xlabel(r'Number of Grid Points')
    # plt.ylabel(r'Convergence Time (s)')
    # plt.tight_layout()
    # plt.savefig('imgs/cg_vol_timing_var.png')
    # plt.close()


def plot_results(phi_r, phi_expected, x_ax, lbl, tag='var'):
    for i, axis in enumerate(["X", "Y", "Z"]):
        plt.scatter(x_ax, np.mean(phi_r, axis=(1, 2) if i == 0 else (0, 2) if i == 1 else (0, 1)),
                    label=lbl, alpha=0.4)
        plt.plot(x_ax, np.mean(phi_expected, axis=(1, 2) if i == 0 else (0, 2) if i == 1 else (0, 1)),
                 label='Analytical', linestyle='--', c='r')
        plt.legend(loc='center right')
        plt.xlabel(f'Box {axis} Axis' + r'($\AA$)')
        plt.ylabel(r'Avg Potential (eV)')
        plt.tight_layout()
        plt.savefig(f'imgs/{lbl[0]}_axis={i}_{tag}.png')
        plt.close()
