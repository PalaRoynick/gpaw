import warnings

from ase.parallel import parprint
from ase.units import Bohr
import numpy as np
from scipy.special import erf

from gpaw.solvation.poisson import (WeightedFDPoissonSolver,
                                    ADM12PoissonSolver,
                                    PolarizationPoissonSolver,
                                    WrappedWeightedFDPoissonSolver,
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


def test_solvation_poisson():
    box = 12.
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
        plt.scatter(range(59), np.mean(phi_expected, axis=(1, 2)))
        plt.savefig('imgs/rexp_axis=0.png')
        plt.close()
        plt.scatter(range(59), np.mean(phi_expected, axis=(0, 2)))
        plt.savefig('imgs/rexp_axis=1.png')
        plt.close()
        plt.scatter(range(59), np.mean(phi_expected, axis=(0, 1)))
        plt.savefig('imgs/rexp_axis=2.png')
        plt.close()
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
            assert phi == pytest.approx(phi_expected, abs=1e-3)


def test_pw_solvation_poisson():
    box = 12.0
    pd = make_pwd(ecut=20.0, box=box, h=0.4, pbc=[True, True, True])

    def solve(ps, eps, rho, pw=False):
        phi = pd.zeros()
        dielectric = MockDielectric(epsinf=eps.max(), nn=nn)
        dielectric.set_grid_descriptor(pd.gd)
        dielectric.allocate()
        dielectric.eps_gradeps[0][...] = eps
        dielectric.update(None)
        with warnings.catch_warnings():
            # ignore production code warning for alternative psolvers
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

    psolvers = [(WrappedWeightedFDPoissonSolver, False), (ConjugateGradientPoissonSolver, True)]

    # test neutral system with constant permittivity
    parprint('neutral, constant permittivity')
    epsinf = 80.
    eps = pd.gd.zeros()
    eps.fill(epsinf)
    qs = (-1., 1.)
    shifts = (-1., 1.)
    rho = pd.gd.zeros()
    phi_expected = pd.gd.zeros()
    for q, shift in zip(qs, shifts):
        gauss_norm = q / np.sqrt(4 * np.pi)
        gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        rho += gauss_norm * gauss.get_gauss(0)
        phi_expected += gauss_norm * gauss.get_gauss_pot(0) / epsinf

    rho_q = pd.fft(rho)
    rho_r = pd.ifft(rho_q)
    rho_q_test = pd.fft(rho_r)
    parprint("Transformation error (rho):", np.abs(rho_q - rho_q_test).max())
    for ps, pw in psolvers:
        start = time.time()
        phi = solve(ps, eps, rho_q, pw)
        end = time.time()
        parprint(f'{pw} Timing: {end - start}')
        phi_r = pd.ifft(phi)

        phi_r -= np.mean(phi_r)
        phi_expected -= np.mean(phi_expected)
        # assert phi_r == pytest.approx(phi_expected, abs=1e-3)

    # test non-constant permittivity
    msgs = ('neutral, non-constant permittivity',
            'charged, non-constant permittivity')
    qss = ((-1., 1.), (2., ))
    shiftss = ((-.4, .4), (-1., ))
    epsshifts = (.0, -1.)

    for msg, qs, shifts, epsshift in zip(msgs, qss, shiftss, epsshifts):
        parprint(msg)
        epsinf = 80.
        gauss = Gaussian(pd.gd, center=(box / 2. + epsshift) * np.ones(3) / Bohr)
        eps = gauss.get_gauss(0)
        eps = epsinf - eps / eps.max() * (epsinf - 1.)

        rho = pd.gd.zeros()
        phi_expected = pd.gd.zeros()
        grad_eps = gradient(pd.gd, eps - epsinf, nn)

        for q, shift in zip(qs, shifts):
            gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
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
            norm = pd.gd.integrate(rho_tmp)
            rho_tmp /= norm * q
            phi_tmp /= norm * q
            rho += rho_tmp
            phi_expected += phi_tmp

        rho_q = pd.fft(rho)
        rho_r = pd.ifft(rho_q)
        rho_q_test = pd.fft(rho_r)
        parprint("Transformation error (rho):", np.abs(rho_q - rho_q_test).max())
        for ps, pw in psolvers:
            start = time.time()
            phi = solve(ps, eps, rho_q, pw)
            end = time.time()
            parprint(f'{pw} Timing: {end - start}')
            phi_r = pd.ifft(phi)

            phi_r -= np.mean(phi_r)
            phi_expected -= np.mean(phi_expected)

            # assert phi_r == pytest.approx(phi_expected, abs=1e-3)
    assert False


def test_constant_permittivity_neutral():
    box = 12.0
    pd = make_pwd(ecut=20.0, box=box, h=0.4, pbc=[True, True, True])

    def solve(ps, eps, rho, pw=False):
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

    psolvers = [(WrappedWeightedFDPoissonSolver, False), (ConjugateGradientPoissonSolver, True)]

    epsinf = 80.
    eps = pd.gd.zeros()
    eps.fill(epsinf)
    qs = (-1., 1.)
    shifts = (-1., 1.)
    rho = pd.gd.zeros()
    phi_expected = pd.gd.zeros()
    for q, shift in zip(qs, shifts):
        gauss_norm = q / np.sqrt(4 * np.pi)
        gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        rho += gauss_norm * gauss.get_gauss(0)
        phi_expected += gauss_norm * gauss.get_gauss_pot(0) / epsinf

    rho_q = pd.fft(rho)
    rho_r = pd.ifft(rho_q)
    rho_q_test = pd.fft(rho_r)
    assert np.abs(rho_q - rho_q_test).max() < 1e-6
    for ps, pw in psolvers:
        start = time.time()
        phi = solve(ps, eps, rho_q, pw)
        end = time.time()
        parprint(f'{pw} Timing: {end - start}')
        phi_r = pd.ifft(phi)

        plt.scatter(range(60), np.mean(phi_r, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_axis=0.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_r, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_axis=1.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_r, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_axis=2.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=0.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=1.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_exp_axis=2.png')
        plt.close()

        assert phi_r == pytest.approx(phi_expected, abs=1e-3)


def test_non_constant_permittivity_neutral():
    box = 12.0
    pd = make_pwd(ecut=20.0, box=box, h=0.4, pbc=[True, True, True])

    def solve(ps, eps, rho, pw=False):
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

    psolvers = [(WrappedWeightedFDPoissonSolver, False), (ConjugateGradientPoissonSolver, True)]

    epsinf = 80.
    gauss = Gaussian(pd.gd, center=(box / 2. + 0.0) * np.ones(3) / Bohr)
    eps = gauss.get_gauss(0)
    eps = epsinf - eps / eps.max() * (epsinf - 1.)

    rho = pd.gd.zeros()
    phi_expected = pd.gd.zeros()
    grad_eps = gradient(pd.gd, eps - epsinf, nn)

    qs = (-1., 1.)
    shifts = (-.4, .4)
    for q, shift in zip(qs, shifts):
        gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        phi_tmp = gauss.get_gauss_pot(0)
        xyz = gauss.xyz
        fac = 2. * np.sqrt(gauss.a) * np.exp(-gauss.a * gauss.r2)
        fac /= np.sqrt(np.pi) * gauss.r2
        fac -= erf(np.sqrt(gauss.a) * gauss.r) / (gauss.r2 * gauss.r)
        fac *= 2.0 * 1.7724538509055159
        grad_phi = fac * xyz
        laplace_phi = -4. * np.pi * gauss.get_gauss(0)
        rho_tmp = -1. / (4. * np.pi) * ((grad_eps * grad_phi).sum(0) + eps * laplace_phi)
        norm = pd.gd.integrate(rho_tmp)
        rho_tmp /= norm * q
        phi_tmp /= norm * q
        rho += rho_tmp
        phi_expected += phi_tmp

    rho_q = pd.fft(rho)
    rho_r = pd.ifft(rho_q)
    rho_q_test = pd.fft(rho_r)
    assert np.abs(rho_q - rho_q_test).max() < 1e-6
    for ps, pw in psolvers:
        start = time.time()
        phi = solve(ps, eps, rho_q, pw)
        end = time.time()
        parprint(f'{pw} Timing: {end - start}')
        phi_r = pd.ifft(phi)

        plt.scatter(range(60), np.mean(phi_r, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_axis=0.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_r, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_axis=1.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_r, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_axis=2.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=0.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=1.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_exp_axis=2.png')
        plt.close()

        assert phi_r == pytest.approx(phi_expected, abs=1e-3)


def test_non_constant_permittivity_charged():
    box = 12.0
    pd = make_pwd(ecut=20.0, box=box, h=0.4, pbc=[True, True, True])

    def solve(ps, eps, rho, pw=False):
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

    psolvers = [(WrappedWeightedFDPoissonSolver, False), (ConjugateGradientPoissonSolver, True)]

    epsinf = 80.
    gauss = Gaussian(pd.gd, center=(box / 2. - 1.0) * np.ones(3) / Bohr)
    eps = gauss.get_gauss(0)
    eps = epsinf - eps / eps.max() * (epsinf - 1.)

    rho = pd.gd.zeros()
    phi_expected = pd.gd.zeros()
    grad_eps = gradient(pd.gd, eps - epsinf, nn)

    qs = (2.,)
    shifts = (-1.,)
    for q, shift in zip(qs, shifts):
        gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        phi_tmp = gauss.get_gauss_pot(0)
        xyz = gauss.xyz
        fac = 2. * np.sqrt(gauss.a) * np.exp(-gauss.a * gauss.r2)
        fac /= np.sqrt(np.pi) * gauss.r2
        fac -= erf(np.sqrt(gauss.a) * gauss.r) / (gauss.r2 * gauss.r)
        fac *= 2.0 * 1.7724538509055159
        grad_phi = fac * xyz
        laplace_phi = -4. * np.pi * gauss.get_gauss(0)
        rho_tmp = -1. / (4. * np.pi) * ((grad_eps * grad_phi).sum(0) + eps * laplace_phi)
        norm = pd.gd.integrate(rho_tmp)
        rho_tmp /= norm * q
        phi_tmp /= norm * q
        rho += rho_tmp
        phi_expected += phi_tmp

    rho_q = pd.fft(rho)
    rho_r = pd.ifft(rho_q)
    rho_q_test = pd.fft(rho_r)
    assert np.abs(rho_q - rho_q_test).max() < 1e-6
    for ps, pw in psolvers:
        start = time.time()
        phi = solve(ps, eps, rho_q, pw)
        end = time.time()
        parprint(f'{pw} Timing: {end - start}')
        phi_r = pd.ifft(phi)

        plt.scatter(range(60), np.mean(phi_r, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_axis=0.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_r, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_axis=1.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_r, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_axis=2.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=0.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=1.png')
        plt.close()
        plt.scatter(range(60), np.mean(phi_expected, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_exp_axis=2.png')
        plt.close()

        assert phi_r == pytest.approx(phi_expected, abs=1e-3)

    assert False

@pytest.fixture
def setup_poisson_solver():
    box = BOX
    pd = make_pwd(ecut=None, box=box, h=0.4, pbc=[True, True, True])
    psolvers = [(ConjugateGradientPoissonSolver, True)]
    # psolvers = [(ConjugateGradientPoissonSolver, True),(WrappedWeightedFDPoissonSolver, False)]
    return pd, psolvers

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

def random_clusters(gd, box):
    clusters = np.random.choice([60.0, 100.0], size=gd.N_c)
    for _ in range(5):
        clusters = (clusters + np.roll(clusters, 1, axis=0) +
                    np.roll(clusters, -1, axis=0) +
                    np.roll(clusters, 1, axis=1) +
                    np.roll(clusters, -1, axis=1) +
                    np.roll(clusters, 1, axis=2) +
                    np.roll(clusters, -1, axis=2)) / 7
    return clusters

@pytest.mark.parametrize("dielectric_function", [
    lambda gd, box: np.full(gd.N_c, 80.0),
    lambda gd, box: Gaussian(gd, center=(box / 2) * np.ones(3) / Bohr).get_gauss(0),
    # lambda gd, box: np.sin(np.linspace(0, 2 * np.pi, gd.N_c[0]))[:, None, None] + 80.0,
    random_clusters,
])
def test_dielectric_variants(setup_poisson_solver, dielectric_function):
    pd, psolvers = setup_poisson_solver
    box = BOX

    gauss = Gaussian(pd.gd, center=(box / 2. + 0.0) * np.ones(3) / Bohr)
    eps = dielectric_function(pd.gd, box)
    eps = 80.0 - eps / eps.max() * (80.0 - 1.)

    rho = pd.gd.zeros()
    phi_expected = pd.gd.zeros()
    grad_eps = gradient(pd.gd, eps - 80.0, nn)

    qs = (-1., 1.)
    shifts = (-.4, .4)
    for q, shift in zip(qs, shifts):
        gauss = Gaussian(pd.gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        phi_tmp = gauss.get_gauss_pot(0)
        xyz = gauss.xyz
        fac = 2. * np.sqrt(gauss.a) * np.exp(-gauss.a * gauss.r2)
        fac /= np.sqrt(np.pi) * gauss.r2
        fac -= erf(np.sqrt(gauss.a) * gauss.r) / (gauss.r2 * gauss.r)
        fac *= 2.0 * 1.7724538509055159
        grad_phi = fac * xyz
        laplace_phi = -4. * np.pi * gauss.get_gauss(0)
        rho_tmp = -1. / (4. * np.pi) * ((grad_eps * grad_phi).sum(0) + eps * laplace_phi)
        norm = pd.gd.integrate(rho_tmp)
        rho_tmp /= norm * q
        phi_tmp /= norm * q
        rho += rho_tmp
        phi_expected += phi_tmp

    cg_dummy = ConjugateGradientPoissonSolver(pd)
    dielectric = MockDielectric(epsinf=eps.max(), nn=nn)
    dielectric.set_grid_descriptor(pd.gd)
    dielectric.allocate()
    dielectric.eps_gradeps[0][...] = eps
    dielectric.update(None)
    cg_dummy.set_dielectric(dielectric)
    rho_q = cg_dummy.operator(pd.fft(phi_expected))

    rho_r = pd.ifft(rho_q)
    rho_q_test = pd.fft(rho_r)
    assert np.abs(rho_q - rho_q_test).max() < 1e-6

    for ps, pw in psolvers:
        start = time.time()
        phi = solve(ps, pd, eps, rho_q, pw, nn)
        end = time.time()
        print(f'{pw} Timing: {end - start}')

        phi_r = pd.ifft(phi)

        plt.scatter(range(int(BOX*5)), np.mean(phi_r, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_axis=0.png')
        plt.close()
        plt.scatter(range(int(BOX*5)), np.mean(phi_r, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_axis=1.png')
        plt.close()
        plt.scatter(range(int(BOX*5)), np.mean(phi_r, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_axis=2.png')
        plt.close()
        plt.scatter(range(int(BOX*5)), np.mean(phi_expected, axis=(1, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=0.png')
        plt.close()
        plt.scatter(range(int(BOX*5)), np.mean(phi_expected, axis=(0, 2)))
        plt.savefig(f'imgs/{pw}_exp_axis=1.png')
        plt.close()
        plt.scatter(range(int(BOX*5)), np.mean(phi_expected, axis=(0, 1)))
        plt.savefig(f'imgs/{pw}_exp_axis=2.png')
        plt.close()
        assert phi_r == pytest.approx(phi_expected, abs=1e-3)



    
# plt.scatter(range(180), np.mean(phi_r, axis=(1, 2)))
# plt.savefig(f'{pw}_axis=0.png')
# plt.close()
# plt.scatter(range(180), np.mean(phi_r, axis=(0, 2)))
# plt.savefig(f'{pw}_axis=1.png')
# plt.close()
# plt.scatter(range(180), np.mean(phi_r, axis=(0, 1)))
# plt.savefig(f'{pw}_axis=2.png')
# plt.close()
# plt.scatter(range(180), np.mean(phi_expected, axis=(1, 2)))
# plt.savefig(f'{pw}_exp_axis=0.png')
# plt.close()
# plt.scatter(range(180), np.mean(phi_expected, axis=(0, 2)))
# plt.savefig(f'{pw}_exp_axis=1.png')
# plt.close()
# plt.scatter(range(180), np.mean(phi_expected, axis=(0, 1)))
# plt.savefig(f'{pw}_exp_axis=2.png')
# plt.close()