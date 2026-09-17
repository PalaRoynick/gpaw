import numpy as np
import pytest

import gpaw.cgpaw as cgpaw
from gpaw.core import UGDesc
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.new.xc import create_functional
from gpaw.xc.libxc import LibXC


device_libxc = (
    hasattr(cgpaw, 'lxcXCFunctionalDeviceAvailable') and
    cgpaw.lxcXCFunctionalDeviceAvailable())

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.libxc,
    pytest.mark.skipif(cupy_is_fake, reason='No CuPy'),
    pytest.mark.skipif(not device_libxc, reason='No device Libxc'),
]


@pytest.mark.parametrize('nspins', [1, 2])
@pytest.mark.parametrize(
    'name',
    ['LDA_X+LDA_C_PW',
     'GGA_X_PBE+GGA_C_PBE',
     'MGGA_X_SCAN+MGGA_C_SCAN'])
def test_libxc_device_matches_host(name: str, nspins: int) -> None:
    """Device Libxc preserves GPAW's energy and derivative conventions."""
    rng = np.random.default_rng(81 + nspins)
    shape = (13, 17)
    density_sg = rng.uniform(1.0e-5, 2.0, (nspins,) + shape)
    potential_sg = rng.normal(size=density_sg.shape)
    energy_g = np.empty(shape)
    functional = LibXC(name)

    gpu_density_sg = cp.asarray(density_sg)
    gpu_potential_sg = cp.asarray(potential_sg)
    gpu_energy_g = cp.empty(shape)

    if functional.type == 'LDA':
        functional.calculate(energy_g, density_sg, potential_sg)
        functional.calculate_gpu(
            gpu_energy_g, gpu_density_sg, gpu_potential_sg)
        pairs = [(energy_g, gpu_energy_g),
                 (potential_sg, gpu_potential_sg)]
    else:
        nsigma = 2 * nspins - 1
        sigma_xg = rng.uniform(0.0, 0.4, (nsigma,) + shape)
        sigma_derivative_xg = np.empty_like(sigma_xg)
        gpu_sigma_xg = cp.asarray(sigma_xg)
        gpu_sigma_derivative_xg = cp.empty_like(gpu_sigma_xg)
        arguments = [energy_g, density_sg, potential_sg,
                     sigma_xg, sigma_derivative_xg]
        gpu_arguments = [gpu_energy_g, gpu_density_sg, gpu_potential_sg,
                         gpu_sigma_xg, gpu_sigma_derivative_xg]
        pairs = [(energy_g, gpu_energy_g),
                 (potential_sg, gpu_potential_sg),
                 (sigma_derivative_xg, gpu_sigma_derivative_xg)]
        if functional.type == 'MGGA':
            tau_sg = rng.uniform(0.01, 1.0, density_sg.shape)
            tau_derivative_sg = np.empty_like(tau_sg)
            gpu_tau_sg = cp.asarray(tau_sg)
            gpu_tau_derivative_sg = cp.empty_like(gpu_tau_sg)
            arguments += [tau_sg, tau_derivative_sg]
            gpu_arguments += [gpu_tau_sg, gpu_tau_derivative_sg]
            pairs.append((tau_derivative_sg, gpu_tau_derivative_sg))
        functional.calculate(*arguments)
        functional.calculate_gpu(*gpu_arguments)

    for expected, actual in pairs:
        assert cp.asnumpy(actual) == pytest.approx(expected, abs=3.0e-14)


@pytest.mark.parametrize(
    'name',
    ['LDA_X+LDA_C_PW',
     'GGA_X_PBE+GGA_C_PBE',
     'MGGA_X_SCAN+MGGA_C_SCAN'])
def test_new_xc_uses_device_libxc(name: str) -> None:
    """The new XC frontend keeps Libxc fields and gradients on the GPU."""
    grid = UGDesc(cell=[4.0, 4.0, 4.0], size=(8, 8, 8))
    rng = np.random.default_rng(8)
    density_sR = rng.uniform(0.01, 1.0, (2, 8, 8, 8))
    cpu_density = grid.empty(2)
    cpu_density.data[:] = density_sR
    gpu_density = grid.empty(2, xp=cp)
    gpu_density.data[:] = cp.asarray(density_sR)

    cpu = create_functional(name, grid)
    gpu = create_functional(name, grid, cp)
    cpu_tau = None
    gpu_tau = None
    if name.startswith('MGGA'):
        cpu_tau = grid.empty(2)
        cpu_tau.data[:] = 0.4
        gpu_tau = grid.empty(2, xp=cp)
        gpu_tau.data[:] = 0.4

    cpu_energy, cpu_potential, cpu_tau_derivative = cpu.calculate(
        cpu_density, cpu_tau)
    gpu_energy, gpu_potential, gpu_tau_derivative = gpu.calculate(
        gpu_density, gpu_tau)

    assert gpu_energy == pytest.approx(cpu_energy, abs=3.0e-14)
    assert cp.asnumpy(gpu_potential.data) == pytest.approx(
        cpu_potential.data, abs=3.0e-14)
    if cpu_tau_derivative is not None:
        assert cp.asnumpy(gpu_tau_derivative.data) == pytest.approx(
            cpu_tau_derivative.data, abs=3.0e-14)
