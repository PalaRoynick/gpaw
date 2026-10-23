"""Tests for GPU-aware libxc wrappers. Needs libxc built with GPU support."""

import pytest
import gpaw.cgpaw as cgpaw
from gpaw.gpu import cupy_is_fake, cupy as cp
from gpaw.xc.libxc import LibXC
import numpy as np

has_gpu_libxc = False if not hasattr(cgpaw, 'libxc_has_gpu_support') \
    else cgpaw.libxc_has_gpu_support()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.libxc,
    pytest.mark.skipif(cupy_is_fake, reason='No CuPy'),
    pytest.mark.skipif(not has_gpu_libxc, reason='No GPU-aware Libxc'),
]


@pytest.mark.parametrize('nspins', [1, 2])
@pytest.mark.parametrize(
    'name',
    ['LDA_X+LDA_C_PW',
     'GGA_X_PBE+GGA_C_PBE',
     'MGGA_X_SCAN+MGGA_C_SCAN'])
def test_libxc_gpu(name: str, nspins: int) -> None:
    """Compares CPU and GPU evaluations of a libxc functional."""

    rng = np.random.default_rng(81 + nspins)
    shape = (13, 9)
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
        result_pairs = [(energy_g, gpu_energy_g),
                        (potential_sg, gpu_potential_sg)]
    else:
        # GGA/MGGA, the latter appends a few more in/out args
        nsigma = 2 * nspins - 1
        sigma_xg = rng.uniform(0.0, 0.4, (nsigma,) + shape)
        sigma_derivative_xg = np.empty_like(sigma_xg)
        gpu_sigma_xg = cp.asarray(sigma_xg)
        gpu_sigma_derivative_xg = cp.empty_like(gpu_sigma_xg)

        arguments = [energy_g, density_sg, potential_sg,
                     sigma_xg, sigma_derivative_xg]
        gpu_arguments = [gpu_energy_g, gpu_density_sg, gpu_potential_sg,
                         gpu_sigma_xg, gpu_sigma_derivative_xg]
        result_pairs = [(energy_g, gpu_energy_g),
                        (potential_sg, gpu_potential_sg),
                        (sigma_derivative_xg, gpu_sigma_derivative_xg)]

        if functional.type == 'MGGA':
            tau_sg = rng.uniform(0.01, 1.0, density_sg.shape)
            tau_derivative_sg = np.empty_like(tau_sg)
            gpu_tau_sg = cp.asarray(tau_sg)
            gpu_tau_derivative_sg = cp.empty_like(gpu_tau_sg)
            arguments += [tau_sg, tau_derivative_sg]
            gpu_arguments += [gpu_tau_sg, gpu_tau_derivative_sg]
            result_pairs.append((tau_derivative_sg, gpu_tau_derivative_sg))

        functional.calculate(*arguments)
        functional.calculate_gpu(*gpu_arguments)

    for expected, actual in result_pairs:
        assert cp.asnumpy(actual) == pytest.approx(expected, abs=3.0e-14)
