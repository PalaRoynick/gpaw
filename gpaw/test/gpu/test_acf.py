import numpy as np
import pytest

from gpaw import GPAW_NO_C_EXTENSION, cgpaw
from gpaw.core import UGDesc
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.mpi import world
from gpaw.new.c import GPU_AWARE_MPI, GPU_ENABLED
from gpaw.spline import Spline


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [float, complex])
def test_acf(dtype):
    if GPAW_NO_C_EXTENSION:
        pytest.skip('GPAW_NO_C_EXTENSION')

    if world.size > 1 and not GPU_AWARE_MPI:
        pytest.skip('MPI not GPU-aware')
    s = Spline.from_data(0, 1.0, [1.0, 0.5, 0.0])
    n = 40
    a = 8.0
    relpos_ac = [(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]]

    grid = UGDesc(cell=[a, a, a], size=(n, n, n), comm=world, dtype=dtype)
    basis_cpu = grid.atom_centered_functions([[s], [s], [s]],
                                             positions=relpos_ac, xp=np)
    basis_gpu = grid.atom_centered_functions([[s], [s], [s]],
                                             positions=relpos_ac, xp=cp)

    P_cpu_ani = basis_cpu.layout.empty()
    P_gpu_ani = basis_gpu.layout.empty()
    P_cpu_ani.data[:] = 1.0
    P_gpu_ani.data[:] = 1.0

    b_cpu = grid.zeros(xp=np)
    b_gpu = grid.zeros(xp=cp)
    basis_cpu.add_to(b_cpu, P_cpu_ani)
    basis_gpu.add_to(b_gpu, P_gpu_ani)

    assert b_cpu.data == pytest.approx(b_gpu.data.get(), abs=1e-12)

    out_cpu_ani = basis_cpu.integrate(b_cpu)
    out_gpu_ani = basis_gpu.integrate(b_gpu)

    for a, out_cpu_ni in out_cpu_ani.items():
        assert out_cpu_ni == pytest.approx(out_gpu_ani[a].get(), abs=1e-12)


@pytest.mark.gpu
def test_native_lfc_bloch_buffers_are_released():
    """Rebuilding periodic LFC objects must not retain their CUDA phase tables."""
    if GPAW_NO_C_EXTENSION or cupy_is_fake or not GPU_ENABLED:
        pytest.skip('Requires the native extension and a real GPU')
    if world.size != 1:
        pytest.skip('Device memory accounting requires one MPI rank per GPU')

    # One localized function on 4096 disjoint grid intervals. Its 128 Bloch
    # phases occupy 8 MiB on the device, outside the CuPy memory pool.
    intervals = 4096
    phases = np.exp(2j * np.pi * np.arange(128) / 128)[:, None]
    args = ([np.ones((intervals, 1))], np.zeros(1, np.intc),
            np.arange(2 * intervals, dtype=np.intc),
            np.tile(np.array([0, -1], np.intc), intervals), 1.0, phases)
    cpu = cgpaw.LFC(*args)
    coefficients = np.ones(1, complex)
    expected = np.zeros((1, 1, 2 * intervals), complex)
    q = len(phases) - 1
    cpu.add(coefficients, expected, q)
    expected_integral = np.zeros(1, complex)
    cpu.integrate(expected, expected_integral, q)

    coefficients_gpu = cp.asarray(coefficients)
    values_gpu = cp.zeros(expected.shape, complex)
    integral_gpu = cp.zeros(1, complex)

    def rebuild():
        native = cgpaw.LFC(*args, 1)
        values_gpu.fill(0)
        integral_gpu.fill(0)
        native.add_gpu(coefficients_gpu.data.ptr, coefficients_gpu.shape,
                       values_gpu.data.ptr, values_gpu.shape, q)
        native.integrate_gpu(values_gpu.data.ptr, values_gpu.shape,
                             integral_gpu.data.ptr, integral_gpu.shape, q)
        np.testing.assert_allclose(values_gpu.get(), expected, atol=1e-12,
                                   rtol=0)
        np.testing.assert_allclose(integral_gpu.get(), expected_integral,
                                   atol=1e-12, rtol=0)
        del native
        cp.cuda.runtime.deviceSynchronize()

    rebuild()  # Warm up CUDA and CuPy before accounting for retained memory.
    free_before, _ = cp.cuda.runtime.memGetInfo()
    for _ in range(4):
        rebuild()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    # Without releasing phase_i_gpu, these rebuilds retain at least 32 MiB.
    assert free_before - free_after < 2**20, 'Native LFC device memory leaked'
