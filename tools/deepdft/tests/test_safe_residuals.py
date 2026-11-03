"""The large-grid workaround preserves the residual and native small-grid path."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'run_reference_safe_residuals',
    Path(__file__).resolve().parents[1] / 'run_reference_safe_residuals.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('shape', [(2, 31), (2, 16 * 65535),
                                  (2, 16 * 65535 + 1), (2, 128, 128, 128)])
def test_safe_residuals(monkeypatch, dtype, shape):
    from gpaw.new.pwfd import eigensolver
    from gpaw.purepython import calculate_residuals_gpu

    calls = []

    def native(residual, eigenvalues, wavefunctions):
        calls.append(residual.shape)
        assert np.prod(residual.shape[1:]) <= module.MAX_NATIVE_GRID_POINTS
        calculate_residuals_gpu(residual, eigenvalues, wavefunctions)

    monkeypatch.setattr(eigensolver, 'calculate_residuals_gpu', native)
    # The monkeypatch fixture must restore the installed wrapper as well.
    original = module.install_workaround()
    try:
        residual = np.full(shape, 0.75, dtype=dtype)
        wavefunctions = np.full(shape, 0.25, dtype=dtype)
        if dtype is np.complex128:
            wavefunctions += 0.125j
        eigenvalues = np.array([0.5, -0.25])
        expected = residual - eigenvalues.reshape((-1,) + (1,) * (len(shape) - 1)) * wavefunctions
        eigensolver.calculate_residuals_gpu(residual, eigenvalues, wavefunctions)
        np.testing.assert_allclose(residual, expected, atol=1e-14, rtol=0)
        assert bool(calls) == (np.prod(shape[1:]) <= module.MAX_NATIVE_GRID_POINTS)
    finally:
        eigensolver.calculate_residuals_gpu = original
