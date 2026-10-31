"""FD force regression through real CUDA or explicitly emulated device arrays."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('family', ['bulk', 'slab', 'finite'])
def test_fd_force_and_density_parity(tmp_path, family, monkeypatch):
    from gpaw.gpu import cupy_is_fake
    if cupy_is_fake:
        import gpaw.new.builder
        monkeypatch.setattr(gpaw.new.builder, 'GPAW_CPUPY', True)
    spec = importlib.util.spec_from_file_location(
        'fd_checker', Path(__file__).resolve().parents[1] / 'check_gpaw_cuda.py')
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    cpu = check.calculate(tmp_path, False, 'fd', family)
    device = check.calculate(tmp_path, True, 'fd', family)
    np.testing.assert_allclose(device[0], cpu[0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(device[1], cpu[1], rtol=0, atol=2e-5)
    relative = np.abs(device[2] - cpu[2]).sum() / np.abs(cpu[2]).sum()
    assert relative < 1e-6
