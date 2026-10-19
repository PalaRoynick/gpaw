import numpy as np
import pytest

import gpaw.mpi as mpi
from gpaw.response.g0w0 import EXXVXCCache


class DummyEXXVXCCalculator:
    def __init__(self):
        self.calls = 0

    def calculate(self, *, n1, n2, kpt_indices):
        self.calls += 1
        shape = (1, len(kpt_indices), n2 - n1)
        vxc_skn = np.arange(np.prod(shape), dtype=float).reshape(shape)
        exx_skn = vxc_skn + 10.0
        return vxc_skn, exx_skn


def test_exx_vxc_cache_reuses_calculated_values(tmp_path, monkeypatch):
    calculator = DummyEXXVXCCalculator()
    path = tmp_path / 'gw_exx_vxc.json'
    cache = EXXVXCCache(path, comm=mpi.SerialCommunicator())

    vxc_skn, exx_skn = cache.calculate(calculator, n1=1, n2=3,
                                       kpt_indices=[0, 2])
    assert calculator.calls == 1
    assert path.is_file()

    def raise_if_recalculated(**kwargs):
        raise AssertionError('EXX/VXC was recalculated')

    monkeypatch.setattr(calculator, 'calculate', raise_if_recalculated)

    cached_vxc_skn, cached_exx_skn = cache.calculate(
        calculator, n1=1, n2=3, kpt_indices=[0, 2])

    np.testing.assert_allclose(cached_vxc_skn, vxc_skn)
    np.testing.assert_allclose(cached_exx_skn, exx_skn)
    assert calculator.calls == 1


def test_exx_vxc_cache_rejects_mismatching_inputs(tmp_path):
    calculator = DummyEXXVXCCalculator()
    cache = EXXVXCCache(tmp_path / 'gw_exx_vxc.json',
                        comm=mpi.SerialCommunicator())

    cache.calculate(calculator, n1=1, n2=3, kpt_indices=[0, 2])

    with pytest.raises(RuntimeError, match='mismatching input parameters'):
        cache.calculate(calculator, n1=1, n2=4, kpt_indices=[0, 2])

    assert calculator.calls == 1
