import pytest


@pytest.mark.gpu
def test_mixed_transforms():
    from gpaw.mixed_transforms import MixedTransforms
    import cupy as cp
    import numpy as np
    from gpaw.gpu import cpupy as cpup

    cpt = MixedTransforms(cp)
    npt = MixedTransforms(np)
    cpupt = MixedTransforms(cpup)

    def matrices():
        for sx in [1, 5, 6, 32, 50, 52]:
            for sy in [1, 5, 6, 32, 51, 52]:
                for sz in [5, 6, 32, 52, 53]:
                    A = cp.random.rand(sx, sy, sz)
                    yield A, cp.asnumpy(A), cpup.ndarray(cp.asnumpy(A))

    for cpm, npm, cpupm in matrices():
        # Test rfftn and irfftn
        for axes in [[0,1], [1,0], [0, 2], [1, 2], [2, 1], [2, 0]]:
            print(axes, cpm.shape, cpm.shape)
            cpr = cpt.rfst2(cpm, axes=axes)
            npr = npt.rfst2(npm, axes=axes)
            cpupr = cpupt.rfst2(cpupm, axes=axes)
            assert np.allclose(cp.asnumpy(cpr), npr, 1e-14)
            assert np.allclose(cpupr._data, npr, 1e-14)
            
            cpm_ = cpt.irfst2(cpr, axes=axes)
            assert cp.allclose(cpm, cpm_, 1e-14)

            npm_ = npt.irfst2(npr, axes=axes)
            assert np.allclose(npm, npm_, 1e-14)
            
            cpupm_ = cpupt.irfst2(cpupr, axes=axes)
            assert cpup.allclose(cpupm_, cpupm, rtol=1e-14)

        # Test fst and ifst
        for axis in [0, 1, 2]:
            cpr = cpt.fst(cpm, axis)
            npr = npt.fst(npm, axis)
            cpupr = cpupt.fst(cpupm, axis)
            assert np.allclose(cp.asnumpy(cpr), npr, 1e-14)
            assert np.allclose(cpupr._data, npr, 1e-14)
            
            cpm_ = cpt.ifst(cpr, axis)
            assert cp.allclose(cpm, cpm_, 1e-14)

            npm_ = npt.ifst(npr, axis)
            assert np.allclose(npm, npm_, 1e-14)
            
            cpupm_ = cpupt.ifst(cpupr, axis)
            assert cpup.allclose(cpupm_, cpupm, rtol=1e-14)

