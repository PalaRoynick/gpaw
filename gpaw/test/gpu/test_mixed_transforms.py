import pytest
from itertools import product, permutations


@pytest.mark.gpu
@pytest.mark.serial
def test_mixed_transforms():
    from gpaw.mixed_transforms import MixedTransforms
    try:
        import cupy as cp
    except ImportError:
        pytest.skip('No cupy.')
    import numpy as np
    from gpaw.gpu import cpupy as cpup

    cpt = MixedTransforms(cp)
    npt = MixedTransforms(np)
    cpupt = MixedTransforms(cpup)

    def matrices():
        for sx in [1, 5, 6, 31, 32]:
            for sy in [1, 5, 6, 31, 32]:
                for sz in [1, 5, 6, 33, 34]:
                    A = cp.random.rand(sx, sy, sz)
                    yield A, cp.asnumpy(A), cpup.ndarray(cp.asnumpy(A))

    for cpm, npm, cpupm in matrices():
        print(cpm.shape)
        # Test rfftn and irfftn
        for axes in map(list, permutations(range(3), 2)):
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
            # Test transform2
            for pbcs in product(* [[True, False]] * 2):
                kwargs = dict(axes=axes, pbc=pbcs)
                cpr = cpt.transform2(cpm, **kwargs)
                npr = npt.transform2(npm, **kwargs)
                cpupr = cpupt.transform2(cpupm, **kwargs)
                assert np.allclose(cp.asnumpy(cpr), npr, 1e-14)
                assert np.allclose(cpupr._data, npr, 1e-14)
                cpm_ = cpt.itransform2(cpr, **kwargs)
                assert cp.allclose(cpm, cpm_, 1e-14)
                npm_ = npt.itransform2(npr, **kwargs)
                assert np.allclose(npm, npm_, 1e-14)
                cpupm_ = cpupt.itransform2(cpupr, **kwargs)
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

            # Test transform
            for pbc in [True, False]:
                kwargs = dict(axis=axis, pbc=pbc)
                cpr = cpt.transform(cpm, **kwargs)
                npr = npt.transform(npm, **kwargs)
                cpupr = cpupt.transform(cpupm, **kwargs)
                assert np.allclose(cp.asnumpy(cpr), npr, 1e-14)
                assert np.allclose(cpupr._data, npr, 1e-14)
                cpm_ = cpt.itransform(cpr, **kwargs)
                assert cp.allclose(cpm, cpm_, 1e-14)
                npm_ = npt.itransform(npr, **kwargs)
                assert np.allclose(npm, npm_, 1e-14)
                cpupm_ = cpupt.itransform(cpupr, **kwargs)
                assert cpup.allclose(cpupm_, cpupm, rtol=1e-14)
