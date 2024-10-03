import pytest
from gpaw.xc import XC
from gpaw.gpu import cupy as cp, cupy_is_fake



@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('nspins', [1, 2])
@pytest.mark.parametrize('xc', ['PBE', 'PBEH'])
def test_gpu_pbe(nspins, xc):
    from gpaw.cgpaw import evaluate_pbe_gpu, evaluate_pbeh_gpu
    evaluate_gga = {'PBE': evaluate_pbe_gpu, 'PBEH': evaluate_pbeh_gpu}[xc]
    ng = 10000
    n_sg = cp.exp(cp.log(10) * 5 * (cp.random.rand(nspins, ng) - 0.5))
    import numpy as np
    np.set_printoptions(precision=3)

    sigma_xg = cp.exp(cp.log(10) * 2.5 * (
                      cp.random.rand(2 * nspins - 1, ng) - 0.5))
    #sigma_xg[:] = 10.0   
    #n_sg = cp.ones((nspins, ng))
    #n_sg[0,:] /= 2
    #sigma_xg = cp.zeros((2 * nspins - 1, ng))
    #for i in range(2 * nspins -1):
    #    sigma_xg[i,:] = cp.linspace(0, 10, ng)

    dedsigma_xg = cp.zeros_like(sigma_xg)
    cp.cuda.runtime.deviceSynchronize()
    v_sg = cp.zeros_like(n_sg)
    e_g = cp.zeros((ng,))

    cpun_sg = cp.asnumpy(n_sg)
    cpuv_sg = cp.asnumpy(v_sg)
    cpue_g = cp.asnumpy(e_g)
    cpusigma_xg = cp.asnumpy(sigma_xg)
    cpudedsigma_xg = cp.asnumpy(dedsigma_xg)
    cpuxc = XC({'PBE': 'GGA_X_PBE+GGA_C_PBE', 'PBEH':'HYB_GGA_XC_PBEH'}[xc])
    import time
    start = time.time()
    cpuxc.kernel.calculate(cpue_g, cpun_sg, cpuv_sg, cpusigma_xg, cpudedsigma_xg)
    stop = time.time()
    cpu = stop - start
    print('CPU took', stop - start)

    cp.cuda.runtime.deviceSynchronize()
    start = time.time()
    evaluate_gga(n_sg, v_sg, e_g, sigma_xg, dedsigma_xg)
    cp.cuda.runtime.deviceSynchronize()
    stop = time.time()
    print('GPU took', stop - start)
    gpu = stop - start
    print('speedup', cpu / gpu)
    if len(n_sg) == 2:
        for args in zip(n_sg[0], n_sg[1], sigma_xg[0], sigma_xg[1], sigma_xg[2], e_g, cpue_g, v_sg[0], cpuv_sg[0], v_sg[1], cpuv_sg[1]):
            if cp.max(args[5]-args[6]) > 0.01:
                print(*args)
    print('max error v_g', cp.max(cp.abs(v_sg - cp.asarray(cpuv_sg)).ravel()))
    # LIBXC uses different PBE parametrization (unrounded) as GPAW
    # Therefore, loosen up the tolerance when comparing to PBEH
    if xc == 'PBEH':
        atol, rtol = 2e-5, 1e-5
    else:
        atol, rtol = 1e-7, 1e-13
    assert cp.allclose(v_sg, cpuv_sg, atol=atol, rtol=rtol)
    print('max error dedsigmax_g',
          cp.max(cp.abs(dedsigma_xg - cp.asarray(cpudedsigma_xg)).ravel()))
    assert cp.allclose(dedsigma_xg, cpudedsigma_xg, atol=atol, rtol=rtol)
    print('max error e_g', cp.max(cp.abs(e_g - cp.asarray(cpue_g)).ravel()))
    assert cp.allclose(e_g, cpue_g, atol=atol, rtol=rtol)



@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('nspins', [1, 2])
def test_gpu_lda(nspins):
    from gpaw.cgpaw import evaluate_lda_gpu
    ng = 10000
    n_sg = cp.exp(cp.log(10) * 5 * (cp.random.rand(nspins, ng) - 0.5))
    cp.cuda.runtime.deviceSynchronize()
    v_sg = cp.zeros_like(n_sg)
    e_g = cp.zeros((ng,))

    cpun_sg = cp.asnumpy(n_sg)
    cpuv_sg = cp.asnumpy(v_sg)
    cpue_g = cp.asnumpy(e_g)
    xc = XC('LDA')
    import time
    start = time.time()
    xc.calculate_impl(None, cpun_sg, cpuv_sg, cpue_g)
    stop = time.time()
    cpu = stop - start
    print('CPU took', stop - start)

    cp.cuda.runtime.deviceSynchronize()
    start = time.time()
    evaluate_lda_gpu(n_sg, v_sg, e_g)
    cp.cuda.runtime.deviceSynchronize()
    stop = time.time()
    print('GPU took', stop - start)
    gpu = stop - start
    print('speedup', cpu / gpu)

    assert cp.allclose(v_sg, cpuv_sg, atol=1e-13, rtol=1e-13)
    assert cp.allclose(e_g, cpue_g, atol=1e-14, rtol=1e-14)
