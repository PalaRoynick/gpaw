import time
import numpy as np

import pytest

import gpaw.fftw as fftw

N = 16
blocksize = 2
# n1, n2, n3 = 32, 28, 128
n1, n2, n3 = 8, 6, 12


def check(Plans, flags, dtype):
    t0 = time.time()
    tmp = Plans._overwrite_flags
    Plans._overwrite_flags = flags  # Hack to avoid the test overwrite
    plan = Plans((n1, n2, n3), dtype=dtype, blocksize=blocksize)
    Plans._overwrite_flags = tmp  # Restore the value
    t1 = time.time()
    t = 0.0
    for i in range(N):
        plan.tmp_R[:] = 1.3
        t2 = time.time()
        plan.fft()
        plan.ifft()
        t3 = time.time()
        t += t3 - t2
    return t1 - t0, t


@pytest.mark.skipif(not fftw.have_fftw(), reason='No FFTW')
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('plan_info', [(fftw.NumpyFFTPlans, 117),
                                       (fftw.FFTWPlans, fftw.ESTIMATE),
                                       (fftw.FFTWPlans, fftw.MEASURE),
                                       (fftw.FFTWPlans, fftw.PATIENT),
                                       (fftw.FFTWPlans, fftw.EXHAUSTIVE)])
def test_fft(plan_info, dtype):
    Plans = plan_info[0]
    flags = plan_info[1]
    tplan, tfft = check(Plans, flags, dtype)
    print('%-12s %3d %10.6f %10.6f' %
          (Plans.__name__, flags, tplan, tfft))
