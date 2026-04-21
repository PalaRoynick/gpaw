import pytest
import numpy as np
from gpaw.gpu import cupy_is_fake

@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
def test_general_xdst():
    import cupy as cp
    from gpaw.poisson import xdst
    
    # Test data on GPU
    data_cpu = np.random.rand(8, 8, 8)
    data_gpu = cp.asarray(data_cpu)
    
    for dst_type in [1, 2, 3]:
        # Reference result from SciPy
        from scipy.fftpack import dst as scipydst
        expected = scipydst(data_cpu, axis=0, type=dst_type)
        
        # Result from our general wrapper
        result_gpu = xdst(data_gpu, axis=0, type=dst_type)
        result_cpu = cp.asnumpy(result_gpu)
        
        assert np.allclose(result_cpu, expected), f"DST Type {dst_type} failed!"