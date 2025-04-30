import pytest
import cupy as cp

from gpaw.gpu import gpu_gemm

pytestmark = pytest.mark.skipif(gpu_gemm is None, reason='no GPU gemm')
def test_gemm():
    # transa, transb, a, b, out=None, alpha=1.0, beta=0.0
    A = cp.ones((10, 10))
    B = cp.ones((10, 10))
    C = cp.ones((10, 10))
    gpu_gemm('N', 'H', A, B, out=C, alpha=1.0, beta=0.0)
    assert cp.allclose(C, 10)
    C[:] = 42
    
    gpu_gemm('N', 'N', A, B[:, 2:5], out=C[:, 4:7], alpha=1.0, beta=0.0)
    assert cp.allclose(C[:, 4:7], 10)
    C[:] = 42
    
    A[:, 3:] = 42
    B[:, :2] = 42
    gpu_gemm('N', 'T', A[:, 0:3], B[:, 2:5], out=C, alpha=1.0, beta=0.0)
    assert cp.allclose(C, 3)