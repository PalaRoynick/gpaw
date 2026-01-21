import numpy as np
import scipy

def test_scipy():
    a = np.asanyarray([[-0.52768062+1.20370622e-35j]])
    eigvals, eigvecs = scipy.linalg.eigh(a)
