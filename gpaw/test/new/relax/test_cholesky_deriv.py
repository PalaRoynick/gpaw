import numpy as np
from gpaw.new.relax import chol_derivative

def numerical_chol_derivative(A, dA, eps=1e-10):
    dA = (dA + dA.T) / 2
    L = np.linalg.cholesky(A - eps * dA)
    Lp = np.linalg.cholesky(A + eps * dA)
    return (Lp - L) / (2*eps)

def test_chol_derivative():
    for i in range(10):
        # Create a random positive definite matrix
        A = np.random.rand(3,3)
        A = A @ A.T

        # Create a random perturbation
        dA = np.random.rand(3, 3)
        dA = dA - dA.T
        numdL = numerical_chol_derivative(A, dA)
        print('numdL', numdL)
        analdL = chol_derivative(A, dA)
        print('analdL', analdL)

        assert np.allclose(numdL, analdL, atol=1e-4)



