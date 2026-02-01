import numpy as np
from gpaw.new.relax import chol_derivative

def numerical_chol_derivative(A, dA, eps=1e-8):
    L = np.linalg.cholesky(A)
    Lp = np.linalg.cholesky(A + eps * dA)
    return (Lp - L) / eps

def test_chol_derivative():
    for i in range(10):
        # Create a random positive definite matrix
        A = np.random.rand(4,4)
        A = A @ A.T

        # Create a random perturbation
        dA = np.random.rand(4,4)

        numdL = numerical_chol_derivative(A, dA)

        analdL = chol_derivative(A, dA)

        assert np.allclose(numdL, analdL)



