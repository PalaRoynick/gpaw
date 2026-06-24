import numpy as np


def cholesky(a):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.cholesky(a._data))


def inv(a):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.inv(a._data))


def eigh(a, UPLO='L'):
    from gpaw.gpu import cupy as cp
    eigvals, eigvecs = np.linalg.eigh(a._data, UPLO)
    return cp.ndarray(eigvals), cp.ndarray(eigvecs.T.copy().T)


def solve(a, b):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.solve(a._data, b._data))


def matrix_rank(a, tol=None, hermitian=False, *, rtol=None):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(
        np.linalg.matrix_rank(a._data, tol, hermitian, rtol=rtol))


def eigvalsh(a, UPLO='L'):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.eigvalsh(a._data, UPLO))


def norm(x, ord=None, axis=None, keepdims=False):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.norm(x._data, ord, axis, keepdims))


def svd(a, full_matrices=True, compute_uv=True):
    from gpaw.gpu import cupy as cp
    s, v, d = np.linalg.svd(a._data, full_matrices, compute_uv)
    return cp.ndarray(s), cp.ndarray(v), cp.ndarray(d)
