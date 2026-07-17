import numpy as np


def get_ndim(ibzwfs):

    return ibzwfs.nbands


def get_n_occ(f_n):
    occupied = f_n > 1.0e-10
    n_occ = len(f_n[occupied])

    return n_occ


def vec2skewmat(a_vec, dim, ind_up, dtype):
    """
    Convert parameter vector to full skew-Hermitian matrix.

    Parameters
    ----------
    a_vec : ndarray
        1D vector of independent upper-triangular entries.
    dim : int
        Dimension of the full matrix.
    ind_up : tuple
        Indices of the upper-triangular elements.
    dtype : type
        float or complex.

    Returns
    -------
    a_mat : ndarray
        Full skew-Hermitian matrix with A^\\dagger = -A.
    """
    a_mat = np.zeros((dim, dim), dtype=dtype)
    a_mat[ind_up] = a_vec
    a_mat -= a_mat.T.conj()  # enforce skew-Hermitian property
    np.fill_diagonal(a_mat, a_mat.diagonal() * 0.5)  # fix diagonal factor
    return a_mat


def random_a(shape, dtype, seed=None):
    """
    Generate random parameter vector for skew-Hermitian matrices.

    Parameters
    ----------
    shape : tuple
        Shape of the vector.
    dtype : type
        float or complex.

    Returns
    -------
    a : ndarray
        Random vector of given dtype.
    """
    if seed:
        np.random.seed(seed)
    a = np.random.random_sample(shape)
    if dtype == complex:
        a = a.astype(complex)
        a += 1.0j * np.random.random_sample(shape)
    return a
