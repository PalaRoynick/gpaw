import numpy as np
from gpaw.directmin.tools import (d_matrix, expm_ed, expm_ed_unit_inv)
from gpaw.new.etdm.tools import vec2skewmat


class SkewHermitian:
    """
    Class for working with skew-Hermitian matrices A (i.e., A^\\dagger = -A).

    Only the independent upper-triangular elements are stored
    in a 1D vector (`self.data`).

    Attributes
    ----------
    ndim : int
        Dimension N of the skew-Hermitian matrix A (total number of orbitals).
    nocc: int
        Number M of occupied orbitals.
    dtype : type
        Either float or complex.
        - float → real skew-symmetric matrices
        - complex → skew-Hermitian matrices
    data : ndarray
        1D array of independent parameters of the skew-Hermitian matrix, as
        selected by ``representation``.
    representation : str
        Can be either "full" (default), "u-invar", or "sparse" for storing
        the independent parameters (upper-triangular) of the full matrix A,
        occupied-unoccupied block only, or occupied-occupied plus
        occupied-unoccupied blocks, respectively.
    """

    def __init__(self,
                 ndim: int,
                 nocc: int,
                 dtype: type,
                 representation='full',
                 data: np.ndarray = None):
        """
        Initialize a SkewHermitian object.

        Parameters
        ----------.
        ndim: int
            ndim = ibzwfs.nbands
        nocc : int
            Number M of occupied orbitals.
        dtype : type
            Either float or complex.
        data : np.ndarray
            1D vector of independent parameters (upper-triangular entries).
        representation : str
            can be either "full", "u-invar", "sparse"
        """

        self._dtype = dtype

        # Cached values: recomputed only when data changes
        self._data = None
        self._evecs = None      # eigenvectors of i*A
        self._evals = None      # eigenvalues of i*A
        self._rotation_mat = None  # U = exp(A)
        self._representation = representation

        self._ndim = ndim
        self._nocc = nocc

        # if all the orbitals are occupied
        # the only possible representation is full
        if self._ndim == self._nocc:
            self._representation = 'full'

        self.ind_up = self._make_ind_up()

        # Number of independent parameters
        self._len = len(self.ind_up[0])

        # Assign initial data (if provided)
        if data is not None:
            self.data = data
        else:
            self.data = np.zeros(shape=self._len, dtype=self._dtype)

    def _make_ind_up(self):
        # Choose which independent elements of the skew-Hermitian A are
        # stored in self.data. For real dtype, A is skew-symmetric and only
        # the strictly upper triangle is independent. For complex dtype, A
        # is skew-Hermitian and the diagonal is also independent, so the
        # upper triangle including the diagonal is stored.
        if self._representation == 'full':
            # Independent elements of the full N x N matrix
            # - if real → strictly upper triangle
            # - if complex → include diagonal
            if self._dtype == float:
                return np.triu_indices(self._ndim, 1)
            elif self._dtype == complex:
                return np.triu_indices(self._ndim)
        elif self._representation == 'u-invar':
            # Independent elements of the N * (M - N)
            # occupied-unoccupied block
            ind_up_uinv1, ind_up_uinv2 = \
                np.indices((self._nocc, (self._ndim - self._nocc)))
            return ((np.concatenate(ind_up_uinv1)).tolist(),
                    (np.concatenate(ind_up_uinv2 + self._nocc)).tolist())
        elif self._representation == 'sparse':
            # Independent elements of the N * M matrix made
            # of the occupied-occupied and occupied-unoccupied blocks
            return np.triu_indices(self._nocc, 1, self._ndim)
        else:
            raise ValueError(f'Unknown representation: '
                             f'{self._representation}')

    # ------------------------
    # Properties
    # ------------------------
    @property
    def data(self):
        """Return the parameter vector (upper-triangular entries)."""
        return self._data

    @data.setter
    def data(self, array):
        """
        Set the parameter vector (upper-triangular entries).
        Resets cached eigenvectors, eigenvalues, and rotation matrix.
        """
        if isinstance(array, np.ndarray):
            # Must be 1D vector of correct length
            assert len(array.shape) == 1
            assert len(array) == self._len

        if array is not None:
            assert self.dtype == array.dtype

        self._data = array

        # Invalidate cached values
        self._evecs = None
        self._evals = None
        self._rotation_mat = None

    @property
    def dtype(self):
        """Return matrix data type (float or complex)."""
        return self._dtype

    @property
    def representation(self):
        """Return storage representation ('full', 'u-invar', 'sparse')."""
        return self._representation

    # ------------------------
    # Eigen-decomposition and rotation
    # ------------------------
    @property
    def evecs(self):
        """
        Eigenvectors of i*A.

        Returns
        -------
        evecs : ndarray
            Eigenvectors of i*A
        """
        if self._evecs is None:
            # Convert vector to full skew-Hermitian matrix
            a_mat = vec2skewmat(self.data, self.ndim, self.ind_up, self.dtype)
            # Compute matrix exponential, eigenvectors, and eigenvalues
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True)
        return self._evecs

    @property
    def evals(self):
        """
        Eigenvalues of i*A.

        Returns
        -------
        evals : ndarray
            Eigenvalues of i*A
        """
        if self._evecs is None:  # triggers only if not already computed
            a_mat = vec2skewmat(self.data, self._ndim, self.ind_up, self.dtype)
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True)
        return self._evals

    @property
    def rotation_mat(self):
        """
        The unitary rotation matrix U = exp(A).

        Returns
        -------
        u_nn : ndarray
            Full N×N unitary matrix.
        """
        if self._data is None:
            return None
        elif self._rotation_mat is None:  # compute only once
            a_mat = vec2skewmat(self.data, self._ndim, self.ind_up, self.dtype)

            if self._representation == 'u-invar':
                # it only needs the upper right block of the A matrix
                a_upp_r = a_mat[:self._nocc, -(self._ndim - self._nocc):]
                self._rotation_mat = expm_ed_unit_inv(a_upp_r)
                # they do not get calculated by expm_ed_unit_inv
                self._evals, self._evecs = np.linalg.eigh(1.0j * a_mat)

            else:
                self._rotation_mat, self._evecs, self._evals = expm_ed(
                    a_mat, evalevec=True)

        return self._rotation_mat

    # ------------------------
    # Operator overloads
    # ------------------------
    def __add__(self, other):
        """
        Allow A + B for SkewHermitian objects or numpy arrays.

        Returns
        -------
        new : SkewHermitian
            New SkewHermitian object with data = self.data + other.data
        """
        new = SkewHermitian(
            self.ibzwfs, self.dtype, representation=self._representation
        )

        if isinstance(other, np.ndarray):
            new.data = self.data + other
            return new

        # Otherwise, other must be another SkewHermitian
        assert self._ndim == other._ndim
        assert self._representation == other.representation
        assert self.dtype == other.dtype

        new.data = self.data + other.data
        return new

    def __sub__(self, other):
        """
        Allow A - B for SkewHermitian objects or numpy arrays.

        Returns
        -------
        new : SkewHermitian
            New SkewHermitian object with data = self.data - other.data
        """
        new = SkewHermitian(
            self.ibzwfs, self.dtype, representation=self._representation
        )

        if isinstance(other, np.ndarray):
            new.data = self.data - other
            return new

        # Otherwise, other must be another SkewHermitian
        assert self._ndim == other._ndim
        assert self._representation == other.representation
        assert self.dtype == other.dtype

        new.data = self.data - other.data
        return new

    # ------------------------
    # Gradient calculation
    # ------------------------
    def calc_gradient(self, h_nn):
        """
        Compute gradient with respect to the skew-Hermitian
        parameters given a matrix h_nn.

        Parameters
        ----------
        h_nn : ndarray
            Hamiltonian-like matrix of same dimension as A.

        Returns
        -------
        grad : ndarray
            1D vector (same shape as self.data) containing gradient values.
        """
        # Step 1: eigen-decomposition of i*A
        evecs = self.evecs
        evals = self.evals

        # Step 2: transform h_nn into eigenbasis of i*A
        g_mat = evecs.T.conj() @ h_nn @ evecs

        # Step 3: Weight matrix in the eigenbasis by D, which encodes
        # the derivative of the matrix exponential:
        # D_{ij} = i * (exp(-i*(omega_i - omega_j)) - 1) / (omega_i - omega_j),
        # with D_{ii} = 1
        # This corresponds to the derivative of exp(A)
        # w.r.t. the skew-Hermitian parameters
        g_mat = g_mat * d_matrix(evals)

        # Step 4: transform back to original basis
        g_mat = evecs @ g_mat @ evecs.T.conj()

        # Step 5: fix diagonal scaling (skew-Hermitian structure)
        for i in range(g_mat.shape[0]):
            g_mat[i, i] *= 0.5

        # Step 6: if real dtype, discard imaginary part
        if self.dtype == float:
            g_mat = g_mat.real

        # Step 7: return only the independent upper-triangular elements
        return 2.0 * g_mat[self.ind_up]
