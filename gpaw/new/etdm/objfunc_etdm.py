import numpy as np
from .skewherm_matrix import SkewHermitian
from abc import ABC, abstractmethod
from gpaw.new.etdm.tools import get_n_occ


class ObjectiveFunctionETDM(ABC):
    """
    Abstract base class for the ETDM objective function.

    This class defines the structure for any objective function
    that depends on a set of skew-Hermitian matrices `a_vec_u`.
    These matrices parameterize unitary transformations.

    Attributes
    ----------
    _a_vec_u : list of SkewHermitian
        List of skew-Hermitian matrices representing the parameters
        for each k-point or subsystem.
    _energy : float or None
        Cached value of the objective function (energy).
    _gradient : ndarray or None
        Cached gradient of the objective function with respect to
        the independent parameters.
    _ndim : int
        Dimension of each skew-Hermitian matrix.
    _dtype : type
        Data type (float or complex) of the matrices.
    _nkps : int
        Number of k-points or independent systems.
    """

    def __init__(self, ndim: int, f_n: np.ndarray, dtype: type,
                 representation='full'):
        """
        Initialize the objective function with default skew-Hermitian matrices.

        Parameters
        ----------
        ndim : int
            Dimension of each skew-Hermitian matrix.
        f_n: np.ndarray
            array containing the occupation numbers for each spin and kpt,
                            f_n = [spin, kpt, occ]
            From: ibzwfs.get_all_eigs_and_occs
        dtype : type
            Data type, either float or complex
        nkps : int
            Number of k-points
        representation : str
            Skew-Hermitian parameterization. Can be either "full" (default),
            "u-invar", or "sparse".
        """

        # Create a list of `nkps * nspins` SkewHermitian objects,
        # each initialized with `ndim`, `dtype`, `f_n`, `representation`.
        # These hold the independent parameters that will be optimized.
        self._a_vec_u = self._create_skew_hermitian_objects(
            ndim, f_n, dtype, representation
        )

        # Initialize cached values for energy and gradient as None.
        # These will be computed lazily when requested.
        self._energy = None
        self._gradient = None

        # Store dimensions and metadata
        self._ndim = ndim
        self._dtype = dtype

    def _create_skew_hermitian_objects(self, ndim: int, f_n: np.ndarray,
                                       dtype: type, representation: str):
        nspins, nkpts = f_n.shape[:2]

        return [
            SkewHermitian(
                ndim,
                get_n_occ(f_n[spin, kpt, :]),
                dtype,
                representation,
            )
            for kpt in range(nkpts)
            for spin in range(nspins)
        ]

    @property
    def a_vec_u(self):
        """
        Accessor for the list of SkewHermitian matrices.

        Returns
        -------
        list of SkewHermitian
        """
        return self._a_vec_u

    @a_vec_u.setter
    def a_vec_u(self, a_u):
        """
        Setter for the SkewHermitian matrices.

        Updates the `data` of each SkewHermitian object in `_a_vec_u`
        with the corresponding vector from `a_u`.

        Parameters
        ----------
        a_u : list or array of 1D vectors
            Each element is the vector of independent parameters for a
            SkewHermitian matrix.
        """
        # Update the internal SkewHermitian objects
        for u, a in enumerate(self._a_vec_u):
            a.data = a_u[u]

        # Invalidate cached energy and gradient since parameters changed
        self._energy = None
        self._gradient = None

    @property
    def energy(self):
        """
        Compute or return cached value of the objective function (energy).
        Computes energy only if it hasn't been computed
        or if parameters were updated.

        Returns
        -------
        float
        """
        if self._energy is None:
            self._energy, self._gradient = self._calc_energy_and_gradient()
        return self._energy

    @property
    def gradient(self):
        """
        Compute or return cached gradient of the objective function
        with respect to the independent parameters in `a_vec_u`.
        Computes gradient only if it hasn't been computed
        or if parameters were updated.

        Returns
        -------
        ndarray
        """
        if self._gradient is None:
            self._energy, self._gradient = self._calc_energy_and_gradient()
        return self._gradient

    def _calc_energy_and_gradient(self, a_u: np.ndarray = None):
        """
        Calculate the objective function value (energy) and its gradient
        with respect to the skew-Hermitian parameters.

        Parameters
        ----------
        a_u : ndarray or list, optional
            If provided, updates `_a_vec_u` with this new set of parameters
            before calculating energy and gradient.

        Returns
        -------
        energy : float
            Value of the objective function.
        gradient : ndarray
            Gradient with respect to each independent parameter in `_a_vec_u`.
        """
        # If a new parameter vector is given,
        # update the internal SkewHermitian objects
        if a_u is not None:
            self.a_vec_u = a_u

        # Compute energy and Hamiltonian matrix elements
        energy, h_unn = self._calc_obf_value_and_matrix_elements()

        # Compute gradient for each SkewHermitian matrix
        # h_unn - h_unn.T.conj() ensures a skew-Hermitian
        # matrix for derivative
        gradient = np.array(
            [
                a.calc_gradient(h_nn - h_nn.T.conj())
                for (a, h_nn) in zip(self.a_vec_u, h_unn)
            ]
        )

        return energy, gradient

    @abstractmethod
    def _calc_obf_value_and_matrix_elements(self):
        """
        Abstract method to compute the value of the objective function (energy)
        and the Hamiltonian matrix elements at the current `_a_vec_u`.

        Subclasses must implement this method.

        Returns
        -------
        energy : float
            Value of the objective function.
        h_unn : ndarray
            Array of Hamiltonian matrix elements for each k-point or system.
            Shape = (nkps, ndim, ndim)
        """
        pass
