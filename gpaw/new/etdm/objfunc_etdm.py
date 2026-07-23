import inspect
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
                 representation='full', matrix_exp='egdecomp'):
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
        Current energy of the objective function

        Returns
        -------
        float
        """

        return self._energy
    
    @energy.setter
    def energy(self, shuttle):

        if self._energy is None:
            shuttle['a_vec_u'] = self.a_vec_u
            self._energy, self._gradient = \
                self._calc_energy_and_gradient(shuttle)

    @property
    def gradient(self):
        """
        Current gradient of the objective function w.r.t. `a_u`

        Returns
        -------
        ndarray
        """

        return self._gradient

    @gradient.setter
    def gradient(self, shuttle):

        if self._gradient is None:
            shuttle['a_vec_u'] = self.a_vec_u
            self._energy, self._gradient = \
                self._calc_energy_and_gradient(shuttle)

    def _calc_energy_and_gradient(self, shuttle):
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
        # if shuttle['a_vec_u'] is not None:
        #     self._a_vec_u = shuttle['a_vec_u']
        # else:
        #     shuttle['a_vec_u'] = self._a_vec_u

        # Compute energy and Hamiltonian matrix elements
        energy, h_unn = self._calc_obf_value_and_matrix_elements(shuttle)

        gradient = self.compute_gradient(h_unn, self.a_vec_u)
        
        return energy, gradient
    
    def compute_gradient(self, h_unn, a_vec_u):
        # Compute gradient for each SkewHermitian matrix
        # h_unn - h_unn.T.conj() ensures a skew-Hermitian
        # matrix for derivative
        if self.use_egdecomp:
            gradient = np.array(
                [
                    a.calc_gradient(h_nn - h_nn.T.conj())
                    for (a, h_nn) in zip(a_vec_u, h_unn)
                ]
            )
        else:
            # In this case h_nn is already modified in objfunc_lcao
            gradient = np.array(
                [h_nn for h_nn in h_unn]
            )
            # print(gradient.shape)
            # gradient = h_unn
        return gradient

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



    ###############################################################################################

    def update_density_and_potential(
        self, ibzwfs, density, pot_calc, potential, hamiltonian
    ):
        """
        Updates density and potential

        Returns:
            energies: DFTEnergy
        """

        density.update(ibzwfs, ked=pot_calc.xc.type == "MGGA")
        potential, energies, _ = pot_calc.calculate(density, ibzwfs, potential.vHt_x)
        # Recalculate kinetic energy directly
        energies.set(
            kinetic=ibzwfs.calculate_kinetic_energy(hamiltonian, density), band=0.0
        )
        self.matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(
            potential
        )

        return energies
    
    def rotate_wfs(self, a_vec_u, ibzwfs, ref_orbitals):
        """
        Apply unitary transformation U = exp(A) to
        the reference orbitals
        """
        for u, (a, wfs) in enumerate(zip(a_vec_u, ibzwfs)):

            # Make exponential matrix
            u_nn = a.rotation_mat

            self.apply_transformation(u_nn.T, wfs, ref_orbitals[u])

    def apply_transformation(self, u_mat, wfs, c_ref):

        dimens1 = u_mat.shape[1]
        dimens2 = u_mat.shape[0]

        wfs.C_nM.data[:dimens2] = u_mat @ c_ref.data[:dimens1]

        # update projections
        wfs._P_ani = None

    def get_ham_in_mol_orb_repr(self, h_mm, wfs, representation, full_ham, occupied_only=False):
        """
        H = (C_nM @ H_MM @ C_nM.T.conj()).conj()
        for sparse and u_inv representation we calculate
        H_ij and H_ia, where i,j -- occupied and a - virtual
        H_ij is really needed only to calculate the residual later
        but it is not needed for u-invar representation.
        When matrix exp calculated using egdecomp method
        we need the whole matrix H though

        :return: H@C_nM[:occ].T, H_ij, H_ia or
                 H@C_nM.T, H, H_ia
        """

        c_nm = wfs.C_nM.data
        f_n = wfs.myocc_n > 1.0e-10
        
        caller_frame = inspect.stack()[1]
        caller_file_path = caller_frame.filename

        if caller_file_path.__contains__('objfunc_lcao_sic.py'):
            if occupied_only:
                occ = sum(f_n)
            else:
                occ = c_nm.shape[0]
        else:
            occ = sum(f_n)
            
        if representation in ["sparse", "u-invar"] and not full_ham:
            hc_mn = h_mm.conj() @ c_nm[:occ].T
            h_ij = hc_mn.T.conj() @ c_nm[:occ].T
            h_ia = hc_mn.T.conj() @ c_nm[occ:].T
        else:
            if caller_file_path.__contains__('objfunc_lcao_sic.py'):
                hc_mn = np.dot(h_mm.conj(), c_nm[:occ].T)
                h_ij = np.dot(c_nm[:occ].conj(), hc_mn)
            else:
                hc_mn = h_mm.conj() @ c_nm.T
                h_ij = c_nm.conj() @ hc_mn
            h_ia = h_ij[:occ][:, occ:]

        return hc_mn, h_ij, h_ia

    def get_residual_error(self, wfs, hc_mn, h_ij):
        """
        Calculate residual error of KS equations
        """

        c_nm = wfs.C_nM.data
        k = 2 / wfs.nspins
        f_n = k * wfs.myocc_n
        # f_n =  wfs.myocc_n
        occ = sum(f_n > 1.0e-10)

        # calculate residual
        S_MM = wfs.S_MM.data
        hc_mn = hc_mn[:, :occ] - S_MM.conj() @ c_nm[:occ].T @ h_ij[:occ, :occ]
        norm = sum(hc_mn.conj() * hc_mn * f_n[:occ])
        error = sum(norm.real)
        return error

