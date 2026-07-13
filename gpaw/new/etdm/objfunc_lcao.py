from gpaw.new.etdm.objfunc_etdm import ObjectiveFunctionETDM
from gpaw.new.etdm.tools import get_ndim
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.lcao.sic import PZSICLCAO
# from gpaw.new.etdm.lcao import get_functional
import numpy as np


class LCAOObjectiveFunction(ObjectiveFunctionETDM):

    def __init__(self, ibzwfs: IBZWaveFunctions, representation: str, matrix_exp: str):

        self.ibzwfs = ibzwfs
        self._representation = representation
        ndim = get_ndim(ibzwfs)
        dtype = self.ibzwfs.dtype
        self.f_n = ibzwfs.get_all_eigs_and_occs(broadcast=False)[1]
        self.matrix_exp = matrix_exp
        self.use_egdecomp = self.matrix_exp == "egdecomp"

        super().__init__(ndim, self.f_n, dtype, self._representation, self.matrix_exp)
    
        # self.func = get_functional(functional, wfs, dens, ham)

        # initialize stored values
        self._energy = None
        self._gradient = None

    @property
    def kpt_comm(self):
        return self.ibzwfs.kpt_comm

    def _calc_obf_value_and_matrix_elements(self, shuttle):
        """
        Calculate value of the objective function (energy)
        and hamiltonian matrix elements at self._a_vec_u

        Returns
        -------
        energy : float
            Value of the objective function.
        h_unn : ndarray
            Array of Hamiltonian matrix elements for each k-point or system.
            Shape = (nkps, ndim, ndim)
        """

        # UNPACK SHUTTLE

        ibzwfs = shuttle["ibzwfs"]
        density = shuttle["density"]
        ref_orbitals = shuttle["ref_orbitals"]
        pot_calc = shuttle["pot_calc"]
        potential = shuttle["potential"]
        hamiltonian = shuttle["hamiltonian"]
        a_vec_u = shuttle["a_vec_u"]

        self.rotate_wfs(a_vec_u, ibzwfs, ref_orbitals)

        energy = self.update_density_and_potential(
            ibzwfs, density, pot_calc, potential, hamiltonian
        )

        self.error = 0.0
        h_unn = []

        for wfs in ibzwfs:
            h_nn = self.matrix_calculator.calculate_matrix(wfs)

            h_nn.tril2full()

            hc_mn, h_ij, h_ia = self.get_ham_in_mol_orb_repr(
                h_nn.data, wfs, self._representation, self.use_egdecomp
            )

            self.error += self.get_residual_error(wfs, hc_mn, h_ij)

            k = 2 / wfs.nspins
            f_n = k * wfs.myocc_n
            if self.use_egdecomp:
                h_ij = f_n * h_ij
                h_unn.append(h_ij)

            else:
                ind_up = self._a_vec_u[0].ind_up
                if self._representation == "u-invar":
                    h_ij = h_ia
                else:
                    if self._representation == "full" and wfs.dtype == complex:
                        indl = np.tril_indices(h_ij.shape[0], -1)
                    else:
                        indl = np.tril_indices(h_ij.shape[0])
                    h_ij[indl] = np.inf
                    if self._representation == "sparse":
                        h_ij = np.concatenate((h_ij, h_ia), axis=1)
                h_ij = h_ij.ravel()
                h_ij = h_ij[h_ij != np.inf]

                ones_mat = np.ones(shape=(len(f_n), len(f_n)))
                anti_occ = f_n * ones_mat - f_n[:, np.newaxis] * ones_mat
                anti_occ = anti_occ[ind_up]
                h_unn.append(anti_occ * h_ij)

        self.tot_energy = energy

        # print(h_unn)

        return energy, h_unn

    @property
    def tot_energy(self):
        """
        Current energy of the objective function

        Returns
        -------
        DFTEnergy
        """
        return self._tot_energy

    @tot_energy.setter
    def tot_energy(self, en):

        self._tot_energy = en
