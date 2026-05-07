from gpaw.new.etdm.objfunc_etdm import ObjectiveFunctionETDM
from gpaw.new.etdm.tools import get_ndim
import numpy as np


class LCAOObjectiveFunction(ObjectiveFunctionETDM):

    def __init__(self, ibzwfs, representation, matrix_exp):

        self.ibzwfs = ibzwfs
        self._representation = representation
        ndim = get_ndim(ibzwfs)
        dtype = self.ibzwfs.dtype
        self.f_n = ibzwfs.get_all_eigs_and_occs(broadcast=False)[1]
        self.matrix_exp = matrix_exp
        self.use_egdecomp = self.matrix_exp == 'egdecomp'

        super().__init__(ndim, self.f_n, dtype, self._representation,
                         self.matrix_exp)

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

        ibzwfs = shuttle['ibzwfs']
        density = shuttle['density']
        ref_orbitals = shuttle['ref_orbitals']
        pot_calc = shuttle['pot_calc']
        potential = shuttle['potential']
        hamiltonian = shuttle['hamiltonian']
        a_vec_u = shuttle['a_vec_u']

        self.rotate_wfs(a_vec_u, ibzwfs, ref_orbitals)

        energy = self.update_density_and_potential(ibzwfs, density, pot_calc,
                                                   potential, hamiltonian)
        
        self.error = 0.0
        h_unn = []

        for wfs in ibzwfs:
            h_nn = self.matrix_calculator.calculate_matrix(wfs)

            h_nn.tril2full()

            hc_mn, h_ij, h_ia =\
                self.get_ham_in_mol_orb_repr(h_nn.data, wfs,
                                             self._representation,
                                             self.use_egdecomp)
            
            self.error += self.get_residual_error(wfs, hc_mn, h_ij)

            k = 2 / wfs.nspins
            f_n = k * wfs.myocc_n
            if self.use_egdecomp:
                h_ij = f_n * h_ij
                h_unn.append(h_ij)

            else:
                ind_up = self._a_vec_u[0].ind_up
                if self._representation == 'u-invar':
                    h_ij = h_ia
                else:
                    if self._representation == 'full' and wfs.dtype == complex:
                        indl = np.tril_indices(h_ij.shape[0], -1)
                    else:
                        indl = np.tril_indices(h_ij.shape[0])
                    h_ij[indl] = np.inf
                    if self._representation == 'sparse':
                        h_ij = np.concatenate((h_ij, h_ia), axis=1)
                h_ij = h_ij.ravel()
                h_ij = h_ij[h_ij != np.inf]

                ones_mat = np.ones(shape=(len(f_n), len(f_n)))
                anti_occ = f_n * ones_mat - \
                    f_n[:, np.newaxis] * ones_mat
                anti_occ = anti_occ[ind_up]
                h_unn.append(anti_occ * h_ij)

        self.tot_energy = energy
        
        return energy, h_unn

    @property
    def tot_energy(self):
        """
        Current energy of the objective function

        Returns
        -------
        float
        """
        return self._tot_energy

    @tot_energy.setter
    def tot_energy(self, en):

        self._tot_energy = en

    def update_density_and_potential(self,
                                     ibzwfs,
                                     density,
                                     pot_calc,
                                     potential,
                                     hamiltonian):

        density.update(ibzwfs, ked=pot_calc.xc.type == 'MGGA')
        potential, energies, _ = pot_calc.calculate(density,
                                                    ibzwfs,
                                                    potential.vHt_x)
        # Recalculate kinetic energy directly
        energies.set(kinetic=ibzwfs.calculate_kinetic_energy(
            hamiltonian, density), band=0.0)
        self.matrix_calculator = \
            hamiltonian.create_hamiltonian_matrix_calculator(potential)

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

    def get_ham_in_mol_orb_repr(self, h_mm, wfs, representation,
                                full_ham):

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
        occ = sum(f_n)

        if representation in ['sparse', 'u-invar'] \
                and not full_ham:
            hc_mn = h_mm.conj() @ c_nm[:occ].T
            h_ij = hc_mn.T.conj() @ c_nm[:occ].T
            h_ia = hc_mn.T.conj() @ c_nm[occ:].T
        else:
            hc_mn = h_mm.conj() @ c_nm.T
            h_ij = c_nm.conj() @ hc_mn
            h_ia = h_ij[:occ][:, occ:]

        return hc_mn, h_ij, h_ia
    
    def get_residual_error(self, wfs,
                           hc_mn, h_ij):
        """
        Calculate residual error of KS equations
        """

        c_nm = wfs.C_nM.data
        f_n = wfs.myocc_n * 2
        occ = sum(f_n > 1.0e-10)

        # calculate residual
        S_MM = wfs.S_MM.data
        hc_mn = hc_mn[:, :occ] - \
            S_MM.conj() @ c_nm[:occ].T @ h_ij[:occ, :occ]
        norm = sum(hc_mn.conj() * hc_mn * f_n[:occ])
        error = sum(norm.real)
        return error
