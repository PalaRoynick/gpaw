from gpaw.new.etdm.objfunc_etdm import ObjectiveFunctionETDM
from gpaw.new.etdm.tools import get_ndim
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.lcao.sic import PZSICLCAO
# from gpaw.new.etdm.lcao import get_functional
import numpy as np


class LCAOObjectiveFunctionSIC(ObjectiveFunctionETDM):
    def __init__(self, ibzwfs: IBZWaveFunctions, representation: str, matrix_exp: str,sic_calc):

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
        self.sic_calc = sic_calc
        self.counter = 0  # number of calls of this class

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

        self.e_sic=0
        for wfs in ibzwfs:
            # print('iteration')

            h_nn = self.matrix_calculator.calculate_matrix(wfs)

            h_nn.tril2full()

            hc_mn, h_ij, h_ia = self.get_ham_in_mol_orb_repr(
                h_nn.data, wfs, self._representation, self.use_egdecomp
            )

            # print(self._representation) # full
            # print(self.use_egdecomp) # False

            
            l_odd, b_mn,sic_energy_by_orb = self.sic_calc.get_sic_gradient(wfs,ibzwfs, h_ij,hc_mn)
            hc_mn+=b_mn
            h_ij+=l_odd
            self.error += self.get_residual_error(wfs, hc_mn, h_ij)
            # print(sic_energy_by_orb.sum())
            self.e_sic +=sic_energy_by_orb.sum()
            
            # energy += sic_energy.sum()

            # f_n = wfs.myocc_n

            k = 2 / wfs.nspins
            f_n = k * wfs.myocc_n

            n_occ = 0
            nbands = ibzwfs.nbands
            while n_occ < nbands and f_n[n_occ] > 1e-10:
                n_occ += 1
            c_nm=wfs.C_nM.data
            nbs = c_nm.shape[0]

            ############## From old, I dont get it  ##############
            # print(self.counter)
            
            
            ############## From old, I dont get it  ##############

            # placeholder
            
            # if occupied_only:
            #     nbs = n_occ
            # else:
            # nbs = c_nm.shape[0]
            # placeholder

            
            
            if self.counter <2: # alpha and beta channel 
                h_new = 0.5 * (h_ij + h_ij.T.conj())
                wfs.myeig_n[:nbs] = np.linalg.eigh(h_new)[0]


            if self.use_egdecomp:
                h_ij = f_n *h_ij #- f_n[:, np.newaxis] * (h_ij -l_odd + l_odd.T.conj())
                h_unn.append(h_ij)


            # # GRADIENT RIGHT PRECONDITIONIN WRONG
            # if self.counter <2: # alpha and beta channel 
            #     h_ij = 0.5 * (h_ij + h_ij.T.conj())
            #     wfs.myeig_n[:nbs] = np.linalg.eigh(h_ij)[0]
            

            # else:
            #     print(nbs,'nbs')
            #     ind_up = self._a_vec_u[0].ind_up
            #     print(h_ij.shape)
            #     print(h_ij[:nbs, :nbs].shape)
            #     print(l_odd.shape)
            #     h_ij = f_n * (h_ij[:nbs, :nbs] + l_odd) - f_n[:, np.newaxis] * (h_ij[:nbs, :nbs] + l_odd.T.conj())
            #     h_ij = np.ascontiguousarray(h_ij)
            #     if self._representation in ['sparse', 'u-invar']:
            #         h_ij = h_ij[ind_up]
            #     print(h_ij.shape)
            #     h_unn.append(h_ij)

        
        self.tot_energy = energy
        # ?????????????? OR UPDATE DIFFERENT OVJECT HERE??? COMPARE OLD TO NEW!!!
        self.counter += 1
        # energy._total_free += sic_energy

        # copy_energy_sic = energy
        # if self.objfunc.e_sic is not None:
            # print('energie added')
        self.tot_energy._energies['xc'] += self.e_sic
            # tot_free += self.objfunc.e_sic
            # self.objfunc.tot_energy._total_free += self.objfunc.e_sic
        


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
