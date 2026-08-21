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

        self.e_sic = 0
        for wfs in ibzwfs:

            h_nn = self.matrix_calculator.calculate_matrix(wfs)

            h_nn.tril2full()

            hc_mn, h_ij, h_ia = self.get_ham_in_mol_orb_repr(
                h_nn.data, wfs, self._representation, self.use_egdecomp
            )

            l_odd, b_mn, sic_energy_by_orb = self.sic_calc.get_sic_gradient(
                wfs, ibzwfs)
            hc_mn += b_mn

            self.e_sic += sic_energy_by_orb.sum()
            k = 2 / wfs.nspins
            f_n = k * wfs.myocc_n

            # Add if statement for occupied_only
            c_nm = wfs.C_nM.data
            nbs = c_nm.shape[0]

            if self.use_egdecomp:
                h_ij += l_odd
                h_unn.append(f_n * h_ij)
            else:
                h_nn = f_n * (h_ij[:nbs, :nbs] + l_odd) - \
                    f_n[:, np.newaxis] * (h_ij[:nbs, :nbs] + l_odd.T.conj())
                ind_up = self._a_vec_u[0].ind_up
                h_unn.append(h_nn[ind_up])
                h_ij += l_odd

            self.error += self.get_residual_error(wfs, hc_mn, h_ij)

        self.tot_energy = energy
        self.tot_energy._energies['xc'] += self.e_sic
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
