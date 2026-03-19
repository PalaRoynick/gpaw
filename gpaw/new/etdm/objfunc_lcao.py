from gpaw.new.etdm.objfunc_etdm import ObjectiveFunctionETDM
from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
import numpy as np
from gpaw.new.etdm.tools2 import get_ndim


class LCAOObjectiveFunction(ObjectiveFunctionETDM):

    def __init__(self, ibzwfs, representation):

        self.ibzwfs = ibzwfs
        self._representation = representation
        ndim = get_ndim(ibzwfs)
        dtype = self.ibzwfs.dtype
        f_n = ibzwfs.get_all_eigs_and_occs(broadcast=False)[1]

        super().__init__(ndim, f_n, dtype, self._representation)

        # initialize stored values
        self._energy = None
        self._gradient = None

    @property
    def kpt_comm(self):
        return self.ibzwfs.kpt_comm
    
    def _calc_obf_value_and_matrix_elements(self):
        """Calculate value of the objective function (energy)
        and hamiltonian matrix elements at self._a_vec_u
        
        Returns
        -------
        energy : float
            Value of the objective function.
        h_unn : ndarray
            Array of Hamiltonian matrix elements for each k-point or system.
            Shape = (nkps, ndim, ndim)
        """
        # should it be here or at the beginning of
        # iterate in dir_opt?
        self.rotate_wfs()

        h_unn = []

        for wfs in self.ibzwfs:
            h_nn = self.matrix_calculator.calculate_matrix(wfs)
            h_unn.append(h_nn.data)  # i need it as a list

        energy = self.update_density_and_potential()
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

    def update_objfunc(self, density, potential,
                       pot_calc, ibzwfs, hamiltonian):
        """
        To be called from inside iterate of dir_opt to update
        the quantities needed inside the etdm step
        """
        self.density = density
        self.potential = potential
        self.pot_calc = pot_calc
        self.ibzwfs = ibzwfs
        self.hamiltonian = hamiltonian
        self.matrix_calculator = \
            hamiltonian.create_hamiltonian_matrix_calculator(potential)

    def update_density_and_potential(self):
        self.density.update(self.ibzwfs, ked=self.pot_calc.xc.type == 'MGGA')
        potential, energies, _ = self.pot_calc.calculate(self.density,
                                                         self.ibzwfs,
                                                         self.potential.vHt_x)
        energies.set(kinetic=self.ibzwfs.calculate_kinetic_energy(
            self.hamiltonian,
            self.density), band=0.0)

        return energies  # dictionary

    def rotate_wfs(self):

        for a, wfs in zip(self._a_vec_u, self.ibzwfs):

            u_nn = a.rotation_mat

            self.apply_transformation(u_nn.T, wfs, wfs.C_nM.data)

    def apply_transformation(self, u_mat, wfs, c_ref):

        dimens1 = u_mat.shape[1]
        dimens2 = u_mat.shape[0]

        print(wfs.C_nM.data[0])

        wfs.C_nM.data[:dimens2] = u_mat @ c_ref[:dimens1]

        # atomic_correction.calculate_projections
        # dense or sparse atomic correction?
        wfs._P_ani = None  # is this enough to update the projections?
