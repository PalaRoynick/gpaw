import numpy as np
from gpaw.new.energies import DFTEnergies
from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.potential import Potential
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.etdm.searchdir import LBFGS
from gpaw.new.etdm.etdm import ETDM
#from gpaw.directmin.tools import sort_orbitals_according_to_occ
from gpaw.new.etdm.objfunc_lcao import LCAOObjectiveFunction


class DirOptLCAO(LCAOEigensolver):
    def __init__(self,
                 basis,
                 hamiltonian,
                 excited_socc: bool = False,
                 sctate: bool = False,
                 converge_unalapack_params=(None, 1, 1, None)):
        
        self.search_dir: LBFGS | None = None
        self.name = "etdm-lcao"
        # altro?
#        self.scalapack = scalapack_params

        # move inside initialize?
        super().__init__(basis)
#--------------------------------------------------------------------
        # this will be called inside scf.py when the mode = 'lcao'

    def initialize(self, ibzwfs, hamiltonian, density,
                   occ_calc, pot_calc,
                   potential):

        self.representation = "full"  # for now

        super().iterate(ibzwfs, density, potential, hamiltonian)

        nelectrons = density.nvalence - density.charge + pot_calc.charge
        ibzwfs.calculate_occs(occ_calc, nelectrons)
        # initial sort orbitals (MISSING: changes with mom, sic)
        self.sort_orbitals_according_to_occ(ibzwfs)  # credo

        self.objfunc = LCAOObjectiveFunction(ibzwfs, self.representation)
        # a_vec_u is a list, needs to be unpacked into an array?
        a = []
        for u in range(len(self.objfunc.a_vec_u)):
            a.append(self.objfunc.a_vec_u[u].data)  # così accedo al vettore creato sopra
        a_u_init = np.array(a)
        self.etdm = ETDM(self.objfunc, a_u_init, update_ref=True)
           
        # STEPS THAT IN OLD WERE IN ETDM_HELPER_LCAO

        # initialize_orbitals
        # MISSING: coeff read from file, need_init_orb,
        # need canon coeff?

        # MISSING: get functional, randomize orbitals
            
        # calculate occupation numbers (MISSING: if not read from file)
        
        # moved above

        # MISSING: localize for sic
        # MISSING: initialize mom

#        self.set_reference_orbitals(ibzwfs)

        # set reference orbitals, serve anche sotto
        # RIFARE, is it needed here?
    def set_reference_orbitals(self, ibzwfs):
        ndim = ibzwfs.nbands
        # these are coefficients
        for wfs in ibzwfs:   # ???????
            u = self.kpointval(kpt)
            self.reference_orbitals[u] = np.copy(wfs.C_nM[:ndim[u]])

    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian,
                pot_calc,
                energies) -> tuple[float, float, DFTEnergies]:
        """
        One iteration of direct optimization for LCAO
        """
        # MISSING: set iters for counter=20
        # MISSING SET REF ORBITALS E A VEC

        # this should update the quantities inside the objective 
        # function, to be used by etdm
        self.objfunc.update_objfunc(density, potential,
                                    pot_calc, ibzwfs, hamiltonian)
        # relative update, MISSING: counter=20 reset

        # one step of etdm
        self.etdm.one_iteration()

        self.error = 0.0
        # update wavefunctions??
        for wfs in ibzwfs:
            # wfs._P_ani = None 
            error = self.get_residual_error(wfs,
                                            potential,
                                            hamiltonian,
                                            self.representation)
            self.error += error
            # wfs.kd.comm.sum_scalar(self.error)?

        print(self.objfunc.tot_energy)
        return 0.0, self.error, self.objfunc.tot_energy


    def update_to_canonical_orbitals(self, wfs, hamiltonian, potential):
        """
        update to canonical orbitals at the end of the scf cycle
        """

        matrix_calculator = \
            hamiltonian.create_hamiltonian_matrix_calculator(potential)
        h_mm = matrix_calculator.calculate_matrix(wfs)

        wfs.C_nM, eps = self.rotate_subspace(h_mm, wfs.C_nM)
        # diagonalize entire matrix or equally occupied subspaces separately?
        # missing sic option

    def rotate_subspace(h_mm, c_nm):
        """
        choose canonical orbitals
        """
        l_nn = c_nm @ h_mm @ c_nm.conj().T
        # check if diagonal then don't rotate? it could save a bit of time
        eps, w = np.linalg.eigh(l_nn)
        return w.T.conj() @ c_nm, eps
        
    def get_residual_error(self, wfs,
                           potential,
                           hamiltonian,
                           representation):
        """
        Calculate residual error of KS equations
        """
        # for now gets recalculated, also move the calculator
        matrix_calculator = \
            hamiltonian.create_hamiltonian_matrix_calculator(potential)
        h_mm_matrix = matrix_calculator.calculate_matrix(wfs)
        h_mm = h_mm_matrix.data

        # get hamiltonian in molecular
        # orbital representation
        c_nm = wfs.C_nM.data
        f_n = wfs.myocc_n > 1.0e-10
        occ = sum(f_n)

        if representation in ['sparse', 'u-invar']:
            hc_mn = h_mm.conj() @ c_nm[:occ].T
            h_ij = hc_mn.T.conj() @ c_nm[:occ].T
            h_ia = hc_mn.T.conj() @ c_nm[occ:].T
        else:
            hc_mn = h_mm.conj() @ c_nm.T
            h_ij = c_nm.conj() @ hc_mn
            h_ia = h_ij[:occ][:, occ:]

        # calculate residual
        S_MM = wfs.S_MM.data
        hc_mn = hc_mn[:, :occ] - \
            S_MM.conj() @ c_nm[:occ].T @ h_ij[:occ, :occ]
        norm = sum(hc_mn.conj() * hc_mn * f_n[:occ])
        error = sum(norm.real)

        return error

    def update_ref_orbitals(self, ibzwfs, hamiltonian, density):

        """
        Update reference orbitals and reset search direction algorithm
        """
            # for now we don't use it, relative update?
            # MISSING: check if canonical orbitals are needed, or mom restart,
            # get_canonical_representation

        counter = self.update_ref_orb_counter

        if (self.iters % counter == 0 and self.iters > 1):
            self.set_reference_orbitals(ibzwfs)
            self.search_dir.reset()

    def set_ref_orbitals_and_a_vec(self, ibzwfs):

        self.set_reference_orbitals(ibzwfs)
        self.a_vec_u = np.zeros_like(self.a_vec_u)

    def sort_orbitals_according_to_occ(self, ibzwfs):

        """Sort orbitals according to the occupation
        numbers so that there are no holes in the
        distribution of occupation numbers
        :return:
        """

        for wfs in ibzwfs:

            occupied = wfs.myocc_n > 1.0e-10
            n_occ = sum(occupied)
            if n_occ != 0.0 and np.min(wfs.myocc_n[:n_occ]) == 0:
                ind_occ = np.argwhere(occupied)
                ind_unocc = np.argwhere(~occupied)
                ind = np.vstack((ind_occ, ind_unocc))
                ind = np.squeeze(ind)

                wfs.C_nM[np.arange(len(ind)), :] = wfs.C_nM[ind, :]
                wfs._P_ani = None

                wfs.myocc_n = wfs.myocc_n[ind]
                # update eps?
