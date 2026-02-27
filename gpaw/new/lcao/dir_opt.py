# file for direct optimization in lcao mode
# should follow etdm_lcao.py and etdm_lcao_helper.py of GPAW old

# there should be the initializations and updates, an iterate function.

import numpy as np
from gpaw.new.energies import DFTEnergies
from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.potential import Potential
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.etdm.searchdir import LBFGS
from gpaw.new.etdm import ETDM

from gpaw.directmin.tools import sort_orbitals_according_to_occ

from gpaw.new.etmd.lcao.objfunc_lcao import LCAOObjectiveFunction


# eig_error, wfs_error, energies = self.eigensolver.iterate(
#                ibzwfs, density, potential,
#                self.hamiltonian, pot_calc, energies)

class DirOptLCAO(LCAOEigensolver):
    def __init__(self,
                 *,
                 hamiltonian,
                 excited_state: bool = False,
                 converge_unocc: bool = False,
                 scalapack_params=(None, 1, 1, None)):
        
        self.search_dir: LBFGS | None = None
        # altro?
        self.scalapack = scalapack_params



#--------------------------------------------------------------------
        # this will be called inside scf.py when the mode = 'lcao'
        def initialize(self, ibzwfs, hamiltonian, density, potential):

            # move inside initialize?
            super().__init__(ibzwfs._wfs_u.basis)

        # representation va definita qui? in old viene data

            representation = "full"  # for now

        # devo passare un data iniziale all'objective? ma dovrebbe avere le dimensioni giuste, come il param_vec del test
        # devo mettere in skew_herm una creazione di vettore azzerato quando inizializza se data è vuoto?
        # credo che l'init di skh sia chiamato una volta sola (nell'init di objfunc_etmd) quindi dovrebbe essere fattibile
        # leggere bene se potrebbe azzerare per sbaglio in altri step

        # need to rotate across all bands, in ogni punto devo tenerne conto.

            self.objfunc = LCAOObjectiveFunction(ibzwfs, representation)
            a_u_init = self.objfunc.a_vec_u  # così accedo al vettore creato sopra
            self.etdm = ETDM(self.objfunc, a_u_init, update_ref=True)
           
            # STEPS THAT IN OLD WERE IN ETDM_HELPER_LCAO

            # initialize_orbitals
            # MISSING: coeff read from file, need_init_orb,
            # need canon coeff?
            super().iterate(ibzwfs, density, potential, hamiltonian)

            # MISSING: get functional, randomize orbitals
            
            # calculate occupation numbers (MISSING: if not read from file)
            ibzwfs.get_all_eigs_and_occs(broadcast=False)[1]

            # initial sort orbitals (MISSING: changes with mom)
            sort_orbitals_according_to_occ(ibzwfs.wfs_u)  # credo

            # MISSING: localize for sic
            # MISSING: initialize mom

            set_reference_orbitals(ibzwfs)

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
            # no per ora, usiamo il relative update, MISSING: counter=20 reset
            # self.update_ref_orbitals(ibzwfs, hamiltonian, density)  ########

            # one step of etmd
            self.etdm.one_iteration()

            self.error = 0.0
            # update wavefunctions??
            for wfs in ibzwfs:
                wfs._P_ani = None
            
            error = get_residual_error()
            self.error += error

            return 0.0, self.error, self.etmd.energy

        # attenzione a projector operators: the update must be done each time the coefficients change
        # update densità e energia, dove vengono fatti (inside etdm?)
        # ridurre al minimo flag ma segnare cosa è stato omesso per futuri sviluppi

        def update_to_canonical_orbitals(self, wfs, hamiltonian, potential):
            """
            update to canonical orbitals at the end of the scf cycle
            """

            matrix_calculator = \
                hamiltonian.create_hamiltonian_matrix_calculator(potential)
            h_mm = matrix_calculator.calculate_matrix(wfs)

            wfs.C_nM, eps = rotate_subspace(h_mm, wfs.C_nM)
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
                               representation):
            """
            Calculate residual error of KS equations
            """
            # for now gets recalculated, also move the calculator
            matrix_calculator = \
                hamiltonian.create_hamiltonian_matrix_calculator(potential)
            h_mm = matrix_calculator.calculate_matrix(wfs)

            # get hamiltonian in molecular
            # orbital representation
            c_nm = wfs.C_nM
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
            S_MM = wfs.S_MM
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

            # should it be here or inside etmd?
            if (self.iters % counter == 0 and self.iters > 1):
                self.set_reference_orbitals(ibzwfs)
                self.search_dir.reset()

        def set_ref_orbitals_and_a_vec(self, ibzwfs):

            self.set_reference_orbitals(ibzwfs)
            # qui dovrei azzerare a_vec_u???? o va fatto dentro etdm
            self.a_vec_u = np.zeros_like(self.a_vec_u)
