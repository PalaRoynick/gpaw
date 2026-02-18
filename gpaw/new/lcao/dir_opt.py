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

#from gpaw.new.etmd.lcao import LCAOObjectiveFunction


#eig_error, wfs_error, energies = self.eigensolver.iterate(
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

        super().__init__(ibzwfs._wfs_u.basis) #si chiama così?

#--------------------------------------------------------------------
        # this will be called inside scf.py when the mode = 'lcao'
        def initialize(self, ibzwfs, hamiltonian, density, potential):
        
        # representation va definita qui?

            # representation =

        # devo passare un data iniziale all'objective? ma dovrebbe avere le dimensioni giuste, come il param_vec del test
        # devo mettere in skew_herm una creazione di vettore azzerato quando inizializza se data è vuoto?
        # credo che l'init di skh sia chiamato una volta sola (nell'init di objfunc_etmd) quindi dovrebbe essere fattibile
        # leggere bene se potrebbe azzerare per sbaglio in altri step

        # need to rotate across all bands, in ogni punto devo tenerne conto.

            objfunc = LCAOObjectiveFunction(ibzwfs, representation)       #DA SCRIVERE
            a_u_init = objfunc.a_vec_u  #così accedo al vettore creato sopra
            self.etdm = ETDM(objfunc, a_u_init, update_ref=True) # questa a_u_init come lo definisco?
           
            ex_initialize_helper_lcao(ibzwfs, hamiltonian, density, potential)
           
        # to be merged in the definition above
        def ex_initialize_helper_lcao(self, ibzwfs, hamiltonian, density, potential):

        # steps that in old were in etdm_helper_lcao.py

            # initialize_orbitals
            # MISSING: coeff read from file, need_init_orb, 
            # need canon coeff
            super().iterate(ibzwfs,density,potential,hamiltonian)

            # MISSING: get functional, randomize orbitals
            
            # calculate occupation numbers (MISSING: if not read from file)
            ibzwfs.get_all_eigs_and_occs(broadcast=False)[1] #if not read from file flag

            # initial sort orbitals (MISSING: changes with mom)
            sort_orbitals_according_to_occ(ibzwfs.wfs_u) #credo

            # MISSING: localize for sic

            # MISSING: initialize mom

        # set reference orbitals, serve anche sotto
        # CONTROLLARE
        def set_reference_orbitals(self, ibzwfs):
            ndim = ibzwfs.nbands
            # these are coefficients
            for wfs in ibzwfs._wfs_u:   #???????
                u = self.kpointval(kpt)
                self.reference_orbitals[u] = np.copy(kpt.C_nM[:ndim[u]])
                                                     #wfs.C_nM
                                    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
        def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian,
                pot_calc,       #?????
                energies) -> tuple[float, float, DFTEnergies]:
            
            """
            One iteration of direct optimization for LCAO
            """

            # MISSING: set iters for counter=20

            # devo costruire matrix calculator qui? nell'lcao eigensolver è dentro iterate, cosa è potential
            # non so se serve ora
            #matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(potential)

            # no per ora, usiamo il relative update, MISSING: counter=20 reset
            #self.update_ref_orbitals(ibzwfs, hamiltonian, density)  ########

            # one step of etmd
            self.etdm.one_iteration()

        # attenzione a projector operators: the update must be done each time the coefficients change
        # update densità e energia, dove vengono fatti (inside etdm?)
        # ridurre al minimo flag ma segnare cosa è stato omesso per futuri sviluppi

        def update_ref_orbitals(self, ibzwfs, hamiltonian, density):

            """
            Update reference orbitals and reset search direction algorithm
            """

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
