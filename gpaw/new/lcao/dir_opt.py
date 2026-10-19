import numpy as np
import warnings
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.density import Density
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.potential import Potential
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.etdm.etdm import ETDM
from gpaw.new.etdm.objfunc_lcao import LCAOObjectiveFunction
from gpaw.new.etdm.tools import sort_orbitals_according_to_occ, \
    sort_orbitals_according_to_energies


class DirOptLCAO(LCAOEigensolver):

    def __init__(self,
                 basis,
                 occ_calc,
                 mixer,
                 representation='sparse',
                 matrix_exp='egdecomp',
                 searchdir_algo_name='l-bfgs',
                 linesearch_algo={'name': 'max-step',
                                  'max_step': 0.25},
                 excited_state=False
                 ):

        """Class for direct orbital optimization in LCAO mode

        Args:
            basis : basis set
            occ_calc : occupation number calculator
            representation: str
                How to store the elements of the anti-Hermitian matrix for the
                matrix exponential. Can be one of 'sparse' (default), 'full',
                'u-invar'. The latter can be used only for unitary invariant
                functionals, such as Kohn-Sham functionals when the occupied
                orbitals have the same occupation number, but not for orbital
                density dependent functionals, such as when using PZ-SIC.
                Defaults to 'sparse'.
            matrix_exp: str
                Algorithm for calculating the matrix exponential and the
                gradient with respect to the elements of the anti-Hermitian
                matrix for the exponential transformation.
                Can be one of 'pade-approx' or 'egdecomp'
                Defaults to 'egdecomp'.
            searchdir_algo_name: str
                Search direction algorithm. Can be one of the algorithms
                available in searchdir.py:
                'l-bfgs': Limited-memory BFGS
                'l-bfgs-p': Limited-memory BFGS with preconditioner presented
                    in :doi:`10.1016/j.cpc.2021.108047`
                'LSR1': limited-memory SR1 algorithm presented in
                    :doi:`10.1021/acs.jctc.0c00597`
                Defaults to 'l-bfgs'.
            linesearch_algo: dict
                Line search algorithm, only max step is implemented.
                'max-step': The quasi-Newton step is scaled if it exceeds a
                    maximum step length.
                    The default maximum step length is 0.25, and can be changed
                    by supplying a dictionary:
                    {'name': 'max-step', 'max_step': max_step}, where max_step
                    should be a float
                Defaults to {'name': 'max-step', 'max_step': 0.25}.
        """
        self.name = 'etdm-lcao'
        self.representation = representation
        self.matrix_exp = matrix_exp
        self.searchdir_algo_name = searchdir_algo_name
        self.occ_calc = occ_calc
        self.mixer = mixer
        self.linesearch_algo = linesearch_algo
        self.iter = 0
        self.update_ref_orbitals_counter = 20
        self.excited_state = excited_state
        # This should be true if the excited state is false
        self.need_init_orbs = True

        if self.excited_state:
            self.need_init_orbs = False
            print('Excited state is true!!')

        super().__init__(basis)

        # MISSING: get functional, randomize orbitals
        # MISSING: localize for sic
        # MISSING: initialize mom

    def initialize(self, ibzwfs, density, potential, hamiltonian, pot_calc):
        """
        Initialization. One iteration of eigensolver, calculation of
        occupation numbers, reordering of orbitals and creation of
        objective function and etdm classes.
        Set first reference orbitals and keyword for preconditioner
        """

        # First diagonalization
        # this could be skipped in certain circumstances, such as during
        # an excited state calculation (we do not need initial orbitals in
        # that case)
        if not self.excited_state:
            super().iterate(ibzwfs, density, potential, hamiltonian)
        else:
            # i need to add a keyword for the type of orthonormalization
            ibzwfs.orthonormalize()

        # this also does not need to be done if they are read
        nelectrons = density.nvalence - density.charge + pot_calc.charge
        ibzwfs.calculate_occs(self.occ_calc, nelectrons)
        # is this a repetition?
        # sort_orbitals_according_to_occ(ibzwfs)

        # initial sorting of the orbitals to be put here,
        # there can't be gaps in the occupations
        self.initial_sort_orbitals(ibzwfs, self.occ_calc)

        # If excited state then i have to overwrite the
        # default occupation numbers stored in occ_calc
        # Initialize objective function
        self.objfunc = LCAOObjectiveFunction(ibzwfs, self.representation,
                                             self.matrix_exp, self.mixer)

        for wfs in ibzwfs:
            f_unique = np.unique(wfs.myocc_n)
            if len(f_unique) > 2 and self.representation == 'u-invar':
                warnings.warn("Use representation == 'sparse' when "
                              "there are unequally occupied orbitals "
                              "as the functional is not unitary invariant")

        # Set a_u_init and indices
        a = []
        self.ind_up = []
        shape = []
        for u in range(len(self.objfunc.a_vec_u)):
            shape.append(len(self.objfunc.a_vec_u[u].data))
            a.append(self.objfunc.a_vec_u[u].data)
            self.ind_up.append(self.objfunc.a_vec_u[u].ind_up)

        # initialize mom reference orbitals
        #self.initialize_mom_reference_orbitals(ibzwfs, density)

        # Set reference orbitals
        self.ref_orbitals = []
        for wfs in ibzwfs:
            self.ref_orbitals.append(wfs.C_nM.copy())

        # there is a problem when the two spin channels
        # have different occupations, i have to fix the shape of this
        # the problem is treating it as an array (which must have equal lenghts)
        # also when we get to the gradient it wants it as an array.....
        # vedi anche in old g_vec_u non è un array ma un dictionary,
        # quindi non ho il problema delle shape.
        # devo sistemare questo.
        a_u_init = np.zeros((len(self.objfunc.a_vec_u), max(shape)))
        # a_u_init = np.array(a)

        # Initialize ETDM
        self.etdm = ETDM(self.objfunc, a_u_init,
                         mode='lcao',
                         update_ref=False,
                         searchdir_algo_name=self.searchdir_algo_name,
                         linesearch_algo=self.linesearch_algo,
                         )

        self.precond = False
        self.iter += 1

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

        # Initialization
        if self.iter == 0:
            self.initialize(ibzwfs, density, potential, hamiltonian, pot_calc)
        # Dictionary to "deliver" main quantities to
        # various sections of the code
        shuttle = {'ibzwfs': ibzwfs,
                   'density': density,
                   'ref_orbitals': self.ref_orbitals,
                   'pot_calc': pot_calc,
                   'potential': potential,
                   'hamiltonian': hamiltonian,
                   'a_vec_u': None
                   }
        if shuttle['a_vec_u'] is not None:
            self.objfunc._a_vec_u = shuttle['a_vec_u']
        else:
            shuttle['a_vec_u'] = self.objfunc._a_vec_u

        # If first iteration or the reference orbitals have been updated
        # calculate energy and gradient here
        # if self.iter == 1:
        #     self.etdm._energy, self.etdm._gradient = \
        #         self.objfunc._calc_energy_and_gradient(shuttle)
        #     self.old_gradient = None
        # else:
        #     self.etdm._gradient = self.old_gradient

        # print('grad before')
        # print(self.etdm._gradient[0,:10])

        # Update reference orbitals
        # if self.excited_state:
            # print(ibzwfs.get_all_eigs_and_occs()[1])
        self.update_ref_orbitals(ibzwfs)

        if self.iter == 1:
            self.etdm._energy, self.etdm._gradient = \
                self.objfunc._calc_energy_and_gradient(shuttle)
            self.old_gradient = None
        else:
            self.etdm._gradient = self.old_gradient

        # Preconditioning
        if not self.precond or self.searchdir_algo_name == 'l-bfgs-p':
            self.prec = self.get_preconditioner(ibzwfs, self.ind_up)
            self.precond = True
        # print('precond')
        # print(self.prec[0,:10])

        # One step of etdm
        self.etdm.one_iteration(shuttle, precond=self.prec)

        error = self.objfunc.error

        # Calculate the gradient for the next iteration
        self.etdm._gradient = None
        self.etdm.gradient = shuttle
        self.old_gradient = self.etdm.gradient

        # print('grad after')
        # print(self.etdm.gradient[0,:10])
        # print(self.etdm._gradient[0,:10])

        self.iter += 1

        # error on the density
        self.dens_error = self.objfunc.dens_error
        self.density = self.objfunc.density
        self.potential = self.objfunc.potential

        return 0.0, error, self.objfunc.tot_energy

    def update_to_canonical_orbitals(self, wfs, hamiltonian,
                                     potential,
                                     update_orb_to_canonical=False):
        """
        Update to canonical orbitals at the end of the scf cycle
        """

        matrix_calculator = \
            hamiltonian.create_hamiltonian_matrix_calculator(potential)
        h_mm = matrix_calculator.calculate_matrix(wfs)

        # REMOVE FIRST OPTION
        if update_orb_to_canonical:
            # Diagonalize entire Hamiltonian matrix
            wfs.C_nM.data, wfs.eig_n = \
                self.rotate_subspace(h_mm.data, wfs.C_nM.data)
        else:
            # Diagonalize equally occupied subspaces separately
            occ_n = wfs.myocc_n
            f_unique = np.unique(occ_n)
            for f in f_unique:
                wfs.C_nM.data[occ_n == f, :], wfs.eig_n[occ_n == f] = \
                    self.rotate_subspace(h_mm.data,
                                         wfs.C_nM.data[occ_n == f, :])

        # missing sic option

        # update projections
        wfs._P_ani = None

    def rotate_subspace(self, h_mm, c_nm):
        """
        Choose canonical orbitals
        """
        l_nn = c_nm @ h_mm @ c_nm.conj().T
        eps, w = np.linalg.eigh(l_nn)
        return w.T.conj() @ c_nm, eps

    def update_ref_orbitals(self, ibzwfs):
        """
        Update reference orbitals and reset search direction algorithm
        """
        # MISSING: check if canonical orbitals are needed, or mom restart,

        counter = self.update_ref_orbitals_counter

        if (self.iter % counter == 0 and self.iter > 1):
            # set reference orbitals
            for u, wfs in enumerate(ibzwfs):
                n_dim = self.objfunc._a_vec_u[u]._ndim
                self.ref_orbitals[u].data = np.copy(wfs.C_nM.data[:n_dim])
                # self.etdm.a_u[u] = np.zeros_like(
                #     self.etdm.a_u[u])
                self.objfunc._a_vec_u[u].data = np.zeros_like(
                    self.objfunc._a_vec_u[u].data)

            self.iter = 1
            self.etdm.searchdir_algo.reset()

    def postprocess(self, ibzwfs, density, potential, hamiltonian,
                    maxiter, cc, log):

        for wfs in ibzwfs:
            self.update_to_canonical_orbitals(wfs,
                                              hamiltonian,
                                              potential)

        if self.excited_state:
            # reorder orbitals according to energies
            sort_orbitals_according_to_energies(hamiltonian, ibzwfs)
            # reorder occupation numbers
            for i, wfs in enumerate(ibzwfs):
                wfs._occ_n = self.initial_occupation_numbers[i]

        # put this back to zero
        self.iter = 0

    def get_preconditioner(self, ibzwfs, ind_up):
        """
        Get the preconditioner for the search algorithm:
        approximate analytical hessian and then apply correction
        """
        if self.searchdir_algo_name == 'l-bfgs-p':
            beta0 = self.etdm.searchdir_algo.beta0
            gamma = 0.25
        else:
            beta0 = 1.0
            gamma = 0.0

        # print('self.beta')
        # print(beta0)
        prec = []

        for wfs in ibzwfs:
            w = ((2.0 / wfs.nspins) * wfs.weight) / (3.0 - wfs.nspins)
            if not self.precond:  # only calculate approx. hessian once
                self.hess = self.get_approx_analytical_hessian(
                    wfs, ibzwfs.dtype, ind_up)
            hess = self.hess

            precond = np.zeros_like(hess)
            correction = w * gamma * beta0 ** (-1)
            # print('beta0')
            # print(beta0)
            if self.searchdir_algo_name != 'l-bfgs-p':
                correction = np.zeros_like(hess)
                zeros = abs(hess) < 1.0e-4
                correction[zeros] = 1.0
            precond += 1. / ((1 - gamma) * hess.real + correction)
            if ibzwfs.dtype == complex:
                precond += 1.j / ((1 - gamma) * hess.imag + correction)

            prec.append(precond)

        return np.array(prec)

    def get_approx_analytical_hessian(self, wfs, dtype, ind_up):
        """
        Calculate the following diagonal approximation to the Hessian:
        h_{lm, lm} = -2.0 * (eps_n[l] - eps_n[m]) * (f[l] - f[m])
        """
        k = 2 / wfs.nspins
        f_n = k * wfs.myocc_n
        eps_n = wfs.myeig_n

        if ind_up:
            il1 = list(ind_up[0])
        else:
            il1 = list(np.tril_indices(eps_n.shape[0], -1))  # from dirmintools

        hess = np.zeros(len(il1[0]), dtype=dtype)
        x = 0
        for n, m in zip(*il1):
            df = f_n[n] - f_n[m]
            hess[x] = -2.0 * (eps_n[n] - eps_n[m]) * df
            if abs(hess[x]) < 1.0e-10:
                hess[x] = 0.0
            if dtype == complex:
                hess[x] += 1.0j * hess[x]
            x += 1

        return hess

    def initial_sort_orbitals(self, ibzwfs, occ_calc):
        """ Sort the orbitals to avoid gaps in the occupations
        """

        # i removed any mom options for now

        # we use these at the end 
        self.initial_occupation_numbers = [wfs.occ_n for wfs in ibzwfs]
        sort_orbitals_according_to_occ(ibzwfs)

    def initialize_mom_reference_orbitals(self, ibzwfs, density, occ_calc):
        # Reinitialize the MOM reference orbitals
        # after orthogonalization/localization
        occ_name = getattr(occ_calc.occ, 'name', None)

        if occ_name == 'mom':
            wfs.occupations.initialize_reference_orbitals()
            wfs.calculate_occupation_numbers(dens.fixed)
