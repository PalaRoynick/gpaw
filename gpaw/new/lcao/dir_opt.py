import numpy as np
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.density import Density
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.potential import Potential
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.etdm.searchdir import LBFGS
from gpaw.new.etdm.etdm import ETDM
from gpaw.new.etdm.objfunc_lcao import LCAOObjectiveFunction


class DirOptLCAO(LCAOEigensolver):

    def __init__(self,
                 basis,
                 representation,
                 matrix_exp):
        self.search_dir: LBFGS | None = None
        self.name = "etdm-lcao"
        self.representation = representation
        self.matrix_exp = matrix_exp
        self.iter = 0

        super().__init__(basis)

        # initialize_orbitals

        # MISSING: get functional, randomize orbitals
            
        # MISSING: localize for sic
        # MISSING: initialize mom

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
            eig_error, wfs_error, energies = super().iterate(ibzwfs, density,
                                                             potential,
                                                             hamiltonian)
            self.iter += 1

            energies = DFTEnergies(
                coulomb=0.0, zero=0.0, xc=0.0, external=0.0,
                kinetic_correction=0.0, spinorbit=0.0, band=0.0, entropy=0.0,
                extrapolation=0.0)
            return eig_error, wfs_error, energies

        if self.iter == 1:

            self.sort_orbitals_according_to_occ(ibzwfs)

            self.objfunc = LCAOObjectiveFunction(ibzwfs, self.representation,
                                                 self.matrix_exp)
            a = []
            self.ind_up = []
            self.ref_orbitals = []
            for u in range(len(self.objfunc.a_vec_u)):
                a.append(self.objfunc.a_vec_u[u].data)
                self.ind_up.append(self.objfunc.a_vec_u[u].ind_up)

            for wfs in ibzwfs:
                self.ref_orbitals.append(wfs.C_nM.copy())

            a_u_init = np.array(a)
            self.etdm = ETDM(self.objfunc, a_u_init, update_ref=False)
        
            self.precond = False
            self.iter += 1

        # MISSING: set iters for counter=20

        # updates the quantities inside the objective function
        self.objfunc.update_objfunc(density, potential,
                                    pot_calc, ibzwfs,
                                    hamiltonian,
                                    self.ref_orbitals)
        # relative update, MISSING: counter=20 reset

        # preconditioning
        if not self.precond:

            self.prec = self.get_preconditioner(ibzwfs, self.ind_up)
            self.precond = True

        # one step of etdm
        self.etdm.one_iteration(precond=self.prec)

        error = self.objfunc.error

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
                    self.rotate_subspace(h_mm.data, wfs.C_nM.data[occ_n == f, :])
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

    #def set_reference_orbitals(self, ibzwfs):
    #    ndim = ibzwfs.nbands
    #    # these are coefficients
    #    for wfs in ibzwfs:   # ???????
    #        u = self.kpointval(kpt)
    #        self.reference_orbitals[u] = np.copy(wfs.C_nM[:ndim[u]])

    def set_ref_orbitals_and_a_vec(self, ibzwfs):

        self.set_reference_orbitals(ibzwfs)
        self.a_vec_u = np.zeros_like(self.a_vec_u)

    def sort_orbitals_according_to_occ(self, ibzwfs):

        """
        Sort orbitals according to the occupation
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

    def postprocess(self, ibzwfs, density, potential, hamiltonian,
                    maxiter, cc, log):
        for wfs in ibzwfs:
            self.update_to_canonical_orbitals(wfs,
                                              hamiltonian,
                                              potential)

    def get_preconditioner(self, ibzwfs, ind_up):
        """
        Get the preconditioner for the search algorithm:
        approximate analytical hessian and then apply correction
        """
        # for now beta0 = 1, no l-bfgs-p, to be added later
        beta0 = 1.0
        gamma = 0.0

        # add a counter=1000 to recalculate?

        prec = []

        for wfs in ibzwfs:
            w = wfs.weight / (3.0 - wfs.nspins)

            #k = len(ibzwfs.ibz.kpt_kc) + wfs.nspins     # FIX (TOOLS)
            # could one determine this value at the start of everything and
            # use it to declare the size of all these structures?

            hess = self.get_approx_analytical_hessian(
                wfs, ibzwfs.dtype, ind_up)
            precond = np.zeros_like(hess)
            correction = w * gamma * beta0 ** (-1)
            # if self.searchdir_algo.name != 'l-bfgs-p':
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

        f_n = 2 * wfs.myocc_n
        eps_n = wfs.myeig_n

        if ind_up:
            il1 = list(ind_up[0])
        else:
            il1 = list(np.tril_indices(eps_n.shape[0], -1))  # from dirmin tools

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
