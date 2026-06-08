import numpy as np 
from gpaw.transformers import Transformer
from gpaw.poisson import PoissonSolver
from gpaw.directmin.tools import gramschmidt_lcao, get_n_occ
from gpaw.mpi import world
from gpaw.utilities import unpack_density, pack_density
from gpaw.fd_operators import Laplace
from gpaw.utilities.partition import AtomPartition
from gpaw.utilities.tools import tri2full, pick
from gpaw.lfc import LFC
from copy import deepcopy


class TwoElectronIntegrals(object):
    def __init__(self,
                 calc,  # we give it the calculator object
                 symmetrize_pds=False,
                 xcdc=False,
                 lapl_hkin=True,
                 h=1e-5,
                 hamilton_how=1,
                 poisson_solver='FPS'):
        

        # a lot of stuff here, ignore for now

        # things obtained from the calculator, ADAPT TO NEW

        self.world = calc.world  # non so cosa fa
        self.name = '2e integrals'
        self.calc = calc
        self.ibzwfs = calc.wfs.ibzwfs
        self.eigensolver = calc._dft.scf_loop.eigensolver
        #self.ham = calc.hamiltonian
        self.ham = self.calc._dft.scf_loop.hamiltonian
        self.dens = calc.density
        self.potential = calc._dft.potential
        self.pot_calc = calc._dft.pot_calc


        # trovare versione di new
        self.xc = calc.hamiltonian.xc
        self.setups = calc.wfs.setups
        self.cgd = calc.wfs.gd
        self.finegd = self.pot_calc.fine_grid._gd
        self.density = calc._dft.density
        #self.ghat = self.density.ghat
        #self.ghat = pot_calc.ghat_aLr  # ????????
        #self.ghat = calc.dft.pot_calc.poisson_solver.ghat_aLh
        self.ghat = LFC(self.finegd, [setup.ghat_l for setup in self.setups])
        self.scaled_atom_pos = self.ibzwfs._wfs_u[0].relpos_ac

        atomdist = self.density.D_asii.layout.atomdist
        atom_partition = AtomPartition(atomdist.comm, atomdist.rank_a)

        #rank_a = self.ibzwfs.gd.get_ranks_from_positions(self.ibzwfs.spos_ac)
        #atom_partition = AtomPartition(self.ibzwfs.gd.comm, rank_a, name='gd')
        
        self.ghat.set_positions(self.scaled_atom_pos, atom_partition)

        self.timer = calc.wfs.timer
        self.symmetrize_pds = symmetrize_pds
        self.h = h
        self.xcdc = xcdc
        self.hamilton_how = hamilton_how
        self.lapl_hkin = lapl_hkin

        # interpolator, cosa fa pure questo
        self.interpolator = Transformer(self.cgd, self.finegd, 3)

        # azzeramenti vari
        self.wf_ref_old = None
        self.wf_ref = None
        self.P_ani_old = None
        self.P_ani = None
        self.H_nn = None
        self.TwoeI = None
        self.XC_DCI = None
        self.rho_MM = None

        # section where it checkes if there are the basis functions?
        # split for lcao e pw
        if hasattr(calc.wfs, 'basis'):
            self.bfs = self.ibzwfs.basis
            self.lcao = True
            self.P_aqMi = None
        else:   # look more into the pw part, ADAPT
            self.lcao = False
            if self.ibzwfs.mode == 'pw':
                if lapl_hkin:
                    self.kin = Laplace(self.cgd, -0.5, 3, self.ibzwfs.dtype)
                self.G2_G = None
        self.n_bands = self.ibzwfs.nbands

        #assert poisson solver, FPS has to be used

        if self.ibzwfs.mode == 'pw':
            self.poiss = PoissonSolver(use_charge_center=True, 
                                       use_charged_periodic_corrections=True)
        else:
            self.poiss = PoissonSolver(use_charge_center=True)    

        self.poiss.set_grid_descriptor(self.finegd)

        # UPDATE WF REF: boh cose che posso copiare adattate probabilmente

    def update_wf_ref(self, ao_basis=False):
        self.wf_ref_old = deepcopy(self.wf_ref)
        self.P_ani_old = deepcopy(self.P_ani)
        if self.lcao:
            self.P_aqMi = self.ibzwfs.P_aqMi  # questo sembra aver cambiato nome
            if ao_basis:  # do we need it?
                self.wf_ref = np.identity(self.ibzwfs._wfs_u[0].C_nM.data.shape[0])
                self.wf_ref = gramschmidt_lcao(
                    self.wf_ref, self.ibzwfs._wfs_u[0].S_MM)
            else:
                self.wf_ref = deepcopy(self.ibzwfs._wfs_u[0].C_nM.data)

        else:  # anche qui da capire come funzionano le pw
            if self.ibzwfs.mode == 'pw':
                pw = self.pot_calc.pw
                self.G2_G = deepcopy(pw.G_plus_k_Gv[0])  # NON SO COME è IN NEW
                psit_nG = self.cgd.zeros(
                    self.n_bands, dtype=self.ibzwfs.dtype, global_array=False) # Default. Correct?
                
                # USING BACKWARDS COMP!! USING FAKE OBJECT!!!!
                # CAREFUL!!!!!
                for i in range(self.n_bands):
                    psit_nG[i] = self.calc.wfs._get_wave_function_array(0, i)
            else:
                psit_nG = self.ibzwfs._wfs_u[0].psit_nX
            self.wf_ref = deepcopy(psit_nG)
        
        self.P_ani = {a: P_ni.copy() for a, P_ni in
                      self.ibzwfs._wfs_u[0].P_ani.items()}

        # OBTAIN HOPPING MATRIX
    def obtain_hopping_matrix(
        self, kinetic_one_particle_directly=False):
        if self.lcao:
            if kinetic_one_particle_directly:
                H_MM = self.ibzwfs._wfs_u.T_qMM[0]
            else:
                # questa def in new non c'è, redirect ad hamiltonian direi
                #H_MM = self.eigensolver.calculate_hamiltonian_matrix(
                #    self.ham, self.ibzwfs, self.ibzwfs.kpt_u[0])
                self.matrix_calculator = \
                        self.hamiltonian.create_hamiltonian_matrix_calculator(self.potential)
                H_MM = self.matrix_calculator.calculate_matrix(self.ibzwfs._wfs_u[0])

            H_MM.trilfull()
            self.H_nn = self.wf_ref.conj() @ H_MM.conj() @ self.wf_ref.T
        
        # here definitions from eigensolver which where written especially for 
        # this are used, i have to choose where to move them

        else:  # we only do hamilton_how=1
            # ADAPT/FIX
            grad_knG = self.get_gradients_2(self.ham, self.ibzwfs, 
                                            scalewithocc=False)
            for k, wfs in enumerate(self.ibzwfs._wfs_u):
                n_occ = int(wfs.myocc_n[0])
                dim = int(self.n_bands - n_occ)
                grad_knG[k][n_occ:n_occ + dim] = \
                    self.get_gradients_unocc_kpt(
                        self.ham, self.ibzwfs, wfs)
                self.H_nn = self.cgd.integrate(
                    wfs.psit_nX.data.read()[:], grad_knG[k][:], True)

    # ROTATE RANK 2 TENSOR

    # ROTATE RANK 4 TENSOR 

    # EVALUATE_Q_FDPW: FIXXXXXXX
    def evaluate_Q_FDPW(self):
        wf_ref = self.wf_ref.copy()
        self.ibzwfs.pt.integrate(wf_ref, self.P_ani, self.ibzwfs.kpt_u[0].q)

        ns = len(wf_ref)
        self.Q = np.zeros((len(wf_ref), len(wf_ref)))
        for i in range(ns):
            for j in range(ns):
                val = self.ibzwfs.gd.integrate(wf_ref[i], self.wf_ref_old[j])
                proj = 0.0
                for a, P_ni in self.P_ani.items():
                    dO = self.setups[a].dO_ii
                    dPO = P_ni[i] @ dO
                    proj += dPO @ self.P_ani_old[a][j]
                proj = self.ibzwfs.gd.comm.sum(proj)
                #
                self.Q[i, j] = val + proj

    #EVALUATE TWO EL INTEGRALS
    def evaluate_two_e_integrals(self, indices=None, old_wf=False):

        if indices is None:
            indices = list(range(self.n_bands))
        wfs = self.ibzwfs._wfs_u[0]
        ibzwfs = self.ibzwfs
        f_n = wfs.myocc_n

        self.TwoeI = np.zeros((self.n_bands, self.n_bands,
                           self.n_bands, self.n_bands))
        
        # do we add these later?
        #if self.xcdc:
        #    self.XC_DCI = np.zeros((self.n_bands, self.n_bands,
        #                            self.n_bands, self.n_bands))
        
        self.rho_MM = np.zeros((self.n_bands, self.n_bands))

        indexes = self.make_indices(self.n_bands)

        for quad in indexes:

            i = quad[0]
            j = quad[1]
            k = quad[2]
            l = quad[3]

            if self.lcao:
                C_i = self.wf_ref[i] if not old_wf else self.wf_ref_old[i]
                C_j = self.wf_ref[j] if not old_wf else self.wf_ref_old[j]
                C_k = self.wf_ref[k] if not old_wf else self.wf_ref_old[k]
                C_l = self.wf_ref[l] if not old_wf else self.wf_ref_old[l]
                
            else:
                C_i = i
                C_j = j
                C_k = k
                C_l = l

            rhot_gij, Q_aLij, D_apij, rho_MMij = self.get_pair_density(
                C_i, C_j, wfs, ibzwfs, self.setups)

            if i == j and f_n[i] > 0.0:
                if self.lcao:
                    self.rho_MM += rho_MMij * f_n[i]

            # ignore the xcdc?

            self.ghat.add(rhot_gij, Q_aLij)

            vHt_gij = self.finegd.zeros()
            self.poiss.solve(vHt_gij, rhot_gij, zero_initial_phi=False,
                             timer=self.timer, charge=None)  # controllare come è cambiato
            
            rhot_gkl, Q_aLkl, D_apkl = self.get_pair_density(
                        C_k, C_l, wfs, ibzwfs, self.setups)[: 3]
            
            self.ghat.add(rhot_gkl, Q_aLkl)
            ec = self.finegd.integrate(
                        rhot_gkl, vHt_gij, global_integral=False)
            ec = self.finegd.comm.sum(ec)


            ec_paw = 0.0
            for a in self.P_ani.keys():
                M_pp = self.setups[a].M_pp
                ec_paw += np.dot(
                    np.dot(D_apij[a], M_pp), D_apkl[a])
            ec_paw = world.sum(ec_paw)

            I = ec + 2.0 * ec_paw
            sign = 1 if self.symmetrize_pds else -1
            self.TwoeI[i][j][k][l] = I
            self.TwoeI[k][l][i][j] = I
            self.TwoeI[j][i][l][k] = I
            self.TwoeI[l][k][j][i] = I
            self.TwoeI[i][j][l][k] = I * sign
            self.TwoeI[j][i][k][l] = I * sign
            self.TwoeI[k][l][j][i] = I * sign
            self.TwoeI[l][k][i][j] = I * sign

    # EVAL AND DUMPTWO EL INTEGRALS

    # GET PAIR DENSITY

#------these are for testing?

    # TEST BASIS CHANGE

    # COMPARE TO TOTAL COULOMB

    # write rank 2 and rank 4 tensors (these are to make the output files i guess)

    def get_pair_density(self, C1, C2, kpt, wfs, setup):

        if self.lcao:
            if self.symmetrize_pds:
                rho_MMn = 0.5 * np.outer(C1.conj(), C2) \
                          + 0.5 * np.outer(C2.conj(), C1)
            else:
                rho_MMn = np.outer(C1.conj(), C2)
            nt = self.cgd.zeros()
            self.bfs.construct_density(rho_MMn, nt, kpt.q)
            nt_n = self.finegd.zeros()
            self.interpolator.apply(nt, nt_n)

            # remove dictionaries?
            D_apn = {}
            Q_aLn = {}
            for a in self.P_aqMi.keys():
                P_Mi = self.P_aqMi[a][kpt.q]
                rhoP_Mi = rho_MMn @ P_Mi
                D_iin = P_Mi.T.conj() @ rhoP_Mi
                D_apn[a] = D_pn = pack_density(np.real(D_iin))
                Q_aLn[a] = np.dot(D_pn, setup[a].Delta_pL)

            return nt_n, Q_aLn, D_apn, rho_MMn

        else:
            psit1_G = pick(self.wf_ref, C1)
            psit2_G = pick(self.wf_ref, C2)

            nt_G = self.cgd.zeros()
            np.multiply(psit1_G.conj(), psit2_G, nt_G)
            nt_g = self.finegd.zeros()
            self.interpolator.apply(nt_G, nt_g)

            # remove dictionaries?
            D_ap = {}
            Q_aL = {}
            for a, P_ni in self.P_ani.items():
                assert P_ni.dtype == float
                P1_i = P_ni[C1]
                P2_i = P_ni[C2]
                D_ii = np.outer(P1_i.conj(), P2_i)
                D_ap[a] = D_p = pack_density(D_ii)
                Q_aL[a] = np.dot(D_p, setup[a].Delta_pL)

            return nt_g, Q_aL, D_ap, None

    def make_indices(self, nbands):

        import itertools

        iter_array = np.array(list(itertools.product(range(nbands), repeat=4)),
                              dtype=int)
        #print(iter_array)
        #print(iter_array.size)
        #print(iter_array.shape)

        # now i need to apply the appropriate conditions to reduce the amount of elements
        # so

        cond1 = iter_array[:, 1] >= iter_array[:, 0]
        # print("cond1", cond1)
        cond2 = iter_array[:, 2] >= iter_array[:, 0]
        # print("cond2", cond2)
        cond3 = iter_array[:, 3] >= iter_array[:, 2]
        # print("cond3", cond3)

        # now these conditions have to be combined: all three should be satisfied at once

        tot_cond1 = np.logical_and(cond1, cond2)
        # print("tot_cond1", tot_cond1)

        tot_cond2 = np.logical_and(tot_cond1, cond3)
        # print("tot_cond2", tot_cond2)

        maybe_solution = iter_array[tot_cond2]
        # print(maybe_solution)
        # print(maybe_solution.shape)
        # print(maybe_solution.size)

        return maybe_solution
    
#############################################################

    def get_gradients_2(self, ham, ibzwfs, scalewithocc=True):

        """
        FROM ETDM! NEEDED FOR CALCULATION OF HAMILTONIAN MATRIX IN
        two_electron_integrals.py!
        DO NOT MERGE INTO MASTER!!! I REPEAT: DO NOT MERGE INTO MASTER!!!
        calculate gradient dE/dpsi
        :return: H |psi_i>
        """

        grad_knG = {}
        n_kps = len(ibzwfs.ibz)

        for wfs in ibzwfs:
            grad_knG[n_kps * wfs.spin + wfs.q] = \
                self.get_gradients_from_one_k_point_2(
                    ham, ibzwfs, wfs, scalewithocc)

        return grad_knG

    def get_gradients_from_one_k_point_2(
            self, ham, ibzwfs, wfs, scalewithocc=True):
        """
        FROM ETDM! NEEDED FOR CALCULATION OF HAMILTONIAN MATRIX IN
        two_electron_integrals.py!
        DO NOT MERGE INTO MASTER!!! I REPEAT: DO NOT MERGE INTO MASTER!!!
        calculate gradient dE/dpsi for one k-point
        :return: H |psi_i>
        """

        nbands = ibzwfs.nbands
        # from backwards compatibility
        Hpsi_nG = np.empty((nbands,) + self.ibzwfs._wfs_u
                           [self.ibzwfs.u_q[wfs.q]]
                           .psit_nX.data.shape[1:],
                           complex if self.ibzwfs.mode == 'pw' else self.dtype)
        desc = self.ibzwfs._wfs_u[self.ibzwfs.u_q[wfs.q]].psit_nX.desc
        self.ham.apply(
            self.potential.vt_sR,
            None,
            self.ibzwfs,  # needed for hybrids
            getattr(self.density, 'D_asii', None),  # needed for hybrids
            desc.from_data(data=wfs.psit_nX.data.read()),  # CAMBIARE, nG non esiste!
            desc.from_data(data=Hpsi_nG),
            wfs.spin)
        
        #Hpsi_nG = wfs.empty(nbands, q=kpt.q)
        #wfs.apply_pseudo_hamiltonian(kpt, ham, kpt.psit_nG, Hpsi_nG)

        c_axi = {}
        for a, P_xi in wfs.P_ani.items():
            dH_ii = unpack_density(self.potential.gather()[0][a][wfs.spin])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi

        # not sure about this:
        self.ham.xc.xc.add_correction(
            wfs, wfs.psit_nX, Hpsi_nG, wfs.P_ani, c_axi, n_x=None,
            calculate_change=False)
        # add projectors to the H|psi_i>

        from gpaw.new.backwards_compatibility import PT
        pt = PT(ibzwfs)
        pt.add(Hpsi_nG, c_axi, wfs.q)
        # scale with occupation numbers
        if scalewithocc:
            for i, f in enumerate(wfs.myocc_n):
                Hpsi_nG[i] *= f
        return Hpsi_nG

    def get_gradients_unocc_kpt(self, ham, ibzwfs, wfs):
        """
        FROM ETDM! NEEDED FOR CALCULATION OF HAMILTONIAN MATRIX IN
        two_electron_integrals.py!
        DO NOT MERGE INTO MASTER!!! I REPEAT: DO NOT MERGE INTO MASTER!!!
        calculate gradient vector for unoccupied orbitals

        :param ham:
        :param wfs:
        :param kpt:
        :return:
        """
        orbital_dependent = ham.xc.orbital_dependent

        n_occ = int(wfs.myocc_n[0])
        if not orbital_dependent:
            dim = int(self.n_bands - n_occ)
            n0 = int(n_occ)
        else:
            dim = int(self.n_bands)
            n0 = 0

        # calculate gradients:
        psi = wfs.psit_nX.data.read()[n0:n0 + dim].copy()

        from gpaw.new.backwards_compatibility import PT
        pt = PT(ibzwfs)

        P1_ai = pt.dict(shape=dim)
        pt.integrate(psi, P1_ai, wfs.q)

        # from backwards compatibility
        Hpsi_nG = np.empty((self.n_bands,) + self.ibzwfs._wfs_u
                           [self.ibzwfs.u_q[wfs.q]]
                           .psit_nX.data.shape[1:],
                           complex if self.ibzwfs.mode == 'pw' else self.dtype)

        #Hpsi_nG = wfs.empty(dim, q=wfs.q)
        desc = self.ibzwfs._wfs_u[self.ibzwfs.u_q[wfs.q]].psit_nX.desc
        self.ham.apply(
            self.potential.vt_sR,
            None,
            self.ibzwfs,  # needed for hybrids
            getattr(self.density, 'D_asii', None),  # needed for hybrids
            desc.from_data(data=wfs.psit_nX.data.read()),
            desc.from_data(data=Hpsi_nG),
            wfs.spin)
        #wfs.apply_pseudo_hamiltonian(kpt, ham, psi, Hpsi_nG)
        
        c_axi = {}
        for a, P_xi in P1_ai.items():
            dH_ii = unpack_density(self.potential.gather()[0][a][wfs.spin])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi
        # not sure about this:
        ham.xc.xc.add_correction(wfs, psi, Hpsi_nG,
                                 P1_ai, c_axi, n_x=None,
                                 calculate_change=False)
        # add projectors to the H|psi_i>
        pt.add(Hpsi_nG, c_axi, wfs.q)

        if orbital_dependent:
            Hpsi_nG = Hpsi_nG[n_occ:n_occ + dim]

        return Hpsi_nG
