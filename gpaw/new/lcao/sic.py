"""Get PZ SIC energy and gradient"""

from gpaw.lfc import LFC
import numpy as np
from gpaw.poisson import PoissonSolver
from gpaw.utilities import pack_density
# from gpaw.new.poisson import PoissonSolver


class PZSICLCAO:
    """"""

    def __init__(
        self, ibzwfs, density, pot_calc, use_coarse_grid=True, poisson_solver="FPS"
    ):  
        
        self.setups = ibzwfs._wfs_u[0].setups
        self.basis_functions = ibzwfs._wfs_u[0].basis
        self.scaled_atom_pos = ibzwfs._wfs_u[0].relpos_ac

        # self.grid is UGDesc (presumably)
        self.grid = density.nt_sR.desc
        self.coarse_gd = self.grid._gd
        self.fine_gd = pot_calc.fine_grid._gd

        self.use_coarse_grid = use_coarse_grid

        # if self.use_coarse_grid:

        # LFC allows us to describe the augmentation refion around the atom. It contains
        # functions to couple information on the regular mesh with information on the 
        # atomic radial grid
        self.ghat_coarse_gd = LFC(
            self.coarse_gd,
            [setup.ghat_l for setup in self.setups],
            integral=np.sqrt(4 * np.pi),
            forces=True,
        )

        self.ghat_coarse_gd.set_positions(self.scaled_atom_pos)
        self.ghat_fine_gd = None
        # else:

        #     # Only in LCAO
        #     self.ghat_aLr = pot_calc.ghat_aLr  # Same as density.ghat in old?

        #     self.ghat = density.ghat
        #     self.ghat_cg = None

        self.xc = pot_calc.xc.xc
        # setups = wfs.setups

        if poisson_solver == "FPS":
            self.poisson = PoissonSolver(
                use_charge_center=True, use_charged_periodic_corrections=True
            )

        # if self.use_coarse_grid:
        self.poisson.set_grid_descriptor(self.coarse_gd)
        # else:
        #     self.poiss.set_grid_descriptor(self.fine_gd)
        # elif poisson_solver == "GS":
        #     self.poiss = PoissonSolver(
        #         name="fd",
        #         relax=poisson_solver,
        #         eps=1.0e-16,
        #         use_charge_center=True,
        #         use_charged_periodic_corrections=True,
        #     )

        self.dtype = ibzwfs.dtype  # or self.dtype = ibzwfs._wfs_u[0].dtype

        self.e_sic_by_orbitals = {}
        self.num_kpts = len(ibzwfs.ibz)

    def get_full_orb_dependent_potential_matrix(self,wfs,ibzwfs,occupied_only=False):
        f_n = wfs.myocc_n
        n_occ = 0
        nbands = ibzwfs.nbands
        while n_occ < nbands and f_n[n_occ] > 1e-10:
            n_occ += 1


        c_nm=wfs.C_nM.data
        if occupied_only:
            nbs = n_occ
        else:
            nbs = c_nm.shape[0]
        
        # n_set is equal to 2 * nao
        n_set = c_nm.shape[1]

        b_mn = np.zeros(shape=(n_set, nbs), dtype=self.dtype)
        e_total_sic = np.array([])
        for m in range(n_occ):
            F_MM, sic_energy_n = self.get_potential_matrix_and_sic_energy(
                f_n, c_nm, wfs, ibzwfs, wfs.setups, m, n_set
            )

            b_mn[:, m] = np.dot(c_nm[m], F_MM.conj()).T
            e_total_sic = np.append(e_total_sic, sic_energy_n, axis=0)

        return F_MM, e_total_sic


    def get_potential_matrix_and_sic_energy(self, f_n, c_nm, wfs,ibzwfs,setup,m, n_set):
        #
        # kpoint = self.num_kpts * ibzwfs.s + ibzwfs.q
        F_MM = np.zeros(shape=(n_set, n_set), dtype=self.dtype)

        # Get orbital density
        # m is the occupation number i.e. for m in range(n_occ)
        # orb_dens, comp_charges, atomic_dens_matrix
        rho_MM,nt_G = self.get_density(f_n, c_nm, wfs, ibzwfs, m)
        Q_aL, D_ap= self.get_comp_charges(rho_MM, wfs,setup)

        e_sic_m, vt_mG, vHt_g = self.get_pseudo_pot(nt_G, Q_aL, m)

        pass

    def get_density(self,f_n,c_nm, wfs,ibzwfs,m):
        if f_n[m] > 1.0 + 1.0e-4:
            occup_factor = f_n[m] / (3.0 - ibzwfs.nspins)
        else:
            occup_factor = f_n[m]

        # density_matrix 
        rho_MM = occup_factor * np.outer(c_nm[m].conj(), c_nm[m])

        # Calculatr electron density from density matrix
        # orb_dens
        nt_G = self.coarse_gd.zeros()
        self.basis_functions.construct_density(rho_MM, nt_G, wfs.q)

        return rho_MM, nt_G

    def get_comp_charges(self,rho_MM,wfs,setup):

        # calculate  atomic density matrix and
        # compensation charges

        # atomic_dens_matrix, comp_charges 
        D_ap = {}
        Q_aL = {}

        # This object should be added to ibzwfs at some point maybe
        # P_aMi = wfs.P_aMi
        # P_aMi=P_qaMi[q],

        # P_qaMi = [{a: np.asarray(P_Mi) for a, P_Mi in P_aMi.items()}
        #       for P_aMi in P_qaMi]

        for a in wfs.P_aMi.keys():
            P_Mi = wfs.P_aMi[a]

            D_ii = np.zeros(
                (wfs.P_aMi[a].shape[1], wfs.P_aMi[a].shape[1]), dtype=self.dtype
            )

            rhoP_Mi = rho_MM @ P_Mi

            D_ii = P_Mi.T.conj() @ rhoP_Mi

            if self.dtype == complex:
                D_ap[a] = D_p = pack_density(D_ii.real)
            else:
                D_ap[a] = D_p = pack_density(D_ii)

            Q_aL[a] = np.dot(D_p, setup[a].Delta_pL)

        return Q_aL, D_ap
    

    def get_pseudo_pot(self):
        pass



            # D_ii = np.zeros(
            #     (wfs.P_aqMi[a].shape[2], wfs.P_aqMi[a].shape[2]), dtype=self.dtype
            # )
            # rhoP_Mi = rho_MM @ P_Mi
            # D_ii = P_Mi.T.conj() @ rhoP_Mi
            # if self.dtype == complex:
            #     D_ap[a] = D_p = pack_density(D_ii.real)
            # else:
            #     D_ap[a] = D_p = pack_density(D_ii)

            # Q_aL[a] = np.dot(D_p, setup[a].Delta_pL)

        # return nt_G, Q_aL, D_ap