from typing import Optional

import numpy as np
from ase.atoms import Atoms

# from gpaw import debug
debug = True
from ase.parallel import parprint

from gpaw.calculator import GPAW
from gpaw.iks.ae2density import ae2density
from gpaw.lfc import LFC, BasisFunctions
from gpaw.poisson import create_poisson_solver
from gpaw.transformers import Transformer
from gpaw.typing import ArrayND
from gpaw.utilities import unpack_density
from gpaw.xc.numericalxc import NumericalXC, NumericalXCKernel


class iKS(GPAW):
    def __init__(
        self,
        nt_sG: ArrayND,
        weights: str | ArrayND = "uniform",
        scale: float = 1.0,
        stencil: int = 3,
        *args,
        **kwargs,
    ):
        """
        nt_sG: target density on the fine or coarse grid in atomic units
        """
        GPAW.__init__(self, *args, **kwargs)
        self.nt_sG = nt_sG
        self.init_weights(weights)
        self.stencil = stencil
        self.scale = scale
        self.fapoisson = None

    @property
    def na_sG(self):
        na_sG, _ = self.density.get_all_electron_density(
            spos_ac=self.spos_ac, gridrefinement=1
        )
        return na_sG

    def initialize(self, atoms: Atoms, reading: bool = False):
        GPAW.initialize(self, atoms, reading)
        GPAW.set_positions(self, atoms)
        self.gd = self.hamiltonian.gd
        self.finegd = self.hamiltonian.finegd
        self.coarse2fine = Transformer(self.gd, self.finegd).apply
        self.fine2coarse = Transformer(self.finegd, self.gd).apply
        self.nt_sg = self.coarse2fine(self.nt_sG)
        self.update_target_density()
        self.N = int(np.round(np.sum(self.finegd.integrate(self.nt_sg))))
        self.niter = 0
        self.eigensolver_name = getattr(self.wfs.eigensolver, "name", "")
        # n -> bands(occ+virt), m -> occ
        self.psi_nG: Optional[ArrayND] = None
        self.psi_mG: Optional[ArrayND] = None
        self.u_m: Optional[ArrayND] = None
        self.p_mn: Optional[ArrayND] = None
        self.L = 0.0
        self.prev_L = 0.0
        self.init_paw()
        self.init_xckernel()

    def init_paw(self) -> None:
        phi_aj = []
        phit_aj = []
        nc_a = []
        nct_a = []
        splines = {}
        for a, id in enumerate(self.setups.id_a):
            if id in splines:
                phi_j, phit_j = splines[id]
            else:
                phi_j, phit_j, nc, nct = self.setups[a].get_partial_waves()[:4]
                splines[id] = (phi_j, phit_j)
            phi_aj.append(phi_j)
            phit_aj.append(phit_j)
            nc_a.append([nc])
            nct_a.append([nct])

        spos_ac = self.hamiltonian.spos_ac
        self.phi = BasisFunctions(self.gd, phi_aj)
        self.phit = BasisFunctions(self.gd, phit_aj)
        self.phi.set_positions(spos_ac)
        self.phit.set_positions(spos_ac)
        self.nc = LFC(self.gd, nc_a)
        self.nct = LFC(self.gd, nct_a)
        self.nc.set_positions(spos_ac)
        self.nct.set_positions(spos_ac)
        self.dO_MM = np.zeros(
            (self.phi.Mmax, self.phi.Mmax), dtype=self.setups[0].dO_ii.dtype
        )
        for a in range(len(self.phi.M_a)):
            M1 = self.phi.M_a[a]
            M2 = M1 + self.phi.sphere_a[a].Mmax
            self.dO_MM[M1:M2, M1:M2] = self.setups[a].dO_ii

    def init_weights(self, weights: str | ArrayND) -> None:
        if isinstance(weights, ArrayND):
            self.weights_G = weights
        elif weights == "uniform":
            self.weights_G = np.ones_like(self.nt_sG[0])
        elif weights == "density":
            self.weights_G = np.sum(self.nt_sG, axis=0)
        elif weights == "inverse_density":
            self.weights_G = 1.0 / (np.sum(self.nt_sG, axis=0) + 1e-10)
        elif weights == "inverse_density_squared":
            self.weights_G = 1.0 / (np.sum(self.nt_sG, axis=0) ** 2 + 1e-20)
        elif weights == "mask":
            self.weights_G = np.zeros_like(self.nt_sG[0])
            self.weights_G[self.nt_sG[0] > 1e-5] = 1.0
        else:
            raise ValueError(
                'weights must be one of "uniform", "density", \
                "inverse_density", "inverse_density_squared" and "mask"'
            )

    def init_xckernel(self) -> None:
        D_asp = self.density.D_asp
        self.xckernel = NumericalXCKernel()
        self.xckernel.update_vxc(self.fermi_amaldi(self.finegd, self.nt_sg))
        xc = NumericalXC(self.xckernel, self.stencil)
        self.hamiltonian.xc = xc
        xc.set_positions(self.hamiltonian.spos_ac)
        xc.set_grid_descriptor(self.finegd)
        vxc_asG, vxct_asG = self.fermi_amaldi_paw(D_asp)
        self.xckernel.update_vxc_a(vxc_asG)
        self.xckernel.update_vxct_a(vxct_asG)
        xc.set_lfc(self.phi, self.phit, self.nc, self.nct, self.hamiltonian.nspins)

    def fermi_amaldi(self, gd, nt_sg):
        vHt_sg = gd.zeros(self.hamiltonian.nspins)
        N = gd.integrate(nt_sg)
        if self.fapoisson is None:
            self.fapoisson = create_poisson_solver()
        self.fapoisson.set_grid_descriptor(gd)
        for s in range(self.hamiltonian.nspins):
            _ = self.fapoisson.solve(
                vHt_sg[s],
                nt_sg[s],
            )
        return -vHt_sg / N

    def fermi_amaldi_paw(self, D_asp):
        nspins = self.hamiltonian.nspins
        natoms = len(self.setups)
        a_asi = {
            a: np.zeros((nspins, self.nc.get_function_count(a))) for a in range(natoms)
        }
        vxc_asG = self.gd.zeros((natoms, self.hamiltonian.nspins))
        vxct_asG = self.gd.zeros((natoms, self.hamiltonian.nspins))
        for a, D_sp in D_asp.items():
            na_sG = self.gd.zeros(nspins)
            nat_sG = self.gd.zeros(nspins)
            M1 = self.phi.M_a[a]
            M2 = M1 + self.phi.sphere_a[a].Mmax
            for s, D_p in enumerate(D_sp):
                rho_MM = np.zeros((self.phi.Mmax, self.phi.Mmax))
                rho_MM[M1:M2, M1:M2] = unpack_density(D_p)
                self.phi.construct_density(rho_MM, na_sG[s], -1)
                self.phit.construct_density(rho_MM, nat_sG[s], -1)
            # add core density
            c_asi = a_asi.copy()
            c_asi[a] = np.ones_like(c_asi[a])
            self.nc.add(na_sG, c_asi)
            self.nct.add(nat_sG, c_asi)
            vxc_asG[a] = self.fermi_amaldi(self.gd, na_sG)
            vxct_asG[a] = self.fermi_amaldi(self.gd, nat_sG)
        return vxc_asG, vxct_asG

    def update_target_density(self) -> None:
        self.denst = ae2density(
            self.nt_sG, self.density, self.hamiltonian.spos_ac, self.wfs.kptband_comm
        )
        self.denst.calculate_normalized_charges_and_mix()
        self.denst.fixed = True

    def update(self, v_sG: ArrayND, v_asG: ArrayND, vt_asG: ArrayND) -> None:
        self.xckernel.update_vxc(self.coarse2fine(v_sG))
        self.xckernel.update_vxc_a(v_asG)
        self.xckernel.update_vxct_a(vt_asG)
        # scf step
        if not self.wfs.orthonormalized:
            self.wfs.orthonormalize()
        if self.eigensolver_name == "etdm-fdpw":
            raise NotImplementedError
        else:
            self.wfs.eigensolver.iterate(self.hamiltonian, self.wfs)
            kin_en_using_band = True
            e_entropy = self.wfs.calculate_occupation_numbers()

        if hasattr(self.wfs.eigensolver, "e_sic"):
            e_sic = self.wfs.eigensolver.e_sic
        else:
            e_sic = 0.0
        self.hamiltonian.get_energy(
            e_entropy, self.wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic
        )
        self.density.update(self.wfs)
        self.update_target_density()
        self.hamiltonian.update(self.denst)
        # update coefficients
        self.calculate_step()

    def calculate_step(self) -> None:
        self.diff_G = np.sum(self.nt_sG - self.na_sG, axis=0)
        # TODO: replace the integrals with wfs.integrate()
        # integrate = self.wfs.pd.integrate
        # add the kpoint and spin stuff
        k = 0
        s = 0
        occupied_m = self.occupied_m = np.nonzero(self.wfs.kpt_u[k].f_n > 1e-10)[0]
        f_m = self.f_m = self.wfs.kpt_u[k].f_n[occupied_m]
        e_n = self.wfs.collect_eigenvalues(k, s)
        self.psi_nG = self.wfs.kpt_u[k].psit_nG
        if self.wfs.mode == "pw":
            self.psi_nG = np.asarray([self.wfs.pd.ifft(psi_G) for psi_G in self.psi_nG])
        self.psi_mG = self.psi_nG[occupied_m]
        self.wdn_G = self.weights_G * self.diff_G
        self.u_m = 2.0 * f_m * self.gd.integrate(self.wdn_G * self.psi_mG * self.psi_mG)
        self.add_paw_correction_u(k, s)
        w_mn = self.w_mn = (e_n[:, np.newaxis] - e_n[np.newaxis, occupied_m]).T
        np.fill_diagonal(w_mn, 1.0)
        self.p_mn = (
            4.0
            * self.gd.integrate(
                self.psi_nG.conj(),
                (
                    self.wdn_G * self.psi_mG * f_m[..., None, None, None]
                    - 0.5 * self.u_m[..., None, None, None] * self.psi_mG
                ),
            ).T
            / w_mn
        )
        self.add_paw_correction_p()
        self.p_mn = self.p_mn.conj()

    def add_paw_correction_u(self, k: int = 0, s: int = 0) -> None:
        """add paw correction to the lagrange multiplier u_m
        and store the density matrix coefficients P_nM"""
        self.P_nM = np.zeros(
            (len(self.psi_nG), self.phi.Mmax),
            dtype=self.wfs.kpt_u[k].P_ani[0].dtype,
        )
        self.phi.ai_to_M(self.wfs.kpt_u[k].P_ani, self.P_nM)
        self.P_mM = self.P_nM[self.occupied_m]
        for m, P_M in enumerate(self.P_mM):
            rho_MM = np.outer(P_M.conj(), P_M)
            PDelta_G = self.gd.zeros()
            self.phi.construct_density(rho_MM, PDelta_G, q=-1)
            self.phit.construct_density(-rho_MM, PDelta_G, q=-1)
            self.u_m[m] += 2.0 * self.f_m[m] * self.gd.integrate(self.wdn_G * PDelta_G)

    def add_paw_correction_p(self, k: int = 0, s: int = 0) -> None:
        """add paw correction to the expansion coefficients p_mn"""
        for n in range(len(self.P_nM)):
            for m in range(len(self.P_mM)):
                rho_MM = np.outer(self.P_nM[n].conj(), self.P_mM[m])
                PmnDelta_G = self.gd.zeros()
                self.phi.construct_density(rho_MM, PmnDelta_G, q=-1)
                self.phit.construct_density(-rho_MM, PmnDelta_G, q=-1)
                self.p_mn[m, n] += (
                    self.gd.integrate(
                        4.0 * self.f_m[m] * self.wdn_G * PmnDelta_G
                        - 2.0 * self.u_m[m] * np.sum(self.dO_MM * rho_MM)
                    )
                    / self.w_mn[m, n]
                )

    def lagrangian(self) -> float:
        self.prev_L = self.L
        self.L = self.gd.integrate(self.weights_G * self.diff_G**2)
        self.debuglog(f"iter: {self.niter} | L = {self.L:.6e}")
        return self.scale * self.L

    def jacobian(self) -> ArrayND:
        # TODO: include spin polarization
        self.J_sG = np.einsum(
            "mn,n...,m...->...", self.p_mn, self.psi_nG.conj(), self.psi_mG
        )[np.newaxis]
        return self.scale * self.J_sG

    def jacobian_paw(self, k: int = 0) -> tuple[ArrayND, ArrayND]:
        natoms, nspins = self.xckernel.vxc_asG.shape[:2]
        self.J_asG = J_asG = self.gd.zeros((natoms, nspins))
        self.Jt_asG = Jt_asG = self.gd.zeros((natoms, nspins))
        Mmax = self.phi.Mmax
        rho_MM = np.einsum(
            "nM,mN,mn->MN",
            self.P_nM.conj(),
            self.P_mM,
            self.p_mn,
        )
        for a in range(natoms):
            M1 = self.phi.M_a[a]
            M2 = M1 + self.phi.sphere_a[a].Mmax
            mask = np.ones((Mmax, Mmax), dtype=bool)
            mask[M1:M2, M1:M2] = False
            buf_MM = rho_MM.copy()
            buf_MM[mask] = 0.0
            for s in range(nspins):
                self.phi.construct_density(buf_MM, J_asG[a, s], q=-1)
                self.phit.construct_density(-buf_MM, Jt_asG[a, s], q=-1)
        return J_asG, Jt_asG

    def converged(self, Ltol, Jtol) -> None:
        return np.abs(self.L) < Ltol or np.max(np.abs(self.J_sG)) < Jtol

    def debuglog(self, msg: str) -> None:
        if debug:
            parprint(msg)

    def adam_step(self, lr, vxc, J, m, mm):
        epsilon = 1e-10
        beta1 = self.adam_params["beta1"]
        beta2 = self.adam_params["beta2"]
        t = self.niter
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * J
        # Update biased second moment estimate
        mm = beta2 * mm + (1 - beta2) * J**2
        # Update vxc
        vxc -= lr * m / (1 - beta1**t) / (np.sqrt(mm / (1 - beta2**t)) + epsilon)
        return m, mm

    def run_adam(
        self,
        maxiter=300,
        lr_pseudo=0.001,
        lr_paw=0.001,
        beta1=0.9,
        beta2=0.999,
        Ltol=1e-5,
        Jtol=1e-6,
    ):
        """
        Adam optimizer to minimize the Lagrangian function with
                                                    respect to vxc_sg.
        Parameters:
        - maxiter: number of iterations for the optimization.
        - lr: learning rate for the Adam optimizer.
        - beta1: exponential decay rate for the first moment estimates.
        - beta2: exponential decay rate for the second moment estimates.
        - Ltol: tolerance for the Lagrangian function. (abs(L) < Ltol
                                                 for a successful termination)
        - Jtol: tolerance for the Jacobian function. (max(abs(J_G)) < Jtol
                                                 for a successful termination)
        Returns:
        - L_iter: list of Lagrangian values at each iteration.
        """
        self.adam_params = {
            "lr_pseudo": lr_pseudo,
            "lr_paw": lr_paw,
            "beta1": beta1,
            "beta2": beta2,
            "maxiter": maxiter,
            "Ltol": Ltol,
            "Jtol": Jtol,
        }
        vxc_sG = self.fine2coarse(self.xckernel.vxc_sg)
        vxc_asG = self.xckernel.vxc_asG
        vxct_asG = self.xckernel.vxct_asG
        # Initialize first moment vector
        m_sG = np.zeros_like(vxc_sG)
        m_asG = np.zeros_like(vxc_asG)
        mt_asG = np.zeros_like(vxct_asG)
        # Initialize second moment vector
        mm_sG = np.zeros_like(vxc_sG)
        mm_asG = np.zeros_like(vxc_asG)
        mmt_asG = np.zeros_like(vxct_asG)
        #
        L_iter = []
        self.debuglog("-- Adam optimizer --")
        self.debuglog(
            "\t{:>5} | {:>10} | {:>10} | {:>10} | {:>10}".format(
                "iter", "L", "max(J)", "max(J_a)", "max(Jt_a)" 
            )
        )

        for t in range(1, maxiter):
            self.niter = t
            # step
            self.update(vxc_sG, vxc_asG, vxct_asG)
            L = self.lagrangian()
            L_iter.append(L)
            # pseudo
            J_sG = self.jacobian()
            m_sG, mm_sG = self.adam_step(lr_pseudo, vxc_sG, J_sG, m_sG, mm_sG)
            # paw
            J_asG, Jt_asG = self.jacobian_paw()
            m_asG, mm_asG = self.adam_step(lr_paw, vxc_asG, J_asG, m_asG, mm_asG)
            mt_asG, mmt_asG = self.adam_step(lr_paw, vxct_asG, Jt_asG, mt_asG, mmt_asG)
            # log
            self.debuglog(
                "\t{:>5} | {:>10.6e} | {:>10.6e} | {:>10.6e} | {:>10.6e}".format(
                    t, L, np.max(np.abs(J_sG)), np.max(np.abs(J_asG)), np.max(np.abs(Jt_asG)),
                )
            )
            # check convergence
            if self.converged(Ltol, Jtol):
                break
        self.update(vxc_sG, vxc_asG, vxct_asG)
        return L_iter
