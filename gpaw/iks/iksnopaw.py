from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import numpy as np
from gpaw.calculator import GPAW
from gpaw.iks.ae2density import ae2density, copy_density
from gpaw.poisson import create_poisson_solver
from gpaw.pw.poisson import ReciprocalSpacePoissonSolver
from gpaw.transformers import Transformer
from gpaw.typing import ArrayND
from gpaw.xc.numericalxc import NumericalXC, NumericalXCKernel
from scipy.optimize import minimize


class iKSNoPAW:
    def __init__(
        self,
        gs: Union[GPAW, NamedTuple],
        nt_sg: ArrayND,
        stencil: int = 3,
        scale: float = 1.0,
        isae: bool = True,
        weights: str | ArrayND = "uniform",
        use_canonical_orbitals: bool = False,
        log: Optional[Callable[[Any], None]] = None,
    ):
        """
        gs: GPAW calculator (e.g. with hgh setups and arbitrary xc)
            or a NamedTuple containing the hamiltonian, density and wfs
        nt_sg: target density on the fine or coarse grid in atomic units
        """
        self.scale = scale
        self.isae = isae
        self.use_canonical_orbitals = use_canonical_orbitals
        self.log = log
        self.ham = getattr(gs, "hamiltonian", None)
        self.densa = getattr(gs, "density", None)
        self.wfs = getattr(gs, "wfs", None)
        self.stencil = stencil
        self.gd = self.ham.gd
        self.finegd = self.ham.finegd
        self.coarse2fine = Transformer(self.gd, self.finegd).apply
        self.fine2coarse = Transformer(self.finegd, self.gd).apply
        self.init_nt_sg(nt_sg)
        self.update_target_density()
        self.init_weights(weights)
        self.N = int(np.round(np.sum(self.finegd.integrate(self.nt_sg))))
        self.init_xckernel()
        self.niter = 0
        self.eigensolver_name = getattr(self.wfs.eigensolver, "name", "")
        self.psi_jG = None
        self.psi_iG = None
        self.u_i = None
        self.p_ij = None
        self.L = 0.0
        self.prev_L = 0.0

    @property
    def na_sG(self):
        if self.isae:
            na_sG, _ = self.densa.get_all_electron_density(
                spos_ac=self.ham.spos_ac, gridrefinement=1
            )
        else:
            na_sG = self.densa.nt_sG
        return na_sG

    def init_nt_sg(self, nt_sg: ArrayND) -> None:
        if np.all(nt_sg.shape[1:] == self.gd.n_c):
            self.nt_sG = nt_sg
            self.nt_sg = self.coarse2fine(nt_sg)
        elif np.all(nt_sg.shape[1:] == self.finegd.n_c):
            self.nt_sg = nt_sg
            self.nt_sG = self.fine2coarse(self.nt_sg)
        else:
            raise ValueError(
                "the shape of input density doesn't match the grid"
                + "{} != {} or {}".format(nt_sg.shape, self.gd.n_c, self.finegd.n_c)
            )

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
        self.xckernel = NumericalXCKernel()
        self.xckernel.update_vxc(self.fermi_amaldi(self.nt_sg))
        xc = NumericalXC(self.xckernel, self.stencil)
        self.ham.xc = xc
        xc.set_positions(self.ham.spos_ac)
        xc.set_grid_descriptor(self.finegd)
        self.ham.update(self.denst)

    def fermi_amaldi(self, n_g):
        vHt_sg = self.finegd.zeros(self.ham.nspins)
        if isinstance(self.ham.poisson, ReciprocalSpacePoissonSolver):
            poisson = create_poisson_solver()
            poisson.set_grid_descriptor(self.finegd)
            _ = poisson.solve(
                vHt_sg[0],
                self.nt_sg[0],
            )
        else:
            _ = self.ham.poisson.solve(
                vHt_sg[0],
                self.nt_sg[0],
            )
        return -vHt_sg / self.N

    def update_target_density(self) -> None:
        if self.isae:
            self.denst = ae2density(
                self.nt_sG, self.densa, self.ham.spos_ac, self.wfs.kptband_comm
            )
        else:
            self.denst = copy_density(
                self.densa, self.ham.spos_ac, self.wfs.kptband_comm
            )
            # D_asp = self.densa.atomdist.to_aux(self.densa.D_asp)
            D_asp = self.densa.D_asp
            self.denst.initialize_directly_from_arrays(self.nt_sG, None, D_asp)
        self.denst.calculate_normalized_charges_and_mix()
        self.denst.fixed = True

    def update(self, v_sg: ArrayND) -> None:
        self.niter += 1
        self.xckernel.update_vxc(v_sg)
        if not self.wfs.orthonormalized:
            self.wfs.orthonormalize()
        if self.eigensolver_name == "etdm-fdpw":
            if self.use_canonical_orbitals:
                self.wfs.eigensolver.get_canonical_representation(
                    self.ham,
                    self.wfs,
                )
            else:
                if not self.wfs.eigensolver.initialized:
                    print("initialize dm helper")
                    # should it be self.densa or self.denst?
                    self.wfs.eigensolver.initialize_dm_helper(
                        self.wfs, self.ham, self.densa, self.log
                    )
                print("iterate")
                self.wfs.eigensolver.iterate(
                    self.ham, self.wfs, self.densa, self.log, denst=self.denst
                )
                print("iterate done")
                #
                # self.wfs.eigensolver.iters = 0
                # or change line 523 in etdm_fdpw.py
                # Do we need to do this?
                # _ = self.wfs.eigensolver.check_restart(self.wfs)
            e_entropy = 0.0
            kin_en_using_band = False
        elif self.eigensolver_name == "etdm-lcao":
            self.wfs.eigensolver.iterate(self.ham, self.wfs, self.densa)
            _ = self.wfs.eigensolver.check_mom(self.wfs, self.densa)
            e_entropy = 0.0
            kin_en_using_band = False
        else:
            print("iterate")
            self.wfs.eigensolver.iterate(self.ham, self.wfs)
            kin_en_using_band = True
            e_entropy = self.wfs.calculate_occupation_numbers()
            print("iterate done")
            # breakpoint()

        if hasattr(self.wfs.eigensolver, "e_sic"):
            e_sic = self.wfs.eigensolver.e_sic
        else:
            e_sic = 0.0
        self.ham.get_energy(
            e_entropy, self.wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic
        )
        self.densa.update(self.wfs)
        self.update_target_density()
        # what happens to laplacian and kinetic energy
        self.ham.update(self.denst)
        self.calculate_step()

    def update_coarse(self, v_sG: ArrayND) -> None:
        return self.update(self.coarse2fine(v_sG))

    def calculate_step(self) -> None:
        self.diff_G = np.sum(self.nt_sG - self.na_sG, axis=0)
        # TODO: replace the integrals with wfs.integrate()
        # integrate = self.wfs.pd.integrate
        # else:
        #     integrate = self.wfs.integrate
        # add the kpoint stuff
        k = 0
        occupied_i = np.nonzero(self.wfs.kpt_u[k].f_n > 1e-10)[0]
        f_i = self.wfs.kpt_u[k].f_n[occupied_i]
        e_j = self.wfs.collect_eigenvalues(k, 0)
        self.psi_jG = self.wfs.kpt_u[k].psit_nG
        if self.wfs.mode == "pw":
            self.psi_jG = np.asarray([self.wfs.pd.ifft(psi_G) for psi_G in self.psi_jG])
        self.psi_iG = self.psi_jG[occupied_i]
        tmp_G = self.weights_G * self.diff_G
        self.u_i = 2.0 * f_i * self.gd.integrate(tmp_G * self.psi_iG * self.psi_iG)
        w_ij = (e_j[:, np.newaxis] - e_j[np.newaxis, occupied_i]).T
        np.fill_diagonal(w_ij, 1.0)
        self.p_ij = (
            4.0
            * self.gd.integrate(
                self.psi_jG,
                (
                    tmp_G * self.psi_iG * f_i[..., None, None, None]
                    - 0.5 * self.u_i[..., None, None, None] * self.psi_iG
                ),
            ).T
            / w_ij
        )
        # np.fill_diagonal(self.p_ij, 0.0)

    def lagrangian(self, vxc_sG: Optional[ArrayND] = None) -> float:
        self.prev_L = self.L
        if vxc_sG is not None:
            vxc_sG = vxc_sG.reshape(self.nt_sG.shape)
            if (
                not np.allclose(
                    self.coarse2fine(vxc_sG), self.xckernel.vxc_sg, atol=1e-60
                )
                or True
            ):
                print("update_coarse Lagrangian")
                self.update_coarse(vxc_sG)
        self.L = self.gd.integrate((self.diff_G**2) * self.weights_G)
        if self.log is not None:
            self.log(
                "this should be almost zero:",
                np.sum(self.u_i * (self.gd.integrate(self.psi_iG * self.psi_iG) - 1.0)),
            )
            self.log(f"iter: {self.niter} | L = {self.L:.6e}")
        return self.scale * self.L

    def jacobian(self, vxc_sG: Optional[ArrayND] = None) -> ArrayND:
        if vxc_sG is not None:
            vxc_sG = vxc_sG.reshape(self.nt_sG.shape)
            if not np.allclose(
                self.coarse2fine(vxc_sG), self.xckernel.vxc_sg, atol=1e-60
            ):
                print("update_coarse Jacobian")
                self.update_coarse(vxc_sG)
        # p_iG = np.einsum('ij,jxyz->ixyz', self.p_ij, self.psi_jG)
        # J_G = np.sum(p_iG * self.psi_iG, axis=0)
        self.J_G = np.einsum(
            "ij,j...,i...->...", self.p_ij, self.psi_jG.conj(), self.psi_iG
        )
        if self.log is not None:
            self.log(
                f"iter: {self.niter} | mean(J):\
                {np.mean(self.J_G):.6e} | integral(J dr): \
                {self.gd.integrate(self.J_G): .6e}"
            )
        return self.scale * self.J_G

    def converged(self, vxc_sG: ArrayND, Ltol, Jtol) -> None:
        self.L_iter.append(self.L)
        imax = np.unravel_index(np.argmax(np.abs(self.J_G)), self.J_G.shape)
        self.log(
            f"iter: {self.niter} | vxc_sG.shape = {
            vxc_sG.shape}, max(J_sG) -> {imax}:{np.max(np.abs(self.J_G))}"
        )
        self.log(f"abs(L - L_prev): {np.abs(self.L - self.prev_L)}")
        return np.abs(self.L) < Ltol or np.max(np.abs(self.J_G)) < Jtol

    def run_opt(
        self,
        method: str = "L-BFGS-B",
        options: Dict = {
            "disp": True,
            "gtol": 0,
            "ftol": 0,
            "maxiter": 300,
            "maxls": 40,
        },
    ):
        self.niter = 0
        self.L_iter = []
        if self.log is not None:
            self.log(
                "initial diff:",
                self.gd.integrate((self.weights_G * (self.nt_sG - self.na_sG)[0]) ** 2),
            )

        # hack to keep the optimizer running:

        self.LL = self.L

        def lagrangian(x):
            print("lagrangian", self.L, self.prev_L, self.LL)
            L = self.lagrangian(x)
            if L != self.L:
                print("mismatch")
            if np.abs(L - self.prev_L) == 0.0:
                self.update_coarse(x.reshape(self.nt_sG.shape))
                L = self.lagrangian(x)
                if np.abs(L - self.prev_L) == 0.0:
                    print("wth")
            self.LL = L
            return L

        self.opt = minimize(
            lagrangian,
            x0=self.fine2coarse(self.xckernel.vxc_sg).flatten(),
            jac=lambda x: self.jacobian(x).flatten(),
            method=method,
            options=options,
            tol=0,
            callback=lambda x: self.converged(x, 1e-5, 1e-12),
        )
        return self.opt, self.L_iter

    def run_adam(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        Ltol=1e-5,
        Jtol=1e-12,
        epsilon=1e-10,
        maxiter=300,
    ):
        """
        Adam optimizer to minimize the Lagrangian function with
                                                    respect to vxc_sg.
        Parameters:
        - learning_rate: learning rate for the Adam optimizer.
        - beta1: exponential decay rate for the first moment estimates.
        - beta2: exponential decay rate for the second moment estimates.
        - epsilon: small constant to prevent any division by zero
                                                    in the implementation.
        - num_iterations: number of iterations for the optimization.
        - Ltol: tolerance for the Lagrangian function. (abs(L) < Ltol
                                                 for a successful termination)
        - Jtol: tolerance for the Jacobian function. (max(abs(J_G)) < Jtol
                                                 for a successful termination)
        Returns:
        - vxc_sg: the optimized value.
        """
        self.niter = 0
        vxc_sG = self.fine2coarse(self.xckernel.vxc_sg)
        # Initialize first moment vector
        m_sG = np.zeros_like(vxc_sG)
        # Initialize second moment vector
        v_sG = np.zeros_like(vxc_sG)
        L_iter = []
        for t in range(1, maxiter):
            self.update_coarse(vxc_sG)
            L = self.lagrangian()
            J_sG = self.jacobian()[np.newaxis]
            # Update biased first moment estimate
            m_sG = beta1 * m_sG + (1 - beta1) * J_sG
            # Update biased second moment estimate
            v_sG = beta2 * v_sG + (1 - beta2) * (J_sG**2)
            # Update vxc_sg
            vxc_sG = vxc_sG - learning_rate * m_sG / (1 - beta1**t) / (
                np.sqrt(v_sG / (1 - beta2**t)) + epsilon
            )
            L_iter.append(L)
            if np.abs(L) < Ltol or np.max(np.abs(J_sG)) < Jtol:
                break
        self.update_coarse(vxc_sG)
        return L_iter

    def run_lbfgs(
        self, method="LBFGS", maxiter=300, memory=10, Ltol=1e-5, Jtol=1e-10, beta=1.0
    ):
        from gpaw.directmin.sd_etdm import LBFGS, LBFGS_P

        self.niter = 0
        if method == "LBFGS":
            opt = LBFGS(memory=memory)
        elif method == "LBFGS_P":
            opt = LBFGS_P(memory=memory, beta=beta)
        vxc_sG = self.fine2coarse(self.xckernel.vxc_sg)
        L_iter = []
        for i in range(maxiter):
            self.update_coarse(vxc_sG)
            L = self.lagrangian()
            J_sG = self.jacobian()[np.newaxis]
            vxc_sG = opt.update_data(
                self.wfs, {0: vxc_sG}, {0: -J_sG}, dimensions={0: 1}
            )[0]
            imax = np.unravel_index(np.argmax(np.abs(J_sG)), J_sG.shape)
            self.log(
                f"iter: {i} | vxc_sG.shape = {
                vxc_sG.shape}, max(J_sG) -> {imax}:{np.max(np.abs(J_sG))}"
            )
            L_iter.append(L)
            if np.abs(L) < Ltol or np.max(np.abs(J_sG)) < Jtol:
                break
        self.update_coarse(vxc_sG)
        return L_iter
