from math import pi, sqrt

import numpy as np
from ase.atoms import Atoms
from ase.parallel import parprint
from gpaw.arraydict import ArrayDict
from gpaw.calculator import GPAW
from gpaw.gaunt import gaunt
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.iks.ae2density import ae2density
from gpaw.poisson import BasePoissonSolver, create_poisson_solver
from gpaw.pw.hamiltonian import ReciprocalSpaceHamiltonian
from gpaw.sphere.lebedev import Y_nL
from gpaw.transformers import Transformer
from gpaw.typing import ArrayND
from gpaw.utilities import pack_hermitian, unpack_hermitian
from gpaw.utilities.gauss import Gaussian
from gpaw.xc.numericalxc import NumericalXC, NumericalXCKernel

# from typing import Optional


# from gpaw import debug
debug = True


def atomic_fermi_amaldi(setup, D_sp):
    xcc = setup.xc_correction
    rgd = xcc.rgd
    nspins = len(D_sp)

    nc0_sg = rgd.empty(nspins)
    nct0_sg = rgd.empty(nspins)
    nc0_sg[:] = sqrt(4 * pi) / nspins * xcc.nc_g
    nct0_sg[:] = sqrt(4 * pi) / nspins * xcc.nct_g
    if xcc.nc_corehole_g is not None and nspins == 2:
        nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
        nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
    n_qg = xcc.n_qg
    nt_qg = xcc.nt_qg
    D_sLq = np.inner(D_sp, xcc.B_pqL.T)
    n_sLg = np.dot(D_sLq, n_qg)
    nt_sLg = np.dot(D_sLq, nt_qg)
    Lmax = n_sLg.shape[1]
    vxc_sng = rgd.zeros((nspins, len(Y_nL)))
    vxct_sng = rgd.zeros((nspins, len(Y_nL)))
    for n, Y_L in enumerate(Y_nL[:, :Lmax]):
        n_sg = np.dot(Y_L, n_sLg)
        N = rgd.integrate(n_sg)
        nt_sg = np.dot(Y_L, nt_sLg)
        Nt = rgd.integrate(nt_sg)
        for s in range(nspins):
            vxc_sng[s, n, 1:] = -rgd.poisson(n_sg[s])[1:] / rgd.r_g[1:] / N
            vxc_sng[s, n, 0] = vxc_sng[s, n, 1]
            vxct_sng[s, n, 1:] = -rgd.poisson(nt_sg[s])[1:] / rgd.r_g[1:] / Nt
            vxct_sng[s, n, 0] = vxct_sng[s, n, 1]
    return vxc_sng, vxct_sng


def calculate_electrostatic_energy(ham, dens):
    if isinstance(ham, ReciprocalSpaceHamiltonian):
        mode = "pw"
    elif isinstance(ham, RealSpaceHamiltonian):
        mode = "fd"
    else:
        raise ValueError(
            "Hamiltonian must be either ReciprocalSpaceHamiltonian or RealSpaceHamiltonian"
        )
    if mode == "pw":
        vHt_x = np.zeros_like(ham.vHt_q)
        e_coulomb = ham.poisson.solve(vHt_x, dens)
        e_coulomb /= ham.finegd.comm.size
    elif mode == "fd":
        vHt_x = np.zeros_like(dens.rhot_g)
        ham.poisson.solve(vHt_x, dens.rhot_g, charge=-dens.charge)
        e_coulomb = 0.5 * ham.finegd.integrate(
            vHt_x, dens.rhot_g
        )  # global_integral=False
    D_asp = dens.D_asp
    for a, D_sp in D_asp.items():
        if ham.nspins == 2:
            D_p = D_sp.sum(0)
        else:
            D_p = D_sp[0]
        setup = ham.setups[a]
        e_coulomb += setup.M + np.dot(D_p, (setup.M_p + np.dot(setup.M_pp, D_p)))
    return e_coulomb, vHt_x


def calculate_atomic_hamiltonian(vHt_x, ham, dens):
    def getshape(a):
        return (sum(2 * l + 1 for l, _ in enumerate(ham.setups[a].ghat_l)),)

    W_aL = ArrayDict(ham.atomdist.aux_partition, getshape, float)
    dens.ghat.integrate(vHt_x, W_aL)
    return ham.atomdist.to_work(ham.atomdist.from_aux(W_aL))


def fermi_amaldi(
    fapoisson: BasePoissonSolver, gd: GridDescriptor, n_sg: ArrayND
) -> ArrayND:
    # use hamiltonian.get_electrostatic_potential instead?
    nspins = len(n_sg)
    vHt_sg = gd.zeros(nspins)
    N = gd.integrate(n_sg)
    for s in range(nspins):
        _ = fapoisson.solve(
            vHt_sg[s],
            n_sg[s],
        )
    return -vHt_sg / N


class iKSC(GPAW):
    def __init__(
        self,
        nt_sg: ArrayND,
        stencil: int = 3,
        a: float = 100.0,
        *args,
        **kwargs,
    ):
        """
        Perform the PDE-CO inverse Kohn-Sham calculation considering the Coulomb energy difference.
        nt_sG: target density on the fine or coarse grid in atomic units
        stencil: stencil used for evaluation of the energy based on the virial theorem
        a: Gaussian exponent
        """
        GPAW.__init__(self, *args, **kwargs)
        # TODO: don't require exact same spacing for nt_sg, instead interpolate on the current grid
        self.nt_sg = nt_sg
        self.stencil = stencil
        self.a = a
        self.adam_params: dict[str, float | int] = {}
        self.L_iter: list[float] = []

    @property
    def na_sg(self):
        na_sg, _ = self.density.get_all_electron_density(
            spos_ac=self.spos_ac, gridrefinement=2
        )
        return na_sg

    def initialize(self, atoms: Atoms, reading: bool = False):
        GPAW.initialize(self, atoms, reading)
        GPAW.set_positions(self, atoms)
        self.gd = self.hamiltonian.gd
        self.finegd = self.hamiltonian.finegd
        self.coarse2fine = Transformer(self.gd, self.finegd).apply
        self.fine2coarse = Transformer(self.finegd, self.gd).apply
        self.init_target_density()
        self.N = int(np.round(np.sum(self.finegd.integrate(self.nt_sg))))
        self.niter: int = 0
        self.eigensolver_name: str = getattr(self.wfs.eigensolver, "name", "")
        self.L: float = 0.0
        self.prev_L: float = 0.0
        self.init_xckernel()

    def init_xckernel(self) -> None:
        vxc_sg = fermi_amaldi(self.poisson, self.finegd, self.nt_sg)
        # TODO: define these with atomdist workarrays
        vxc_asng = {}
        vxct_asng = {}
        for a, D_sp in self.density.D_asp.items():
            vxc_asng[a], vxct_asng[a] = atomic_fermi_amaldi(self.setups[a], D_sp)
        self.xckernel = NumericalXCKernel(
            vxc_sg=vxc_sg, vxc_asng=vxc_asng, vxct_asng=vxct_asng
        )
        xc = NumericalXC(self.xckernel, self.stencil)
        xc.set_positions(self.hamiltonian.spos_ac)
        xc.set_grid_descriptor(self.finegd)
        self.hamiltonian.xc = xc
        self.B_apqL = {}
        for a, setup in enumerate(self.setups):
            xcc = setup.xc_correction
            #     jlL = xcc.jlL
            #     G_LLL = gaunt(max(l for j, l, _ in jlL))[:, :, : xcc.Lmax]
            #     LGcut = G_LLL.shape[2]
            #     q = 0
            #     for j1, l1, L1 in jlL:
            #         for j2, l2, L2 in jlL[j1:]:
            #             q += 1
            #
            self.B_apqL[a] = np.zeros(
                (xcc.ni * xcc.ni, xcc.B_pqL.shape[1], xcc.B_pqL.shape[2])
            )
            for q in range(xcc.B_pqL.shape[1]):
                for L in range(xcc.B_pqL.shape[2]):
                    self.B_apqL[a][:, q, L] = unpack_hermitian(
                        np.array(xcc.B_pqL[:, q, L])
                    ).flatten()

    def init_target_density(self) -> None:
        self.nt_sG = self.fine2coarse(self.nt_sg)
        self.poisson = create_poisson_solver()
        self.poisson.set_grid_descriptor(self.finegd)
        Z_g = self.finegd.zeros()
        for a, setup in enumerate(self.setups):
            gauss = Gaussian(
                self.finegd,
                a=self.a,
                center=self.spos_ac[a] * self.finegd.cell_cv.sum(0),
            )
            Z_g -= setup.Z * gauss.get_gauss(0) / np.sqrt(4 * np.pi)
        self.rhot_g = self.nt_sg.sum(0) + Z_g
        self.vHtin_R = self.finegd.zeros()
        self.poisson.solve(self.vHtin_R, self.rhot_g)
        self.Ec_in = 0.5 * self.finegd.integrate(self.vHtin_R, self.rhot_g)

    def update(
        self, v_sG: ArrayND, v_asng: dict[ArrayND], vt_asng: dict[ArrayND]
    ) -> None:
        self.xckernel.vxc_sg[:] = self.coarse2fine(v_sG)
        for a in v_asng:
            self.xckernel.vxc_asng[a][:] = v_asng[a]
            self.xckernel.vxct_asng[a][:] = vt_asng[a]
        # scf step
        if not self.wfs.orthonormalized:
            self.wfs.orthonormalize()
        if self.eigensolver_name == "etdm-fdpw":
            # TODO: implement etdm-fdpw
            raise NotImplementedError
        else:
            self.wfs.eigensolver.iterate(self.hamiltonian, self.wfs)
            e_entropy = self.wfs.calculate_occupation_numbers()

        self.hamiltonian.get_energy(
            e_entropy, self.wfs, kin_en_using_band=False, e_sic=0.0
        )
        self.density.update(self.wfs)
        self.hamiltonian.update(self.density)
        # update electrostatic energy
        self.update_coulomb()
        # update coefficients
        self.calculate_step()

    def update_coulomb(self) -> None:
        self.Ec, self.vHt_x = calculate_electrostatic_energy(
            self.hamiltonian, self.density
        )
        if self.wfs.mode == "pw":
            self.vHt_R = self.wfs.pd3.ifft(self.vHt_x)  # distribute=False
        elif self.wfs.mode == "fd":
            self.vHt_R = self.fine2coarse(self.vHt_x)

    def calculate_step(self, k=0, s=0) -> None:
        self.delta_Ec = self.Ec - self.Ec_in
        # TODO: replace the integrals with wfs.integrate() (work with reciprocal space for pw)
        # add the kpoint and spin stuff
        occupied_m = self.occupied_m = np.nonzero(self.wfs.kpt_u[k].f_n > 1e-10)[0]
        f_m = self.f_m = self.wfs.kpt_u[k].f_n[occupied_m]
        self.psi_nG = self.wfs.kpt_u[k].psit_nG
        if self.wfs.mode == "pw":
            self.psi_nG = np.asarray(
                [self.wfs.pd.ifft(psi_G) for psi_G in self.psi_nG]
            )  # local=True, safe=False
        self.psi_mG = self.psi_nG[occupied_m]
        self.u_m = (
            2.0
            * f_m
            * self.delta_Ec
            * self.gd.integrate(self.psi_mG.conj() * self.vHt_R * self.psi_mG)
        )
        self.add_paw_correction_u(k, s)
        e_n = self.wfs.collect_eigenvalues(k, s)
        w_mn = self.w_mn = (e_n[:, np.newaxis] - e_n[np.newaxis, occupied_m]).T
        np.fill_diagonal(w_mn, 1.0)
        self.p_mn = (
            4.0
            * self.gd.integrate(
                self.psi_nG,
                (self.delta_Ec * f_m - 0.5 * self.u_m)[
                    :, np.newaxis, np.newaxis, np.newaxis
                ]
                * self.psi_mG
                * self.vHt_R,
            ).T
            / w_mn
        )
        self.add_paw_correction_p()
        self.p_mn = self.p_mn.conj()

    def add_paw_correction_u(self, k: int = 0, s: int = 0) -> None:
        """add paw correction to the lagrange multiplier u_m"""

        P_ani = self.wfs.kpt_u[k].P_ani
        self.W_aL = W_aL = calculate_atomic_hamiltonian(
            self.vHt_x, self.hamiltonian, self.density
        )
        self.du_aii = {}
        for a, P_ni in P_ani.items():
            setup = self.setups[a]
            D_sp = self.density.D_asp[a]
            if self.hamiltonian.nspins == 2:
                D_p = D_sp.sum(0)
            else:
                D_p = D_sp[0]
            du_p = (
                setup.M_p
                + 2.0 * np.dot(setup.M_pp, D_p)
                + np.dot(setup.Delta_pL, W_aL[a])
            )
            self.du_aii[a] = du_ii = unpack_hermitian(du_p)
            self.u_m += (
                2.0
                * self.f_m
                * np.einsum(
                    "mi, ij, mj->m",
                    P_ni[self.occupied_m].conj(),
                    du_ii,
                    P_ni[self.occupied_m],
                )
            )

    def add_paw_correction_p(self, k: int = 0, s: int = 0) -> None:
        """add paw correction to the expansion coefficients p_mn"""
        P_ani = self.wfs.kpt_u[k].P_ani
        for a, P_ni in P_ani.items():
            setup = self.setups[a]
            dh_mii = (
                4 * self.f_m[:, None, None] * self.delta_Ec * self.du_aii[a][None]
                - 2 * self.u_m[:, None, None] * setup.dO_ii[None]
            )
            self.p_mn += (
                np.einsum("ni, mij, mj->mn", P_ni.conj(), dh_mii, P_ni[self.occupied_m])
                / self.w_mn
            )

    def lagrangian(self) -> float:
        self.prev_L = self.L
        self.L = (self.Ec_in - self.Ec) ** 2
        return self.L

    def jacobian(self) -> ArrayND:
        # TODO: include spin polarization and kpoints
        self.J_sG = np.einsum(
            "mn,n...,m...->...", self.p_mn, self.psi_nG.conj(), self.psi_mG
        )[np.newaxis]
        return self.J_sG

    def jacobian_paw(self, k: int = 0) -> tuple[ArrayND, ArrayND]:
        self.J_asng = {}  # n -> lebedev points
        self.Jt_asng = {}
        for a, P_ni in self.wfs.kpt_u[k].P_ani.items():
            self.J_asng[a] = np.zeros_like(self.xckernel.vxc_asng[a])
            self.Jt_asng[a] = np.zeros_like(self.xckernel.vxct_asng[a])
            setup = self.setups[a]
            xcc = setup.xc_correction
            if xcc is None:
                raise ValueError("No XC correction for atom %d" % a)
            dp_ii = np.einsum(
                "ni, mn, mj->ij", P_ni.conj(), self.p_mn, P_ni[self.occupied_m]
            )
            dp_Lq = np.inner(dp_ii.flatten(), self.B_apqL[a].T)
            dv_Lg = np.dot(dp_Lq, xcc.n_qg)
            dvt_Lg = -np.dot(dp_Lq, xcc.nt_qg)
            for n, Y_L in enumerate(Y_nL):
                self.J_asng[a][:, n] += np.dot(Y_L, dv_Lg)
                self.Jt_asng[a][:, n] += np.dot(Y_L, dvt_Lg)
        return self.J_asng, self.Jt_asng

    def converged(self, Ltol, Jtol) -> None:
        return np.abs(self.L) < Ltol or np.max(np.abs(self.J_sG)) < Jtol

    def debuglog(self, msg: str) -> None:
        if debug:
            parprint(msg)

    def adam_step(self, lr, vxc, J, m, mm):
        epsilon = 1e-8
        beta1 = self.adam_params["beta1"]
        beta2 = self.adam_params["beta2"]
        t = self.niter
        # Update biased first moment estimate
        m *= beta1
        m += (1 - beta1) * J
        # Update biased second moment estimate
        mm *= beta2
        mm += (1 - beta2) * J**2
        # Update vxc
        vxc -= lr * m / (1.0 - beta1**t) / (np.sqrt(mm / (1.0 - beta2**t)) + epsilon)

    def run_adam(
        self,
        maxiter: int = 300,
        lr_pseudo: float = 0.001,
        lr_paw: float = 0.001,
        Ltol: float = 1e-5,
        Jtol: float = 1e-6,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> list[float]:
        """
        Adam optimizer to minimize the Lagrangian function with
                        respect to smooth and atomic xc-potentials.
        Parameters:
        - maxiter: number of iterations for the optimization.
        - lr_pseudo: learning rate used for updating the smooth xc-potential.
        - lr_paw: learning rate used for updating the atomic potentials.
        - Ltol: tolerance for the Lagrangian.
                [abs(L) < Ltol for a successful termination]
        - Jtol: tolerance for the Jacobian of the smooth potential.
                [max(abs(J)) < Jtol for a successful termination]
        - beta1: exponential decay rate for the first moment estimates.
        - beta2: exponential decay rate for the second moment estimates.
        Returns:
        - L_iter: list of Lagrangian values at each iteration.
        """
        if not self.initialized:
            raise ValueError("The calculator is not initialized.")
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
        vxc_asng = self.xckernel.vxc_asng
        vxct_asng = self.xckernel.vxct_asng
        # initialize first moment vector
        m_sG = np.zeros_like(vxc_sG)
        m_asng = {a: np.zeros_like(vxc) for a, vxc in vxc_asng.items()}
        mt_asng = {a: np.zeros_like(vxct) for a, vxct in vxct_asng.items()}
        # initialize second moment vector
        mm_sG = np.zeros_like(vxc_sG)
        mm_asng = {a: np.zeros_like(vxc) for a, vxc in vxc_asng.items()}
        mmt_asng = {a: np.zeros_like(vxct) for a, vxct in vxct_asng.items()}
        # optimization loop
        L_iter = self.L_iter = []
        self.debuglog("-- Adam optimizer --")
        self.debuglog(
            "\t{:>5} | {:>10} | {:>10} | {:>10} | {:>10}".format(
                "iter", "L", "max(J)", "max(J_a)", "max(Jt_a)"
            )
        )

        for t in range(1, maxiter):
            self.niter = t
            # step
            self.update(vxc_sG, vxc_asng, vxct_asng)
            L = self.lagrangian()
            L_iter.append(L)
            # jacobian
            J_sG = self.jacobian()
            J_asng, Jt_asng = self.jacobian_paw()
            # pseudo
            self.adam_step(lr_pseudo, vxc_sG, J_sG, m_sG, mm_sG)
            # paw
            for a in vxc_asng:
                self.adam_step(lr_paw, vxc_asng[a], J_asng[a], m_asng[a], mm_asng[a])
                self.adam_step(
                    lr_paw, vxct_asng[a], Jt_asng[a], mt_asng[a], mmt_asng[a]
                )
            # log
            self.debuglog(
                "\t{:>5} | {:>10.6e} | {:>10.6e} | {:>10.6e} | {:>10.6e}".format(
                    t,
                    L,
                    np.abs(J_sG).max(),
                    np.max([np.abs(J_sng).max() for J_sng in J_asng.values()]),
                    np.max([np.abs(Jt_sng).max() for Jt_sng in Jt_asng.values()]),
                )
            )
            # check convergence
            if self.converged(Ltol, Jtol):
                break
        self.update(vxc_sG, vxc_asng, vxct_asng)
        return L_iter
