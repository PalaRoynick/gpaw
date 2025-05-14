from math import pi, sqrt
from typing import Optional

import numpy as np
from ase.parallel import parprint
from gpaw.calculator import GPAW
from gpaw.gaunt import super_gaunt
from gpaw.grid_descriptor import GridDescriptor
from gpaw.iks.ae2density import copy_density
from gpaw.poisson import BasePoissonSolver, create_poisson_solver
from gpaw.sphere.lebedev import Y_nL
from gpaw.transformers import Transformer
from gpaw.typing import ArrayND
from gpaw.utilities import divrl, unpack_hermitian, pack_hermitian
from gpaw.xc.numericalxc import NumericalXC, NumericalXCKernel

# from typing import Optional


# from gpaw import debug
debug = True


def L_to_lm(L):
    """convert L index to (l, m) index"""
    l = int(sqrt(L))
    m = L - l**2 - l
    return l, m


def atomic_fa_angular(setup, D_sp):
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
    vxc_sLg = rgd.zeros((nspins, Lmax))
    vxct_sLg = rgd.zeros((nspins, Lmax))
    for L in range(Lmax):
        l, m = L_to_lm(L)
        for s in range(nspins):
            v_g = rgd.poisson(n_sLg[s, L], l=l)
            v_g[1:] /= rgd.r_g[1:]
            v_g[0] = v_g[1]
            v_g[-1] = 0.0
            vxc_sLg[s, L] = -v_g / rgd.integrate(n_sLg[s, L])
            v_g = rgd.poisson(nt_sLg[s, L], l=l)
            v_g[1:] /= rgd.r_g[1:]
            v_g[0] = v_g[1]
            v_g[-1] = 0.0
            vxct_sLg[s, L] = -v_g / rgd.integrate(nt_sLg[s, L])
    return vxc_sLg, vxct_sLg


def atomic_fa_lebedev(setup, D_sp):
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
            vxc_sng[s, n] = -divrl(rgd.poisson(n_sg[s]), 1, rgd.r_g) / N
            vxct_sng[s, n] = -divrl(rgd.poisson(nt_sg[s]), 1, rgd.r_g) / Nt
    return vxc_sng, vxct_sng


def atomic_fermi_amaldi(setup, D_sp, use_lebedev=True):
    if use_lebedev:
        return atomic_fa_lebedev(setup, D_sp)
    else:
        return atomic_fa_angular(setup, D_sp)


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


class iKS:
    def __init__(
        self,
        tcalc: GPAW,
        icalc: Optional[GPAW] = None,
        P_snn: Optional[ArrayND] = None,
        n_scf: int = 1,
        weights: str | ArrayND = "uniform",
        vxc0_sg: Optional[ArrayND] = None,
        stencil: int = 2,
        use_lebedev: bool = True,
        calculate_lagrange_mu: bool = False,
    ):
        """
        Perform the PDE-CO iKS.
        tcalc: reference calculation (e.g. a Hartree-Fock calculation for CI or for a local exact exchange)
        icalc: calculation for the initial guess (e.g. a Hartree calculation with xc='null'). defaults to gs.
        P_nn: density matrix from a higher level calculation, if not provided the density matrix is taken from the gs.
        weights: field used to control the weights of density difference in the equations.
        vxc0_sg: if specified the initial guess for pseudo potential is going to be set to this value instead of FA.
        lreg: regularization parameter for the smooth potential
        lregpaw: regularization parameter for the atomic potentials.
        stencil: stencil used for evaluation of the energy based on the virial theorem and the Jacobian
        use_lebedev: use Lebedev quadrature for the atomic potentials
        calculate_lagrange_mu: calculate the Lagrange multipliers for enforcing the orthogonality of the wavefunctions
        """
        self.tcalc = tcalc
        if icalc is None:
            self.icalc = tcalc
        else:
            self.icalc = icalc
        self.stencil = stencil
        self.n_scf = n_scf
        self.adam_params: dict[str, float | int] = {}
        self.wfs = self.icalc.wfs
        self.hamiltonian = self.icalc.hamiltonian
        self.density = self.icalc.density
        self.log = self.icalc.log
        self.init_target_density(P_snn)
        self.init_weights(weights)
        self.setups = icalc.hamiltonian.setups
        self.L_iter: list[float] = []
        self.use_lebedev = use_lebedev
        self.calculate_lagrange_mu = calculate_lagrange_mu
        self.vxc0_sg = vxc0_sg

    def init_target_density(self, P_snn: Optional[ArrayND] = None) -> None:
        # TODO: implement interpolation from the target density (calculation) values to the current setup
        if self.tcalc is self.icalc:
            self.denst = copy_density(
                self.tcalc.density, self.hamiltonian.spos_ac, self.wfs.kptband_comm
            )
            self.denst.calculate_normalized_charges_and_mix()
        else:
            self.denst = self.tcalc.density
        if P_snn is not None:
            nt_sG = np.zeros_like(self.denst.nt_sG)
            D_asp = self.denst.setups.empty_atomic_matrix(
                self.denst.ncomponents, self.denst.atom_partition
            )
            # zero initialize
            for a in D_asp:
                D_asp[a][:] = 0.0
            for kpt in self.tcalc.wfs.kpt_u:
                psit_nG = kpt.psit_nG
                if self.wfs.mode == "pw":
                    psit_nG = np.asarray([self.wfs.pd.ifft(psi_G) for psi_G in psit_nG])
                P_ani = kpt.P_ani
                nt_sG[kpt.s] += np.einsum(
                    "n..., nm, m...->...", psit_nG.conj(), P_snn[kpt.s], psit_nG
                ).real
                for a in D_asp:
                    D_ii = np.einsum(
                        "ni, nm, mj -> ij", P_ani[a].conj(), P_snn[kpt.s], P_ani[a]
                    )
                    D_asp[a][kpt.s] += pack_hermitian(D_ii)
            self.denst.initialize_directly_from_arrays(nt_sG, self.denst.nt_vG, D_asp)
            self.denst.calculate_normalized_charges_and_mix()

        self.denst.fixed = True
        # for debug
        self.D_aspt = {a: D_sp.copy() for a, D_sp in self.denst.D_asp.items()}
        self.nt_sGt = self.denst.nt_sG.copy()

    def initialize(self):
        self.initialized = True
        self.gd = self.hamiltonian.gd
        self.finegd = self.hamiltonian.finegd
        self.niter: int = 0
        self.eigensolver_name: str = getattr(self.wfs.eigensolver, "name", "")
        self.L: float = 0.0
        self.init_xckernel()
        self.init_paw()
        self.u_m: ArrayND = None
        self.p_mn: ArrayND = None
        self.coarse2fine = Transformer(self.gd, self.finegd, nn=self.stencil).apply

    def init_xckernel(self) -> None:
        self.poisson = create_poisson_solver()
        self.poisson.set_grid_descriptor(self.finegd)
        self.vfa_sg = fermi_amaldi(self.poisson, self.finegd, self.denst.nt_sg)
        # TODO: define these with atomdist workarrays
        self.vfa_aszg = {}
        self.vfat_aszg = {}
        for a, D_sp in self.denst.D_asp.items():
            # TODO: add more options for initialization
            self.vfa_aszg[a], self.vfat_aszg[a] = atomic_fermi_amaldi(
                self.setups[a], D_sp, self.use_lebedev
            )
        if self.vxc0_sg is not None:
            vxc0_sg = self.vxc0_sg
        else:
            vxc0_sg = self.vfa_sg
        self.xckernel = NumericalXCKernel(
            vxc_sg=vxc0_sg.copy(),
            vxc_aszg={a: vxc.copy() for (a, vxc) in self.vfa_aszg.items()},
            vxct_aszg={a: vxct.copy() for (a, vxct) in self.vfat_aszg.items()},
            use_lebedev=self.use_lebedev,
        )
        xc = NumericalXC(self.xckernel, self.stencil)
        xc.set_positions(self.hamiltonian.spos_ac)
        xc.set_grid_descriptor(self.finegd)
        self.hamiltonian.xc = xc

    def init_paw(self) -> None:
        self.B_apqL = {}
        for a, setup in enumerate(self.setups):
            xcc = setup.xc_correction
            self.B_apqL[a] = np.zeros(
                (xcc.ni * xcc.ni, xcc.B_pqL.shape[1], xcc.B_pqL.shape[2])
            )
            for q in range(xcc.B_pqL.shape[1]):
                for L in range(xcc.B_pqL.shape[2]):
                    self.B_apqL[a][:, q, L] = unpack_hermitian(
                        np.array(xcc.B_pqL[:, q, L])
                    ).flatten()
        self.Phi_aiii = {}
        symbols = {}
        for a, setup in enumerate(self.setups):
            # cache
            if setup.symbol in symbols:
                self.Phi_aiii[a] = self.Phi_aiii[symbols[setup.symbol]]
                continue
            else:
                symbols[setup.symbol] = a
            data = setup.data
            rgd = data.rgd
            ni = setup.ni  # Number of partial waves
            Phi_iii = np.zeros((ni, ni, ni))
            l_j = setup.l_j  # l-index for each radial function index j
            lmax = max(l_j)
            G_LLLL = super_gaunt(lmax)
            # (Real) radial functions for the partial waves
            phi_jg = data.phi_jg
            phit_jg = data.phit_jg
            pt_jg = data.pt_jg
            i1_counter = 0
            for j1, l1 in enumerate(l_j):
                i2_counter = 0
                c = 2 * l1 + 1
                for j2, l2 in enumerate(l_j):
                    dphipt_g = (
                        phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]
                    ) * pt_jg[j1]
                    for j3, l3 in enumerate(l_j):
                        i3_counter = 0
                        dphi = np.dot(rgd.r_g**2 * rgd.dr_g * dphipt_g, phit_jg[j3])
                        for m1 in range(2 * l1 + 1):
                            L1 = l1**2 + m1
                            i1 = i1_counter + m1
                            for m2 in range(2 * l2 + 1):
                                L2 = l2**2 + m2
                                i2 = i2_counter + m2
                                for m3 in range(2 * l3 + 1):
                                    L3 = l3**2 + m3
                                    i3 = i3_counter + m3
                                    # If the angular integral (super gaunt
                                    # coefficient) is finite,
                                    coeff = G_LLLL[L1, L2, L3, L1]
                                    if abs(coeff) > 1e-10:
                                        Phi_iii[i1, i2, i3] += c * coeff * dphi
                        i3_counter += 2 * l3 + 1
                    i2_counter += 2 * l2 + 1
                i1_counter += 2 * l1 + 1
            self.Phi_aiii[a] = Phi_iii

    def scf_step(self) -> None:
        scf = self.icalc.scf
        if self.eigensolver_name == "etdm-fdpw":
            # TODO: implement etdm-fdpw
            raise NotImplementedError
        else:
            self.wfs.eigensolver.iterate(self.hamiltonian, self.wfs)
            e_entropy = self.wfs.calculate_occupation_numbers(scf.fix_fermi_level)
            self.hamiltonian.get_energy(e_entropy, self.wfs, kin_en_using_band=False)

        self.density.update(self.wfs)
        # update hamiltonian using the target density
        self.hamiltonian.update(self.denst, self.wfs, kin_en_using_band=False)
        scf.niter += 1
        ctx = scf.check_convergence(
            self.density,
            self.hamiltonian,
            self.wfs,
            self.log,
            self.icalc.call_observers,
        )
        return ctx

    def update(
        self, v_sg: ArrayND, v_aszg: dict[ArrayND], vt_aszg: dict[ArrayND]
    ) -> None:
        if v_sg is not self.xckernel.vxc_sg:
            self.xckernel.vxc_sg[:] = v_sg
        if v_aszg is not self.xckernel.vxc_aszg:
            for a in v_aszg:
                self.xckernel.vxc_aszg[a][:] = v_aszg[a]
        if vt_aszg is not self.xckernel.vxct_aszg:
            for a in vt_aszg:
                self.xckernel.vxct_aszg[a][:] = vt_aszg[a]

        self.icalc.scf.reset()
        self.icalc.scf.niter = 0
        for t in range(self.n_scf):
            self.scf_step()
        assert np.allclose(self.denst.nt_sG, self.nt_sGt)
        for a, D_sp in self.denst.D_asp.items():
            assert np.allclose(D_sp, self.D_aspt[a])
        # update coefficients
        self.u_um = {}
        self.p_umn = {}
        for u, kpt in enumerate(self.wfs.kpt_u):
            self.calculate_step(kpt)
            self.u_um[u] = self.u_m
            self.p_umn[u] = self.p_mn

    def calculate_step(self, kpt) -> None:
        self.dn_sG = self.denst.nt_sG - self.density.nt_sG
        # TODO: replace the integrals with wfs.integrate() (work with reciprocal space for pw)
        # add the kpoint and spin stuff
        occupied_m = self.occupied_m = np.nonzero(kpt.f_n > 1e-10)[0]
        f_m = self.f_m = kpt.f_n[occupied_m]
        self.psi_nG = kpt.psit_nG
        if self.wfs.mode == "pw":
            self.psi_nG = np.asarray(
                [self.wfs.pd.ifft(psi_G) for psi_G in self.psi_nG]
            )  # local=True, safe=False
        self.psi_mG = self.psi_nG[occupied_m]
        e_n = kpt.eps_n
        w_mn = (e_n[:, np.newaxis] - e_n[np.newaxis, occupied_m]).T
        np.fill_diagonal(w_mn, 1.0)
        # to prevent division by zero for degenerate states
        w_mn[np.abs(w_mn) < 1e-20] = 1e-20
        self.w_mn = w_mn
        self.add_paw_correction_p(kpt)
        self.p_mn += (
            4.0
            * f_m[:, np.newaxis]
            * self.gd.integrate(
                self.psi_nG,
                self.psi_mG * self.dn_sG[kpt.s] * self.weights_G[np.newaxis],
                global_integral=True,
            ).T
            / w_mn
        )
        if self.calculate_lagrange_mu:
            self.add_paw_correction_u(kpt)
            self.u_m += (
                2.0
                * f_m
                * self.gd.integrate(
                    self.psi_mG.conj()
                    * (self.weights_G[np.newaxis] * self.dn_sG[kpt.s])
                    * self.psi_mG,
                    global_integral=True,
                )
            )
            for m in range(len(self.u_m)):
                self.p_mn[m, m] -= 2.0 * self.u_m[m]
        else:
            np.fill_diagonal(self.p_mn, 0.0)
        self.p_mn = self.p_mn.conj()

    def add_paw_correction_u(self, kpt) -> None:
        """add paw correction to the lagrange multiplier u_m"""
        P_ani = kpt.P_ani
        # TODO: replace with workarrays for parallel
        self.u_m = np.zeros(len(self.psi_mG), dtype=P_ani[0].dtype)
        for a, P_ni in P_ani.items():
            dD_ii = self.dD_aii[a]
            self.u_m += (
                2.0
                * self.f_m
                * np.einsum(
                    "mk, ij, mj, ijk->m",
                    P_ni[self.occupied_m].conj(),
                    dD_ii,
                    P_ni[self.occupied_m],
                    self.Phi_aiii[a],
                )
            )

    def add_paw_correction_p(self, kpt) -> None:
        """add paw correction to the expansion coefficients p_mn"""
        P_ani = kpt.P_ani
        self.p_mn = np.zeros((len(self.psi_mG), len(self.psi_nG)), dtype=P_ani[0].dtype)
        self.dD_aii = {}
        for a, P_ni in P_ani.items():
            self.dD_aii[a] = dD_ii = unpack_hermitian(
                self.denst.D_asp[a][kpt.s] - self.density.D_asp[a][kpt.s]
            )
            # dD_ii = self.dD_aii[a]
            self.p_mn += (
                4
                * self.f_m[:, np.newaxis]
                * np.einsum(
                    "nk, ij, mj, ijk ->mn",
                    P_ni.conj(),
                    dD_ii,
                    P_ni[self.occupied_m],
                    self.Phi_aiii[a],
                )
                / self.w_mn
            )

    def init_weights(self, weights: str | ArrayND) -> None:
        # EOJ : This needs an overhaul
        if isinstance(weights, ArrayND):
            self.weights_G = weights
        elif weights == "uniform":  # EOJ: This
            self.weights_G = np.ones_like(self.denst.nt_sG[0])
        elif weights == "square_root_density_mask":
            self.weights_G = np.sqrt(np.sum(self.denst.nt_sG, axis=0))

        elif weights == "density":  # EOJ: Or this
            self.weights_G = np.sum(self.denst.nt_sG, axis=0)
        elif weights == "inverse_density":  #
            self.weights_G = 1.0 / (np.sum(self.denst.nt_sG, axis=0) + 1e-10)
        elif weights == "inverse_density_squared":  #
            self.weights_G = 1.0 / (np.sum(self.denst.nt_sG, axis=0) ** 2 + 1e-20)
        elif weights == "inverse_density_square_root":
            self.weights_G = 1.0 / (np.sqrt(np.sum(self.denst.nt_sG, axis=0)) + 1e-5)
        elif weights == "inverse_density_square_root_mask":
            from ase.units import Bohr

            self.com = self.tcalc.atoms.get_center_of_mass() / Bohr
            self.weights_G = 1.0 / (np.sqrt(np.sum(self.denst.nt_sG, axis=0)) + 1e-5)
            # Determine mask based on distance of grid point relative to center
            #
            center = self.com
            mask = self.get_radial_mask(center, 2.5)
            self.weights_G *= mask
        elif (
            weights == "mask"
        ):  # Possibly ... we should just enforce that the vxc goes to the correct Fermi-Amaldi limit
            self.weights_G = np.zeros_like(self.denst.nt_sG[0])
            self.weights_G[self.denst.nt_sG[0] > 1e-5] = 1.0
        #
        elif weights == "fr":
            nt = np.sum(self.denst.nt_sG, axis=0)
            self.weights_G = nt / (nt + 1e-6)
        else:
            raise ValueError(
                'weights must be one of "uniform", "density", \
                 "inverse_density", "inverse_density_squared" and "mask"'
            )

    def get_radial_mask(self, r_v, r_c):
        # EOJ : this needs to be tested (visualized?)
        r_d = self.gd.get_grid_point_distance_vectors(r_v)
        r_d = np.sqrt((r_d**2).sum(axis=0))
        return r_d <= r_c

    def lagrangian(self) -> float:
        self.L = self.gd.integrate(self.dn_sG**2).sum()
        self.D_err = 0.0
        for a, D_sp in self.dD_aii.items():
            self.D_err += np.mean(D_sp**2)
        self.L_iter.append(self.L)
        return self.L

    def jacobian(self, kpt) -> ArrayND:
        # TODO: include spin polarization and kpoints
        J_sG = np.einsum(
            "mn,n...,m...->...", self.p_mn, self.psi_nG.conj(), self.psi_mG
        )[np.newaxis].real
        self.J_sg = self.coarse2fine(J_sG)
        if self.lreg > 0.0:
            print(self.lreg)
            self.J_sg += (
                2
                * self.lreg
                * (self.xckernel.vxc_sg - self.vfa_sg)
                / self.density.nt_sg
            )
        return self.J_sg

    def jacobian_paw(self, kpt) -> tuple[ArrayND, ArrayND]:
        self.J_aszg = {}
        self.Jt_aszg = {}
        for a, P_ni in kpt.P_ani.items():
            self.J_aszg[a] = np.zeros_like(self.xckernel.vxc_aszg[a])
            self.Jt_aszg[a] = np.zeros_like(self.xckernel.vxct_aszg[a])
            setup = self.setups[a]
            xcc = setup.xc_correction
            if xcc is None:
                raise ValueError("No XC correction for atom %d" % a)
            dp_ii = np.einsum(
                "ni, mn, mj->ij", P_ni.conj(), self.p_mn, P_ni[self.occupied_m]
            )
            dp_Lq = np.inner(dp_ii.flatten(), self.B_apqL[a].T)
            dv_Lg = np.dot(dp_Lq, xcc.n_qg)
            dvt_Lg = np.dot(dp_Lq, xcc.nt_qg)
            Lmax = len(dvt_Lg)
            if self.use_lebedev:
                for n, Y_L in enumerate(Y_nL[:, :Lmax]):
                    self.J_aszg[a][kpt.s][n] = np.dot(Y_L, dv_Lg).real
                    self.Jt_aszg[a][kpt.s][n] = np.dot(Y_L, dvt_Lg).real
            else:
                self.J_aszg[a][kpt.s] = dv_Lg.real
                self.Jt_aszg[a][kpt.s] = dvt_Lg.real

            if self.lregpaw > 0.0:
                print(self.lregpaw)
                self.J_aszg[a][kpt.s] += (
                    2
                    * self.lregpaw
                    * (self.xckernel.vxc_aszg[a][kpt.s] - self.vfa_aszg[a][kpt.s])
                )
                self.Jt_aszg[a][kpt.s] += (
                    2
                    * self.lregpaw
                    * (self.xckernel.vxct_aszg[a][kpt.s] - self.vfat_aszg[a][kpt.s])
                )
        return self.J_aszg, self.Jt_aszg

    def converged(self, Ltol, LDtol, Jtol) -> None:
        return (
            np.abs(self.L) < Ltol
            and np.max(np.abs(self.J_sg)) < Jtol
            and (self.D_err < LDtol)
        )

    def debuglog(self, msg: str) -> None:
        if debug:
            parprint(msg)

    def warmup(self) -> None:
        self.log("---- Running single-point calculation with the intial guess ----")
        scf = self.icalc.scf
        scf.reset()
        scf.niter = 0
        for i in range(scf.maxiter):
            self.scf_step()
            if scf.converged:
                break
                self.log(
                    f"---- Single-point calculation converged after {scf.niter} iterations ----"
                )
        else:
            self.log(
                f"---- warning: Single-point calculation did not converge after {scf.maxiter} iterations ----"
            )
        self.icalc.summary()

    def adam_step(self, lr, vxc, J, m, mm):
        epsilon = 1e-8
        beta1 = self.adam_params["beta1"]
        beta2 = self.adam_params["beta2"]
        t = self.niter
        # Update biased first and second moment estimate
        m[:] = beta1 * m + (1.0 - beta1) * J
        mm[:] = beta2 * mm + (1.0 - beta2) * J**2
        m_hat = m / (1.0 - beta1**t)
        mm_hat = mm / (1.0 - beta2**t)
        # Update vxc
        vxc[:] -= lr * m_hat / (np.sqrt(mm_hat) + epsilon)

    def run_adam(
        self,
        maxiter: int = 300,
        lr_pseudo: float = 0.001,
        lr_paw: float = 0.001,
        Ltol: float = 1e-6,
        LDtol: float = 1e-4,
        Jtol: float = 1e-9,
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
        - LDtol: tolerance for the atomic density matrices.
        - Jtol: tolerance for the Jacobian of the smooth potential.
                [max(abs(J)) < Jtol for a successful termination]
        - beta1: exponential decay rate for the first moment estimates.
        - beta2: exponential decay rate for the second moment estimates.
        Returns:
        - L_iter: list of Lagrangian values at each iteration.
        """
        if not self.initialized:
            self.initialize()
        self.adam_params = {
            "lr_pseudo": lr_pseudo,
            "lr_paw": lr_paw,
            "beta1": beta1,
            "beta2": beta2,
            "maxiter": maxiter,
            "Ltol": Ltol,
            "Jtol": Jtol,
        }
        vxc_sg = self.xckernel.vxc_sg
        vxc_aszg = self.xckernel.vxc_aszg
        vxct_aszg = self.xckernel.vxct_aszg
        # initialize first moment vector
        m_sg = np.zeros_like(vxc_sg)
        m_aszg = {a: np.zeros_like(vxc) for a, vxc in vxc_aszg.items()}
        mt_aszg = {a: np.zeros_like(vxct) for a, vxct in vxct_aszg.items()}
        # initialize second moment vector
        mm_sg = np.zeros_like(vxc_sg)
        mm_aszg = {a: np.zeros_like(vxc) for a, vxc in vxc_aszg.items()}
        mmt_aszg = {a: np.zeros_like(vxct) for a, vxct in vxct_aszg.items()}
        # optimization loop
        L_iter = self.L_iter = []
        self.log("====> Running Adam optimizer for iKS <====")
        self.log(
            "=>  {:>5} | {:>10} | {:>10} | {:>10} | {:>10}".format(
                "iter", "L", "max(J)", "max(J_a)", "max(Jt_a)"
            )
        )
        self.log.fd.flush()

        for t in range(1, maxiter):
            self.niter = t
            # step
            self.update(vxc_sg, vxc_aszg, vxct_aszg)
            L = self.lagrangian()
            # jacobian
            # then for each kpoint we should have a separate calculation?
            for kpt in self.wfs.kpt_u:
                J_sg = self.jacobian(kpt)
                J_aszg, Jt_aszg = self.jacobian_paw(kpt)
            # pseudo
            self.adam_step(lr_pseudo, vxc_sg, J_sg, m_sg, mm_sg)
            # paw
            for a in vxc_aszg:
                self.adam_step(lr_paw, vxc_aszg[a], J_aszg[a], m_aszg[a], mm_aszg[a])
                self.adam_step(
                    lr_paw, vxct_aszg[a], Jt_aszg[a], mt_aszg[a], mmt_aszg[a]
                )
            # log
            self.log(
                "=>  {:>5} | {:>10.6e} | {:>10.6e} | {:>10.6e} | {:>10.6e}".format(
                    t,
                    L,
                    np.abs(J_sg).max(),
                    np.max([np.abs(J_sLg).max() for J_sLg in J_aszg.values()]),
                    np.max([np.abs(Jt_sLg).max() for Jt_sLg in Jt_aszg.values()]),
                )
            )
            self.log.fd.flush()
            # check convergence
            if self.converged(Ltol, LDtol, Jtol):
                break
        self.update(vxc_sg, vxc_aszg, vxct_aszg)
        return L_iter

    def run_fastest_descent(
        self,
        maxiter: int = 300,
        lr_pseudo: float = 1.0,
        lr_paw: float = 1.0,
        Ltol: float = 1e-6,
        LDtol: float = 1e-4,
        Jtol: float = 1e-9,
        lreg: float = 0.0,
        lregpaw: float = 0.0,
    ) -> list[float]:
        """
        Adam optimizer to minimize the Lagrangian function with
                        respect to smooth and atomic xc-potentials.
        Parameters:
        - maxiter: number of iterations for the optimization.
        - lr_pseudo: learning rate used for updating the smooth xc-potential.
        - lr_paw: learning rate used for updating the atomic potentials.
        - Ltol: tolerance for the Lagrangian (ps-density).
        - LDtol: tolerance for the atomic density matrices.
        - Jtol: tolerance for the Jacobian of the smooth potential.
                [max(abs(J)) < Jtol for a successful termination]
        lreg: regularization parameter for the smooth potential
        lregpaw: regularization parameter for the atomic potentials.
        Returns:
        - L_iter: list of Lagrangian values at each iteration.
        """
        self.lreg = lreg
        self.lregpaw = lregpaw
        if not self.initialized:
            self.initialize()
        vxc_sg = self.xckernel.vxc_sg
        vxc_aszg = self.xckernel.vxc_aszg
        vxct_aszg = self.xckernel.vxct_aszg
        # optimization loop
        L_iter = self.L_iter = []
        self.log("====> Running fastest descent iKS <====")
        self.log(
            "=>  {:>5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
                "iter",
                "ps-density err",
                "D_asp err",
                "max(J)",
                "max(J_a)",
                "max(Jt_a)",
            )
        )
        self.log.fd.flush()

        for t in range(1, maxiter):
            self.niter = t
            # step
            self.update(vxc_sg, vxc_aszg, vxct_aszg)
            L = self.lagrangian()
            # jacobian
            # then for each kpoint we should have a separate calculation?
            for kpt in self.wfs.kpt_u:
                J_sg = self.jacobian(kpt)
                J_aszg, Jt_aszg = self.jacobian_paw(kpt)
            # pseudo
            vxc_sg -= lr_pseudo * J_sg
            # paw
            for a in vxc_aszg:
                vxc_aszg[a] -= lr_paw * J_aszg[a]
                vxct_aszg[a] -= lr_paw * Jt_aszg[a]
            # log
            self.log(
                "=>  {:>5} | {:>10.6e} | {:>10.6e} | {:>10.6e} | {:>10.6e} | {:>10.6e}".format(
                    t,
                    L,
                    self.D_err,
                    np.abs(J_sg).max(),
                    np.max([np.abs(J_sLg).max() for J_sLg in J_aszg.values()]),
                    np.max([np.abs(Jt_sLg).max() for Jt_sLg in Jt_aszg.values()]),
                )
            )
            self.log.fd.flush()
            # check convergence
            if self.converged(Ltol, LDtol, Jtol):
                break
        self.update(vxc_sg, vxc_aszg, vxct_aszg)
        return L_iter

    def run_lbfgs(self):
        from scipy.optimize import minimize

        # optimization loop
        self.L_iter = []
        self.log("====> Running Adam optimizer for iKS <====")
        self.log(
            "=>  {:>5} | {:>10} | {:>10} | {:>10} | {:>10}".format(
                "iter", "L", "max(J)", "max(J_a)", "max(Jt_a)"
            )
        )
        self.log.fd.flush()
        vxc_sg = self.xckernel.vxc_sg
        vxc_aszg = self.xckernel.vxc_aszg
        vxct_aszg = self.xckernel.vxct_aszg
        self.update(vxc_sg, vxc_aszg, vxct_aszg)

        def jac(v):
            n = vxc_sg.size
            vxc_sg[:] = v[:n].reshape(vxc_sg.shape)
            for a in vxc_aszg:
                na = vxc_aszg[a].size
                vxc_aszg[a][n : n + na] = v[
                    vxc_sg.size : vxc_sg.size + vxc_aszg[a].size
                ].reshape(vxc_aszg[a].shape)
                n += na
                na = vxct_aszg[a].size
                vxct_aszg[a][:] = v[n : n + na].reshape(vxct_aszg[a].shape)
                n += na
            self.update(vxc_sg, vxc_aszg, vxct_aszg)
            J_sg = self.jacobian(self.wfs.kpt_u[0])
            J_aszg, Jt_aszg = self.jacobian_paw(self.wfs.kpt_u[0])
            J = np.concatenate(
                [J_sg.flatten()]
                + [J_aszg[a].flatten() for a in J_aszg]
                + [Jt_aszg[a].flatten() for a in Jt_aszg]
            )
            self.log(
                "=>  {:>10.6e} | {:>10.6e} | {:>10.6e} | {:>10.6e}".format(
                    self.L,
                    np.abs(J_sg).max(),
                    np.max([np.abs(J_sLg).max() for J_sLg in J_aszg.values()]),
                    np.max([np.abs(Jt_sLg).max() for Jt_sLg in Jt_aszg.values()]),
                )
            )
            return J.flatten()

        def lag(v):
            n = vxc_sg.size
            vxc_sg[:] = v[:n].reshape(vxc_sg.shape)
            for a in vxc_aszg:
                na = vxc_aszg[a].size
                vxc_aszg[a][n : n + na] = v[
                    vxc_sg.size : vxc_sg.size + vxc_aszg[a].size
                ].reshape(vxc_aszg[a].shape)
                n += na
                na = vxct_aszg[a].size
                vxct_aszg[a][:] = v[n : n + na].reshape(vxct_aszg[a].shape)
                n += na
            self.update(vxc_sg, vxc_aszg, vxct_aszg)
            return self.lagrangian()

        v0 = np.concatenate(
            [vxc_sg.flatten()]
            + [vxc_aszg[a].flatten() for a in vxc_aszg]
            + [vxct_aszg[a].flatten() for a in vxct_aszg]
        )
        minimizer = minimize(
            lag,
            v0,
            method="L-BFGS-B",
            jac=jac,
            options={"disp": True},
        )
        self.log(minimizer)
        self.log.fd.flush()
        return self.L_iter
