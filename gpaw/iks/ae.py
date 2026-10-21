# from itertools import cycle
from typing import Union

import numpy as np
from gpaw.atom.aeatom import AllElectronAtom

# from gpaw.atom.all_electron import AllElectron
from gpaw.typing import Array1D
from gpaw.utilities import hartree
from gpaw.xc.numericalxc import NXCRadialCalculator


class _iKSAE:
    def __init__(self, symbol, nt_sg, weights="uniform"):
        self.nt_sg = nt_sg
        self.nspsins = len(nt_sg)
        self.niter = 0
        self.init_weights(weights)
        self.xc = NXCRadialCalculator(None, "lebedev", True)
        self.L = 0.0
        self.prev_L = 0.0
        self.L_iter = []
        self.J_g = None

    def init_weights(self, weights: Union[str, Array1D]) -> None:
        if isinstance(weights, Array1D):
            self.weights_sg = weights
        elif weights == "uniform":
            self.weights_sg = np.ones_like(self.nt_sg)
        elif weights == "density":
            self.weights_sg = np.sum(self.nt_sg, axis=0)
        elif weights == "density_squared":
            self.weights_sg = np.sum(self.nt_sg, axis=0)
        elif weights == "inverse_density":
            self.weights_sg = 1.0 / (self.nt_sg + 1e-10)
        elif weights == "inverse_density_squared":
            self.weights_sg = 1.0 / (self.nt_sg**2 + 1e-20)
        elif weights == "mask":
            self.weights_sg = np.zeros_like(self.nt_sg)
            self.weights_sg[self.nt_sg > 1e-5] = 1.0
        else:
            raise ValueError(
                'weights must be one of "uniform", "density", \
                "inverse_density", "inverse_density_squared" or "mask"'
            )

    def fermi_amaldi(self):
        vH_sg = self.rgd.empty(self.nspins)
        for s in range(self.nspins):
            hartree(
                0, self.nt_sg[s] * self.rgd.r_g *
                self.rgd.dr_g, self.rgd.r_g, vH_sg[s]
            )
            vH_sg[s, 1:] /= self.rgd.r_g[1:]
            vH_sg[s, 0] = vH_sg[s, 1]
        return -vH_sg / self.Z

    def lagrangian(self):
        self.prev_L = self.L
        self.L = self.rgd.integrate(self.diff_sg**2 * self.weights_sg).sum(0)
        return self.L

    def jacobian(self):
        self.J_sg = np.einsum(
            "ij,sj...,si...->s...", self.p_sij, self.phi_sjg, self.phi_sig
        )
        return self.J_g

    def update(self):
        raise NotImplementedError

    def run_adam(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        Ltol=1e-5,
        Jtol=1e-6,
        epsilon=1e-12,
        maxiter=300,
    ):
        self.niter = 0
        self.L_iter = []
        vxc_sg = self.xc.vxc_szg[:, 0]
        # first moment vector
        m_sg = np.zeros_like(vxc_sg)
        # second moment vector
        mm_sg = np.zeros_like(vxc_sg)
        for t in range(1, maxiter + 1):
            self.update()
            L = self.lagrangian()
            J_sg = self.jacobian()
            self.log(
                f"iter: {t}, prev_L:{self.prev_L}, L: {L}, J: {np.max(np.abs(J_sg))}"
            )
            m_g = beta1 * m_sg + (1 - beta1) * J_sg
            mm_sg = beta2 * mm_sg + (1 - beta2) * (J_sg**2)
            vxc_sg[:] -= (
                learning_rate
                * m_g
                / (1 - beta1**t)
                / (np.sqrt(mm_sg / (1 - beta2**t)) + epsilon)
            )
            self.L_iter.append(L)
            # if t % 100 == 0:
            #     np.save(f"iter{t}_vxc_g.npy", vxc_sg)
            #     np.save(f"iter{t}_J_g.npy", J_g)
            if np.abs(L) < Ltol or np.max(np.abs(J_sg)) < Jtol:
                break
        self.update()
        # np.save("vxc_g.npy", vxc_g)
        # np.save("J_max.npy", self.J_max)
        # np.save("ortho.npy", self.ortho)
        # np.save("norm.npy", self.norm)
        return self.L_iter

    def run_steepest_descent(self, maxiter=313, Ltol=1e-5, Jtol=1e-6, learning_rate=1.0):
        self.niter = 0
        self.L_iter = []
        vxc_sg = self.xc.vxc_szg[:, 0]
        for iter in range(maxiter):
            self.update()
            L = self.lagrangian()
            J_sg = self.jacobian()
            self.L_iter.append(L)
            vxc_sg -= learning_rate * J_sg
            self.log(f"iter: {iter}, L: {L}, J: {np.max(np.abs(J_sg))}")
            if np.abs(L) < Ltol or np.max(np.abs(J_sg)) < Jtol:
                break
        return self.L_iter


class iKSAEAtom(_iKSAE, AllElectronAtom):
    def __init__(
        self,
        symbol,
        nt_sg,
        vxc0_sg=None,
        weights="uniform",
        ngpts=2000,
        rcut=50.0,
        alpha1=0.01,
        alpha2=None,
        eps=1.0e-7,
        ngauss=50,
        silent=True,
        **aekwargs,
    ):
        self.silent = silent
        _iKSAE.__init__(self, symbol, nt_sg, weights)
        AllElectronAtom.__init__(self, symbol, xc=self.xc, **aekwargs)
        self.initialize(
            ngpts=ngpts, rcut=rcut, ngauss=ngauss, alpha1=alpha1, alpha2=alpha2, eps=eps
        )
        if vxc0_sg is None:
            self.xc.vxc_szg = self.fermi_amaldi()[:, None]
        else:
            self.xc.vxc_szg = vxc0_sg[:, None]
        self.n_sg = None
        self.J_max = []
        self.ortho = []
        self.norm = []

    def calculate_density(self) -> Array1D:
        if self.n_sg is None:
            self.n_sg = np.copy(self.nt_sg)

    def log(self, *args, **kwargs):
        if not self.silent:
            return super().log(*args, **kwargs)

    def radial_integrate(self, a_xg):
        rgd = self.rgd
        return np.dot(a_xg[..., 1:], (rgd.r_g**2 * rgd.dr_g)[1:])

    @property
    def na_sg(self):
        na_sg = self.rgd.zeros(self.nspins)
        for channel in self.channels:
            na_sg[channel.s] += channel.calculate_density()
        return na_sg

    def update(self):
        if self.channels is None:
            self.initialize()
        self.step()
        self.niter += 1
        self.diff_sg = self.nt_sg - self.na_sg
        self.p_cij = {}
        self.phi_cig = {}
        self.phi_cjg = {}
        self.mu_ci = {}
        for c, ch in enumerate(self.channels):
            tmp_g = self.weights_sg[ch.s] * self.diff_sg[ch.s]
            phi_jg = ch.basis.expand(ch.C_nb)
            phi_ig = phi_jg[: len(ch.f_n)]
            w_ij = (ch.e_n[:, np.newaxis] -
                    ch.e_n[np.newaxis, : len(ch.f_n)]).T
            np.fill_diagonal(w_ij, 1.0)
            mu_i = 2.0 * ch.f_n * self.radial_integrate(tmp_g * phi_ig**2)
            self.p_cij[c] = (
                4.0
                * self.radial_integrate(
                    np.einsum(
                        "j...,i...->ij...",
                        phi_jg,
                        tmp_g * phi_ig * ch.f_n[:, None] -
                        0.5 * mu_i[:, None] * phi_ig,
                    )
                )
                / w_ij
            )
            np.fill_diagonal(self.p_cij[c], 0.0)
            self.phi_cjg[c] = phi_jg
            self.phi_cig[c] = phi_ig
            self.mu_ci[c] = mu_i

    def jacobian_channel(self, ch):
        J_g = (
            (2 * self.channels[ch].l + 1)
            / (4 * np.pi)
            * np.einsum(
                "ij,j...,i...->...", self.p_cij[ch], self.phi_cjg[ch], self.phi_cig[ch]
            )
        )
        return J_g

    def jacobian(self, vxc_sg=None):
        self.J_sg = self.rgd.zeros(self.nspins)
        for c, ch in enumerate(self.channels):
            self.J_sg[ch.s] += self.jacobian_channel(c)
        return self.J_sg


class iKSRadial(iKSAEAtom):
    def update(self):
        if self.channels is None:
            self.initialize()
        self.step()
        self.niter += 1
        self.diff_sg = self.nt_sg - self.na_sg
        self.p_cij = {}
        self.phi_cig = {}
        self.phi_cjg = {}
        self.mu_ci = {}
        for c, ch in enumerate(self.channels):
            tmp_g = (
                2.0
                * ch.f_n[:, None]
                * (2 * ch.l + 1)
                / (4 * np.pi)
                * self.weights_sg[ch.s]
                * self.diff_sg[ch.s]
            )
            phi_jg = ch.basis.expand(ch.C_nb)
            phi_ig = phi_jg[: len(ch.f_n)]
            w_ij = (ch.e_n[:, np.newaxis] -
                    ch.e_n[np.newaxis, : len(ch.f_n)]).T
            np.fill_diagonal(w_ij, 1.0)
            mu_i = self.radial_integrate(tmp_g * phi_ig**2)
            self.p_cij[c] = (
                self.radial_integrate(
                    np.einsum("j...,i...->ij...", phi_jg, 4.0 * tmp_g * phi_ig)
                )
                / w_ij
            )
            np.fill_diagonal(self.p_cij[c], 0.0)
            self.phi_cjg[c] = phi_jg
            self.phi_cig[c] = phi_ig
            self.mu_ci[c] = mu_i

    def jacobian_channel(self, ch):
        J_g = (
            (2 * self.channels[ch].l + 1)
            / (4 * np.pi)
            * np.einsum(
                "ij,j...,i...->...", self.p_cij[ch], self.phi_cjg[ch], self.phi_cig[ch]
            )
            # * self.rgd.r_g**2
        )
        J_g[0] = J_g[1]
        return J_g
