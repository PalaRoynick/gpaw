# from itertools import cycle
from typing import Dict, Union

import numpy as np
from gpaw.atom.aeatom import AllElectronAtom

# from gpaw.atom.all_electron import AllElectron
from gpaw.typing import Array1D
from gpaw.utilities import hartree
from gpaw.xc.numericalxc import NumericalXC, NumericalXCKernel

# from scipy.optimize import minimize


class _iKSAE:
    def __init__(self, symbol, nt_g, weights="uniform"):
        self.nt_g = nt_g
        self.niter = 0
        self.init_weights(weights)
        self.init_xc()
        self.mu_i = None
        self.p_ij = None
        self.L = 0.0
        self.prev_L = 0.0
        self.L_iter = []
        self.J_g = None

    def init_weights(self, weights: Union[str, Array1D]) -> None:
        if isinstance(weights, Array1D):
            self.weights_g = weights
        elif weights == "uniform":
            self.weights_g = np.ones_like(self.nt_g)
        elif weights == "density":
            self.weights_g = np.sum(self.nt_g, axis=0)
        elif weights == "density_squared":
            self.weights_g = np.sum(self.nt_g, axis=0)
        elif weights == "inverse_density":
            self.weights_g = 1.0 / (self.nt_g + 1e-10)
        elif weights == "inverse_density_squared":
            self.weights_g = 1.0 / (self.nt_g**2 + 1e-20)
        elif weights == "mask":
            self.weights_g = np.zeros_like(self.nt_g)
            self.weights_g[self.nt_g[0] > 1e-5] = 1.0
        else:
            raise ValueError(
                'weights must be one of "uniform", "density", \
                "inverse_density", "inverse_density_squared" or "mask"'
            )

    def init_xc(self):
        self.xc = NumericalXC(NumericalXCKernel())
        self.xc.kernel.set_rgd(self.rgd)
        self.xc.kernel.vxc_asg = self.fermi_amaldi()[None, None]

    def fermi_amaldi(self):
        vH = self.rgd.empty()
        hartree(0, self.nt_g * self.rgd.r_g * self.rgd.dr_g, self.rgd.r_g, vH)
        vH[1:] /= self.rgd.r_g[1:]
        vH[0] = vH[1]
        return -vH / self.Z

    def lagrangian(self, vxc_g=None):
        self.prev_L = self.L
        if vxc_g is not None:
            self.update(vxc_g)
        self.L = self.rgd.integrate(self.diff_g**2 * self.weights_g)
        return self.L

    def jacobian(self, vxc_g=None):
        if vxc_g is not None:
            if not np.allclose(self.xc.vxc_asg[0, 0], vxc_g, atol=1e-30):
                self.update(vxc_g)
        self.J_g = np.einsum("ij,j...,i...->...", self.p_ij, self.phi_jg, self.phi_ig)
        return self.J_g

    def update(self, vxc_g):
        raise NotImplementedError

    def run_adam(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        Ltol=1e-4,
        Jtol=1e-6,
        lambda_reg=0.0,
        epsilon=1e-12,
        maxiter=300,
    ):
        self.niter = 0
        self.L_iter = []
        vxc_g = self.xc.vxc_asg[0, 0]
        # first moment vector
        m_g = np.zeros_like(vxc_g)
        # second moment vector
        v_g = np.zeros_like(vxc_g)
        for t in range(1, maxiter + 1):
            self.update(vxc_g)
            L = self.lagrangian()
            J_g = self.jacobian()
            self.log(
                f"iter: {t}, prev_L:{self.prev_L}, L: {
                L}, J: {np.max(np.abs(J_g))}"
            )
            J_g += 2.0 * lambda_reg * vxc_g
            m_g = beta1 * m_g + (1 - beta1) * J_g
            v_g = beta2 * v_g + (1 - beta2) * (J_g**2)
            vxc_g = vxc_g - learning_rate * m_g / (1 - beta1**t) / (
                np.sqrt(v_g / (1 - beta2**t)) + epsilon
            )
            self.L_iter.append(L)
            if t % 100 == 0:
                np.save(f"iter{t}_vxc_g.npy", vxc_g)
                np.save(f"iter{t}_J_g.npy", J_g)
            if np.abs(L) < Ltol or np.max(np.abs(J_g)) < Jtol:
                break
        self.update(vxc_g)
        np.save("vxc_g.npy", vxc_g)
        np.save("J_max.npy", self.J_max)
        np.save("ortho.npy", self.ortho)
        np.save("norm.npy", self.norm)
        return self.L_iter

    def run_gd(
        self, maxiter=313, Ltol=1e-3, Jtol=1e-6, learning_rate=1.0, lambda_reg=0.0
    ):
        self.niter = 0
        self.L_iter = []
        vxc_g = self.xc.vxc_asg[0, 0]
        for iter in range(maxiter):
            self.update(vxc_g)
            L = self.lagrangian()
            J_g = self.jacobian()
            J_g += 2.0 * lambda_reg * vxc_g
            self.L_iter.append(L)
            vxc_g -= learning_rate * J_g
            self.log(f"iter: {iter}, L: {L}, J: {np.max(np.abs(J_g))}")
            if np.abs(L) < Ltol or np.max(np.abs(J_g)) < Jtol:
                break
        return self.L_iter


class iKSAEAtom(_iKSAE, AllElectronAtom):
    def __init__(
        self,
        symbol,
        nt_g,
        weights="uniform",
        ngpts=2000,
        rcut=50.0,
        alpha1=0.01,
        alpha2=None,
        eps=1.0e-7,
        ngauss=50,
        **aekwargs,
    ):
        AllElectronAtom.__init__(self, symbol, **aekwargs)
        self.initialize(
            ngpts=ngpts, rcut=rcut, ngauss=ngauss, alpha1=alpha1, alpha2=alpha2, eps=eps
        )
        _iKSAE.__init__(self, symbol, nt_g, weights)
        self.n_sg = None
        self.J_max = []
        self.ortho = []
        self.norm = []

    def calculate_density(self) -> Array1D:
        if self.n_sg is None:
            self.n_sg = np.copy(self.nt_g)[np.newaxis]

    def radial_integrate(self, f):
        return self.rgd.integrate(f) / (4.0 * np.pi)

    @property
    def na_g(self):
        na_sg = self.rgd.zeros(self.nspins)
        for channel in self.channels:
            na_sg[channel.s] += channel.calculate_density()
        return na_sg.sum(axis=0)

    def update(self, vxc_g):
        if self.channels is None:
            self.initialize()
        self.xc.kernel.vxc_asg = vxc_g[None, None]
        self.step()
        self.niter += 1
        self.diff_g = self.nt_g - self.na_g
        self.p_cij = {}
        self.phi_cig = {}
        self.phi_cjg = {}
        self.mu_ci = {}
        tmp_g = self.weights_g * self.diff_g
        for c, ch in enumerate(self.channels):
            phi_jg = ch.basis.expand(ch.C_nb)
            phi_ig = phi_jg[: len(ch.f_n)]
            w_ij = (ch.e_n[:, np.newaxis] - ch.e_n[np.newaxis, : len(ch.f_n)]).T
            np.fill_diagonal(w_ij, 1.0)
            mu_i = 2.0 * ch.f_n * self.radial_integrate(tmp_g * phi_ig**2)
            self.p_cij[c] = (
                4.0
                * self.radial_integrate(
                    np.einsum(
                        "j...,i...->ij...",
                        phi_jg,
                        tmp_g * phi_ig * ch.f_n[:, None] - 0.5 * mu_i[:, None] * phi_ig,
                    )
                )
                / w_ij
            )
            # np.fill_diagonal(self.p_cij[c], 0.0)
            self.log("max(p_ij):", np.max(np.abs(self.p_cij[c])))
            self.log("diagonal coeffs:", self.p_cij[c].diagonal())
            self.log(self.p_cij[c])
            self.log("------------------------")
            self.log(ch.e_n)
            self.phi_cjg[c] = phi_jg
            self.phi_cig[c] = phi_ig
            self.mu_ci[c] = mu_i

    def jacobian_channel(self, ch):
        self.log("jacobian_channel", ch)
        self.log(
            f"Normalization of phi_ig: {
            np.sum(self.mu_ci[ch] * (self.radial_integrate(self.phi_cig[ch] ** 2) - 1.0))}"
        )
        self.norm.append(
            (
                ch,
                np.sum(
                    self.mu_ci[ch]
                    * (self.radial_integrate(self.phi_cig[ch] ** 2) - 1.0)
                ),
            )
        )
        p_ig = np.einsum("ij,j...->i...", self.p_cij[ch], self.phi_cjg[ch])
        self.log(
            f"orthogonality of p_i and phi_i: {
            self.radial_integrate(p_ig * self.phi_cig[ch])}"
        )
        self.ortho.append((ch, self.radial_integrate(p_ig * self.phi_cig[ch]).sum()))
        self.log(self.radial_integrate(self.phi_cig[ch][0] * self.phi_cjg[ch][19]))
        self.log("diagonal coeffs:", self.p_cij[ch].diagonal())
        J_g = (
            (2 * self.channels[ch].l + 1)
            / (4 * np.pi)
            * np.einsum(
                "ij,j...,i...->...", self.p_cij[ch], self.phi_cjg[ch], self.phi_cig[ch]
            )
        )
        self.log("max(J_cg):", np.max(np.abs(J_g)), np.argmax(np.abs(J_g)))
        self.J_max.append((ch, np.argmax(np.abs(J_g)), np.max(np.abs(J_g))))
        return J_g

    def jacobian(self, vxc_g=None):
        if vxc_g is not None:
            if not np.allclose(self.xc.vxc_asg[0, 0], vxc_g, atol=1e-30):
                self.update(vxc_g)
        if self.J_g is None:
            self.J_g = self.rgd.zeros()
        else:
            self.J_g.fill(0.0)
        for c, ch in enumerate(self.channels):
            self.J_g += self.jacobian_channel(c)
        return self.J_g


class iKSRadial(iKSAEAtom):
    def update(self, vxc_g):
        if self.channels is None:
            self.initialize()
        self.xc.kernel.vxc_asg = vxc_g[None, None]
        self.step()
        self.niter += 1
        self.diff_g = self.nt_g - self.na_g
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
                * self.weights_g
                * self.diff_g
            )
            phi_jg = ch.basis.expand(ch.C_nb)
            phi_ig = phi_jg[: len(ch.f_n)]
            w_ij = (ch.e_n[:, np.newaxis] - ch.e_n[np.newaxis, : len(ch.f_n)]).T
            np.fill_diagonal(w_ij, 1.0)
            mu_i = self.radial_integrate(tmp_g * phi_ig**2)
            self.p_cij[c] = (
                self.radial_integrate(
                    np.einsum(
                        "j...,i...->ij...",
                        phi_jg,
                        2.0 * tmp_g * phi_ig - 2 * mu_i[:, None] * phi_ig,
                    )
                )
                / w_ij
            )
            # np.fill_diagonal(self.p_cij[c], 0.0)
            self.log("max(p_ij):", np.max(np.abs(self.p_cij[c])))
            self.log("diagonal coeffs:", self.p_cij[c].diagonal())
            self.log(self.p_cij[c])
            self.log("------------------------")
            self.log(ch.e_n)
            self.phi_cjg[c] = phi_jg
            self.phi_cig[c] = phi_ig
            self.mu_ci[c] = mu_i

    def jacobian_channel(self, ch):
        self.log("jacobian_channel", ch)
        self.log(
            f"Normalization of phi_ig: {
            np.sum(self.mu_ci[ch] * (self.radial_integrate(self.phi_cig[ch] ** 2) - 1.0))}"
        )
        self.norm.append(
            (
                ch,
                np.sum(
                    self.mu_ci[ch]
                    * (self.radial_integrate(self.phi_cig[ch] ** 2) - 1.0)
                ),
            )
        )
        p_ig = np.einsum("ij,j...->i...", self.p_cij[ch], self.phi_cjg[ch])
        self.log(
            f"orthogonality of p_i and phi_i: {
            self.radial_integrate(p_ig * self.phi_cig[ch])}"
        )
        self.ortho.append((ch, self.radial_integrate(p_ig * self.phi_cig[ch]).sum()))
        self.log(self.radial_integrate(self.phi_cig[ch][0] * self.phi_cjg[ch][19]))
        self.log("diagonal coeffs:", self.p_cij[ch].diagonal())
        J_g = (
            (2 * self.channels[ch].l + 1)
            / (4 * np.pi)
            * np.einsum(
                "ij,j...,i...->...", self.p_cij[ch], self.phi_cjg[ch], self.phi_cig[ch]
            )
            * self.rgd.r_g**2
        )
        J_g[0] = J_g[1]
        self.log("max(J_cg):", np.max(np.abs(J_g)), np.argmax(np.abs(J_g)))
        self.J_max.append((ch, np.argmax(np.abs(J_g)), np.max(np.abs(J_g))))
        return J_g
