from math import pi, sqrt

import numpy as np

from gpaw.new import trace
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.xc.functional import XCFunctional
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.new.c import evaluate_lda_gpu


def stress_integral(gd, a1_g, a2_g=None):
    return gd.integrate(a1_g, a2_g, global_integral=False)


def stress_lda_term(gd, e_g, n_sg, v_sg):
    if e_g is None:
        P = 0
    else:
        P = stress_integral(gd, e_g)
    for v_g, n_g in zip(v_sg, n_sg):
        P -= stress_integral(gd, v_g, n_g)
    return P * np.eye(3)


class LDARadialExpansion:
    def __init__(self, rcalc, collinear=True):
        self.rcalc = rcalc
        self.collinear = collinear
        self.rgd = None

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg, xp=np):
        n_sLg = D_sLq @ n_qg
        dv_g = rgd.dv_g
        if self.collinear:
            n_sLg[:, 0] += nc0_sg
        else:
            n_sLg[0, 0] += 4 * nc0_sg[0]

        Lmax = n_sLg.shape[1]

        Y2_nL = xp.array(Y_nL[:, :Lmax])
        w_n = xp.array(weight_n)

        n_sng = Y2_nL[None] @ n_sLg
        e_ng = xp.zeros_like(n_sng, shape=n_sng.shape[1:])
        dedn_sng = xp.zeros_like(n_sng)
        if xp is np:
            self.rcalc(e_ng, n_sng, dedn_sng)
        else:
            evaluate_lda_gpu(n_sng, dedn_sng, e_ng)
        dEdD_snq = (dv_g * dedn_sng @ n_qg.T[None])
        dEdD_sqL = xp.einsum('snq, nL -> sqL', dEdD_snq,
                             Y2_nL[:, :Lmax] * w_n[:, None])
        E = rgd.integrate(w_n @ e_ng)

        return E, dEdD_sqL


class fake_rgd:
    def __init__(self, rgd, xp):
        self._rgd = rgd
        self.xp = xp
        dv_g = xp.array(rgd.dv_g)

    def integrate(self, x_g):
        return self.xp.sum(x_g * self._rgd.dv_g)


@trace
def calculate_paw_correction(expansion,
                             setup, D_sp, dEdD_sp=None,
                             addcoredensity=True, a=None,
                             xp=np):
    xcc = setup.xc_correction
    if xcc is None:
        return 0.0

    if xp is not np:
        if hasattr(xcc, 'gpu_rgd'):
            rgd = xcc.gpu_rgd
        else:
            xcc.gpu_rgd = fake_rgd(xcc.rgd, xp)
            rgd = xcc.gpu_rgd
    else:
        rgd = xcc.rgd

    nspins = len(D_sp)
    nc_g = xp.array(xcc.nc_g)
    nct_g = xp.array(xcc.nct_g)
    n_qg = xp.array(xcc.n_qg)
    nt_qg = xp.array(xcc.nt_qg)
    B_pqL = xp.array(xcc.B_pqL)

    if addcoredensity:
        nc0_sg = rgd.empty(nspins)
        nct0_sg = rgd.empty(nspins)
        nc0_sg[:] = sqrt(4 * pi) / nspins * nc_g
        nct0_sg[:] = sqrt(4 * pi) / nspins * nct_g
        if xcc.nc_corehole_g is not None and nspins == 2:
            nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
            nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
    else:
        nc0_sg = 0
        nct0_sg = 0

    D_sLq = xp.einsum('sp, pqL -> sLq', D_sp, B_pqL)

    e, dEdD_sqL = expansion(rgd, D_sLq, n_qg, nc0_sg)
    et, dEtdD_sqL = expansion(rgd, D_sLq, nt_qg, nct0_sg)

    if dEdD_sp is not None:
        dEdD_sp += np.inner((dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
                            xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))

    if addcoredensity:
        return e - et - xcc.e_xc0
    else:
        return e - et


class LDARadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, e_ng, n_nsg, dedn_nsg):

        self.kernel.calculate(e_ng, n_nsg, dedn_nsg)
        return e_ng, dedn_nsg


class LDA(XCFunctional):
    def __init__(self, kernel):
        self.kernel = kernel
        super().__init__(kernel.name, kernel.type)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        self.kernel.calculate(e_g, n_sg, v_sg)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        from gpaw.xc.noncollinear import NonCollinearLDAKernel
        collinear = not isinstance(self.kernel, NonCollinearLDAKernel)
        rcalc = LDARadialCalculator(self.kernel)
        expansion = LDARadialExpansion(rcalc, collinear)
        return calculate_paw_correction(expansion,
                                        setup, D_sp, dEdD_sp,
                                        addcoredensity, a)

    def calculate_radial(self, rgd, n_sLg, Y_L):
        rcalc = LDARadialCalculator(self.kernel)
        return rcalc(rgd, n_sLg, Y_L)

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        if e_g is None:
            e_g = rgd.empty()
        rcalc = LDARadialCalculator(self.kernel)
        e_g[:], dedn_sg = rcalc(rgd, n_sg[:, np.newaxis], [1.0])
        v_sg[:] = dedn_sg
        return rgd.integrate(e_g)

    def stress_tensor_contribution(self, n_sg, skip_sum=False):
        nspins = len(n_sg)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        self.calculate_impl(self.gd, n_sg, v_sg, e_g)
        stress_vv = stress_lda_term(self.gd, e_g, n_sg, v_sg)
        if not skip_sum:
            self.gd.comm.sum(stress_vv)
        return stress_vv


class PurePythonLDAKernel:
    def __init__(self):
        self.name = 'LDA'
        self.type = 'LDA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):

        e_g[:] = 0.
        if len(n_sg) == 1:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40

            # exchange
            lda_x(0, e_g, n, dedn_sg[0])
            # correlation
            lda_c(0, e_g, n, dedn_sg[0], 0)

        else:
            na = 2. * n_sg[0]
            na[na < 1e-20] = 1e-40
            nb = 2. * n_sg[1]
            nb[nb < 1e-20] = 1e-40
            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n

            # exchange
            lda_x(1, e_g, na, dedn_sg[0])
            lda_x(1, e_g, nb, dedn_sg[1])
            # correlation
            lda_c(1, e_g, n, dedn_sg, zeta)


def lda_x(spin, e, n, v):
    assert spin in [0, 1]
    C0I, C1, CC1, CC2, IF2 = lda_constants()

    rs = (C0I / n) ** (1 / 3.)
    ex = C1 / rs
    dexdrs = -ex / rs
    if spin == 0:
        e[:] += n * ex
    else:
        e[:] += 0.5 * n * ex
    v += ex - rs * dexdrs / 3.


def lda_c(spin, e, n, v, zeta):
    assert spin in [0, 1]
    C0I, C1, CC1, CC2, IF2 = lda_constants()

    rs = (C0I / n) ** (1 / 3.)
    ec, decdrs_0 = G(rs ** 0.5,
                     0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

    if spin == 0:
        e[:] += n * ec
        v += ec - rs * decdrs_0 / 3.
    else:
        e1, decdrs_1 = G(rs ** 0.5,
                         0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        alpha, dalphadrs = G(rs ** 0.5,
                             0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                             0.49671)
        alpha *= -1.
        dalphadrs *= -1.
        zp = 1.0 + zeta
        zm = 1.0 - zeta
        xp = zp ** (1 / 3.)
        xm = zm ** (1 / 3.)
        f = CC1 * (zp * xp + zm * xm - 2.0)
        f1 = CC2 * (xp - xm)
        zeta3 = zeta * zeta * zeta
        zeta4 = zeta * zeta * zeta * zeta
        x = 1.0 - zeta4
        decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                  decdrs_1 * f * zeta4 +
                  dalphadrs * f * x * IF2)
        decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                    f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
        ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
        e[:] += n * ec
        v[0] += ec - rs * decdrs / 3.0 - (zeta - 1.0) * decdzeta
        v[1] += ec - rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta


def G(rtrs, gamma, alpha1, beta1, beta2, beta3, beta4):
    Q0 = -2.0 * gamma * (1.0 + alpha1 * rtrs * rtrs)
    Q1 = 2.0 * gamma * rtrs * (beta1 +
                               rtrs * (beta2 +
                                       rtrs * (beta3 +
                                               rtrs * beta4)))
    G1 = Q0 * np.log(1.0 + 1.0 / Q1)
    dQ1drs = gamma * (beta1 / rtrs + 2.0 * beta2 +
                      rtrs * (3.0 * beta3 + 4.0 * beta4 * rtrs))
    dGdrs = -2.0 * gamma * alpha1 * G1 / Q0 - Q0 * dQ1drs / (Q1 * (Q1 + 1.0))
    return G1, dGdrs


def lda_constants():
    C0I = 0.238732414637843
    C1 = -0.45816529328314287
    CC1 = 1.9236610509315362
    CC2 = 2.5648814012420482
    IF2 = 0.58482236226346462
    return C0I, C1, CC1, CC2, IF2
