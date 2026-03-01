from math import pi, sqrt

import numpy as np

from gpaw.fd_operators import FDOperator, laplace
from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
from gpaw.sphere.spherical_harmonics import YL
from gpaw.utilities import divrl
from gpaw.utilities.blas import axpy
from gpaw.xc.functional import XCFunctional
from gpaw.xc.gga import add_radial_gradient_correction as gga_radial_gradient_correction
from gpaw.xc.gga import get_gradient_ops, gga_vars, radial_gga_vars
from gpaw.xc.lda import calculate_paw_correction
from gpaw.xc.pawcorrection import rnablaY_nLv

# from gpaw import debug
debug = False


class SecondDerivative(FDOperator):
    def __init__(self, gd, v, scale=1.0, n=1, dtype=float, xp=np):
        """Second derivative in v direction.

        gd: GridDescriptor
            Descriptor for grid.
        v: direction
            v = 0, 1, 2, 3, 4, 5 for d^2/dx^2, d^2/dy^2,
                d^2/dz^2 d^2/dydz, d^2/dxdz, d^2/dydx
        scale: float
            Scaling factor.
        n: int
            Range of stencil.  Stencil has O(h^(2n)) error.
        dtype: float or complex
            Data-type to work on.
        """
        # Order the 26 neighbor grid points after length
        # (reduced to 13 inequivalent):
        M_ic = np.indices((3, 3, 3)).reshape((3, -3)).T[-13:] - 1
        u_cv = gd.h_cv / (gd.h_cv**2).sum(1)[:, np.newaxis] ** 0.5
        u2_i = (np.dot(M_ic, u_cv) ** 2).sum(1)
        i_d = u2_i.argsort()
        # x^2, y^2, z^2, yz, xz, xy:
        m_mv = np.array(
            [(2, 0, 0), (0, 2, 0), (0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        )
        b_d = [0] * 6
        b_d[v] = 1
        # Try 3, 4, 5 and 6 of the shortest directions:
        for D in range(3, 7):
            h_dv = np.dot(M_ic[i_d[:D]], gd.h_cv)
            A_md = (h_dv ** m_mv[:, np.newaxis, :]).prod(2)
            # Find best stencil coefficients:
            a_d, residual, rank, s = np.linalg.lstsq(A_md, b_d, rcond=-1)
            if residual.sum() < 1e-14 / 3.0:
                if rank != D:
                    raise ValueError(
                        "SecondDerivative: "
                        "You have a weird unit cell!  "
                        "Try to use the maximally reduced Niggli cell.  "
                        "See the ase.build.niggli_reduce() function."
                    )
                break  # D directions was OK

        a_d *= scale
        offsets = [(0, 0, 0)]
        coefs = [laplace[n][0] * a_d.sum()]
        for d in range(D):
            M_c = M_ic[i_d[d]]
            offsets.extend(np.arange(1, n + 1)[:, np.newaxis] * M_c)
            coefs.extend(a_d[d] * np.array(laplace[n][1:]))
            offsets.extend(np.arange(-1, -n - 1, -1)[:, np.newaxis] * M_c)
            coefs.extend(a_d[d] * np.array(laplace[n][1:]))

        FDOperator.__init__(self, coefs, offsets, gd, dtype, xp=xp)

        self.description = (
            "%d*%d+1=%d point O(h^%d) finite-difference Second-derivative xx"
            % ((self.npoints - 1) // n, n, self.npoints, 2 * n)
        )


def get_hessian_ops(gd, n, scale=1.0, mode="mix"):
    H = np.empty((3, 3), dtype=object)

    def double_grad(gd, x, y, v, vp):
        tmp = np.empty_like(y)
        grad_v[v](x, tmp)
        grad_v[vp](tmp, y)

    if mode == "fd-op":
        for i in range(3):
            H[i, i] = SecondDerivative(gd, i, scale=scale, n=n).apply
        H[1, 2] = H[2, 1] = SecondDerivative(gd, 3, n=n, scale=scale).apply
        H[0, 2] = H[2, 0] = SecondDerivative(gd, 4, n=n, scale=scale).apply
        H[0, 1] = H[1, 0] = SecondDerivative(gd, 5, n=n, scale=scale).apply
    elif mode == "grad-op":
        grad_v = get_gradient_ops(gd, n, np)
        for i in range(3):
            H[i, i] = lambda x, y: double_grad(gd, x, y, i, i)
        H[1, 2] = H[2, 1] = lambda x, y: double_grad(gd, x, y, 1, 2)
        H[0, 2] = H[2, 0] = lambda x, y: double_grad(gd, x, y, 0, 2)
        H[0, 1] = H[1, 0] = lambda x, y: double_grad(gd, x, y, 0, 1)
    elif mode == "mix":
        for i in range(3):
            H[i, i] = SecondDerivative(gd, i, scale=scale, n=n).apply
        grad_v = get_gradient_ops(gd, n, np)
        H[1, 2] = H[2, 1] = lambda x, y: double_grad(gd, x, y, 1, 2)
        H[0, 2] = H[2, 0] = lambda x, y: double_grad(gd, x, y, 0, 2)
        H[0, 1] = H[1, 0] = lambda x, y: double_grad(gd, x, y, 0, 1)
    else:
        raise ValueError(f"mode {mode} is not implemented for hessian ops!")
    return H


def msvvg_diag(X_svvg):
    Y_svvg = np.zeros_like(X_svvg)
    for i in range(Y_svvg.shape[1]):
        Y_svvg[:, i, i] = X_svvg[:, i, i]
    return Y_svvg


def hessianrlYL(L, R):
    """Hessian(r^l Y_l) / r^(l-2)"""
    x, y, z = R
    dYdxx = dYdyy = dYdzz = 0.0
    dYdxy = dYdyz = dYdxz = 0.0
    # The 'abs' avoids error in case powx == 0 or powx==1
    for N, (powx, powy, powz) in YL[L]:
        px2 = abs(powx - 2) if powx > 1 else 1
        py2 = abs(powy - 2) if powy > 1 else 1
        pz2 = abs(powz - 2) if powz > 1 else 1
        dYdxx += N * powx * abs(powx - 1) * x**px2 * y**powy * z**powz
        dYdyy += N * powy * abs(powy - 1) * x**powx * y**py2 * z**powz
        dYdzz += N * powz * abs(powz - 1) * x**powx * y**powy * z**pz2
        dYdxy += N * powx * powy * \
            x ** abs(powx - 1) * y ** abs(powy - 1) * z**powz
        dYdyz += N * powy * powz * x**powx * \
            y ** abs(powy - 1) * z ** abs(powz - 1)
        dYdxz += N * powx * powz * \
            x ** abs(powx - 1) * y**powy * z ** abs(powz - 1)
    return np.array(
        [[dYdxx, dYdxy, dYdxz], [dYdxy, dYdyy, dYdyz],
            [dYdxz, dYdyz, dYdzz]], dtype="d"
    )


def hessianrl(l, R):
    """(Hessian of r^l) / r^(l-2)"""
    x, y, z = R
    return (l**2 - 2.0 * l) * np.array(
        [
            [x**2, x * y, x * z],
            [x * y, y**2, y * z],
            [x * z, y * z, z**2],
        ]
    ) + l * np.eye(3)


def radial_second_derivative(rgd, n_g, d2ndr2_g=None, method=2):
    """second partial derivative wrt r"""
    if d2ndr2_g is None:
        d2ndr2_g = rgd.empty()
    if method == 1:
        dndr_g = rgd.zeros()
        rgd.derivative(n_g, dndr_g)
        rgd.derivative(dndr_g, d2ndr2_g)
        d2ndr2_g[1] = (n_g[2] - 2 * n_g[1] + n_g[0]) / rgd.dr_g[1] ** 2
        d2ndr2_g[0] = d2ndr2_g[1]
        d2ndr2_g[-2] = (n_g[-3] - 2 * n_g[-2] + n_g[-1]) / rgd.dr_g[-2] ** 2
        d2ndr2_g[-1] = d2ndr2_g[-2]
    elif method == 2:
        d2ndg2_g = n_g[2:] - 2 * n_g[1:-1] + n_g[:-2]
        d2ndr2_g[1:-1] = d2ndg2_g / rgd.dr_g[1:-1] ** 2
        d2ndr2_g[0] = d2ndr2_g[1]
        d2ndr2_g[-1] = d2ndr2_g[-2]

    return d2ndr2_g


def radial_gradient_zvg(dfdr_zg, R_v):
    """Calculate the gradient of a radial function in Cartesian coordinates."""
    gradient_zvg = np.zeros(
        (dfdr_zg.shape[0], 3, dfdr_zg.shape[-1]), dtype="d")
    for v in range(3):
        gradient_zvg[:, v] = dfdr_zg * R_v[v]
    return gradient_zvg


def radial_hessian_zvvg(d2fdr2_zg, dfdr_zg, r_g, R_v):
    """Calculate the Hessian matrix of a radial function in
    Cartesian coordinates."""
    hessian_zvvg = np.zeros(
        (d2fdr2_zg.shape[0], 3, 3, d2fdr2_zg.shape[-1]), dtype="d")
    dftor_zg = np.array([divrl(dfdr_g, 1, r_g) for dfdr_g in dfdr_zg])
    d2fminusdftor_zg = d2fdr2_zg - dftor_zg
    for v in range(3):
        hessian_zvvg[:, v, v] = d2fminusdftor_zg * R_v[v] ** 2 + dftor_zg
    hessian_zvvg[:, 0, 1] = hessian_zvvg[:, 1,
                                         0] = d2fminusdftor_zg * R_v[0] * R_v[1]
    hessian_zvvg[:, 0, 2] = hessian_zvvg[:, 2,
                                         0] = d2fminusdftor_zg * R_v[0] * R_v[2]
    hessian_zvvg[:, 1, 2] = hessian_zvvg[:, 2,
                                         1] = d2fminusdftor_zg * R_v[1] * R_v[2]
    return hessian_zvvg


def radialxYL_hessian_vvg(
    rgd, n_Lg, dndr_Lg, d2ndr2_Lg, Y_L, R_v, rnablaY_Lv, r2HessianYL_Lvv
):
    hessian_vvg = rgd.zeros((3, 3))
    htmp_Lvvg = radial_hessian_zvvg(d2ndr2_Lg, dndr_Lg, rgd.r_g, R_v)
    hessian_vvg += np.einsum(Y_L, [1], htmp_Lvvg, [1, 2, 3, 4], [2, 3, 4])
    nablan_Lvg = radial_gradient_zvg(dndr_Lg, R_v)
    tmp_Lvg = np.copy(nablan_Lvg)
    tmp_Lvg[:, :, 1:] /= rgd.r_g[1:]
    tmp_Lvg[:, :, 0] = tmp_Lvg[:, :, 1]
    ny_vvg = np.einsum(tmp_Lvg, [1, 0, 2], rnablaY_Lv, [1, 3], [0, 3, 2])
    hessian_vvg += ny_vvg
    hessian_vvg += np.transpose(ny_vvg, [1, 0, 2])
    ntor2_Lg = np.array([divrl(n_g, 2, rgd.r_g) for n_g in n_Lg])
    hessian_vvg += np.sum(
        r2HessianYL_Lvv[..., np.newaxis] * ntor2_Lg[:, np.newaxis, np.newaxis], axis=0
    )
    return hessian_vvg


# ThetaMGGA functionals

MIN_SIGMA = 1e-20


def process_theta_dsigma(gd, n_sg, sigma_xg, gradn_svg, hessian_svvg):
    n_sg = n_sg.clip(min=0.0)
    sigma_sg = sigma_xg[::2].clip(min=MIN_SIGMA)

    inverse_sigma_sg = 1.0 / sigma_sg
    Vsigma_svg = 2.0 * \
        np.einsum("sij..., sj...->si...", hessian_svvg, gradn_svg)
    # alpha_1 = Vn.Vsigma part
    pref_isigma_sg = inverse_sigma_sg**2  # 1 / sigma^2
    alpha_sg = np.einsum("sv...,sv...->s...", gradn_svg, Vsigma_svg)
    dthetadalpha_sg = -4.0 * n_sg * pref_isigma_sg
    theta_sg = 4.0 + dthetadalpha_sg * alpha_sg
    dthetadn_sg = -4.0 * alpha_sg * pref_isigma_sg
    dthetadgradn_svg = (2.0 * dthetadalpha_sg[:, np.newaxis] * (
        Vsigma_svg - (2.0 * inverse_sigma_sg * alpha_sg)[:, np.newaxis] * gradn_svg))
    dthetadDsigma_svg = dthetadalpha_sg[:, np.newaxis] * gradn_svg
    # alpha_2 = V.Vsigma part
    pref_isigma_sg *= inverse_sigma_sg  # 1 / sigma^3
    alpha_sg = np.einsum("sv..., sv...->s...", Vsigma_svg, Vsigma_svg)
    dthetadalpha_sg = (n_sg**2) * pref_isigma_sg
    theta_sg += dthetadalpha_sg * alpha_sg
    dthetadn_sg += 2.0 * n_sg * pref_isigma_sg * alpha_sg
    dthetadgradn_svg += (2.0 * dthetadalpha_sg[:, np.newaxis] * (2.0 * np.einsum(
        "sij..., sj...->si...", hessian_svvg, Vsigma_svg) - (3.0 * inverse_sigma_sg * alpha_sg)[:, np.newaxis] * gradn_svg))
    dthetadDsigma_svg += 2.0 * dthetadalpha_sg[:, np.newaxis] * Vsigma_svg
    # Hessian part
    dthetadhessian_svvg = 2.0 * \
        np.einsum("si...,sj...->sij...", dthetadDsigma_svg, gradn_svg)
    dthetadhessian_svvg += dthetadhessian_svvg.swapaxes(
        1, 2) - msvvg_diag(dthetadhessian_svvg)  # Eq. 23
    # This happens for small density regions
    theta_sg[theta_sg < 0.0] = 0.0
    return theta_sg, dthetadn_sg, dthetadgradn_svg, dthetadhessian_svvg


def process_theta_approx(gd, n_sg, sigma_xg, gradn_svg, hessian_svvg):
    nspins = len(n_sg)
    n_sg = n_sg.clip(min=0.0)
    sigma_sg = sigma_xg[::2].clip(min=MIN_SIGMA)
    inverse_sigma_sg = 1.0 / sigma_sg
    inverse_squared_sigma_sg = inverse_sigma_sg**2
    halfDsigma_svg = np.einsum("sij..., sj...->si...", hessian_svvg, gradn_svg)
    tmp_sg = (np.einsum("sv..., sv...->s...", gradn_svg, halfDsigma_svg)
              * inverse_squared_sigma_sg)  # dn.T H dn / sigma^2
    thetasqrt_sg = 2.0 * (1.0 - n_sg * tmp_sg)  # Eq. 66
    dthetadn_sg = gd.zeros(nspins)
    axpy(-4.0, tmp_sg * thetasqrt_sg, dthetadn_sg)
    dthetadgradn_svg = gd.zeros((nspins, 3))
    axpy(-2.0, (n_sg * inverse_sigma_sg * dthetadn_sg)
         [:, np.newaxis] * gradn_svg, dthetadgradn_svg)
    tmp_sg = n_sg * inverse_squared_sigma_sg * thetasqrt_sg
    axpy(-8.0, tmp_sg[:, np.newaxis] *
         halfDsigma_svg, dthetadgradn_svg)  # Eq. 70-2
    dthetadhessian_svvg = (-4.0 * tmp_sg[:, np.newaxis, np.newaxis]
                           * np.einsum("si...,sj...->sij...", gradn_svg, gradn_svg))
    dthetadhessian_svvg += dthetadhessian_svvg.swapaxes(
        1, 2) - msvvg_diag(dthetadhessian_svvg)
    return thetasqrt_sg**2, dthetadn_sg, dthetadgradn_svg, dthetadhessian_svvg


def process_theta(gd, n_sg, sigma_xg, gradn_svg, hessian_svvg, mode):
    if mode == "dsigma":
        return process_theta_dsigma(gd, n_sg, sigma_xg, gradn_svg, hessian_svvg)
    elif mode == "approx":
        return process_theta_approx(gd, n_sg, sigma_xg, gradn_svg, hessian_svvg)
    else:
        raise ValueError(f"Unknown theta mode: {mode}")


def process_switchf_cauchy(theta_sg, dthetadn_sg, dthetadgradn_svg, dthetadhessian_svvg, a=3.08):
    gt_sg = 1.0 / (1.0 + a * theta_sg**2)
    dg_dtheta_sg = -2.0 * a * theta_sg * (gt_sg**2)
    dthetadn_sg *= dg_dtheta_sg
    dthetadgradn_svg *= dg_dtheta_sg[:, np.newaxis]
    dthetadhessian_svvg *= dg_dtheta_sg[:, np.newaxis, np.newaxis]
    return gt_sg


def process_switchf_fermi(theta_sg, dthetadn_sg, dthetadgradn_svg, dthetadhessian_svvg, a=[0.28, 16.0]):
    x_sg = np.exp(-a[0] * (theta_sg - a[1]))
    gt_sg = x_sg / (1.0 + x_sg)
    dg_dtheta_sg = -a[0] * gt_sg * (1.0 - gt_sg)
    dthetadn_sg *= dg_dtheta_sg
    dthetadgradn_svg *= dg_dtheta_sg[:, np.newaxis]
    dthetadhessian_svvg *= dg_dtheta_sg[:, np.newaxis, np.newaxis]
    return gt_sg


def process_switchf(
    theta_sg, dthetadn_sg, dthetadgradn_svg, dthetadhessian_svvg, mode, a
):
    if debug:
        # assert np.all(np.sign(theta_sg) != -1), "Negative Theta!"
        if not np.all(np.sign(theta_sg) != -1):
            print("negative theta!")
            print(theta_sg[theta_sg < 0.0])
    if mode == "cauchy":
        switchf = process_switchf_cauchy
    elif mode == "fermi":
        switchf = process_switchf_fermi
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return switchf(theta_sg, dthetadn_sg, dthetadgradn_svg, dthetadhessian_svvg, a)


def add_radial_paw_correction(
    rgd, R_v, a_sg, dedgt_sg, dg_dgradn_svg, dg_dhessian_svvg, dedn_sg
):
    vv_g1 = rgd.zeros()
    vv_g2 = rgd.zeros()
    vv_gt = rgd.zeros()
    nspins = len(a_sg)
    dedhessian_svvg = dg_dhessian_svvg * dedgt_sg[:, np.newaxis, np.newaxis]
    for s in range(nspins):
        for v in range(3):
            # First order correction
            rgd.derivative2(
                rgd.dv_g * dedgt_sg[s] * dg_dgradn_svg[s, v] * R_v[v],
                vv_g1,
            )
            vv_gt -= vv_g1
            # Second order correction
            # diag
            rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, v, v], vv_g1)
            vv_g1[1:] /= rgd.r_g[1:]
            vv_g1[0] = vv_g1[1]
            radial_second_derivative(
                rgd, rgd.dv_g * dedhessian_svvg[s, v, v], vv_g2)
            vv_gt += (vv_g2 - vv_g1) * R_v[v] ** 2 + vv_g1
        # xy
        rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, 0, 1], vv_g1)
        vv_g1[1:] /= rgd.r_g[1:]
        vv_g1[0] = vv_g1[1]
        radial_second_derivative(
            rgd, rgd.dv_g * dedhessian_svvg[s, 0, 1], vv_g2)
        vv_gt += 2.0 * (vv_g2 - vv_g1) * R_v[0] * R_v[1]
        # xz
        rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, 0, 2], vv_g1)
        vv_g1[1:] /= rgd.r_g[1:]
        radial_second_derivative(
            rgd, rgd.dv_g * dedhessian_svvg[s, 0, 2], vv_g2)
        vv_gt += 2.0 * (vv_g2 - vv_g1) * R_v[0] * R_v[2]
        # yz
        rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, 1, 2], vv_g1)
        vv_g1[1:] /= rgd.r_g[1:]
        radial_second_derivative(
            rgd, rgd.dv_g * dedhessian_svvg[s, 1, 2], vv_g2)
        vv_gt += 2.0 * (vv_g2 - vv_g1) * R_v[1] * R_v[2]
        vv_gt[1:] /= rgd.dv_g[1:]
        vv_gt[0] = vv_gt[1]
        dedn_sg[s] += vv_gt


class ThetaMGGARadialKernel:
    def __init__(
        self, kernel, a=3.08, theta_mode="dsigma", switchf="cauchy", radial_only=False
    ):
        self.kernel = kernel
        self.theta_mode = theta_mode
        self.switchf = switchf
        self.a = a
        self.radial_only = radial_only

    def __call__(
        self, rgd, n_sLg, dndr_sLg, d2ndr2_sLg, R_v, Y_L, rnablaY_Lv, r2HessianYL_Lvv
    ):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg) = radial_gga_vars(
            rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv)
        nspins = len(n_sg)

        hessian_svvg = rgd.zeros((nspins, 3, 3))
        for s in range(nspins):
            hessian_svvg[s] = radialxYL_hessian_vvg(
                rgd, n_sLg[s], dndr_sLg[s], d2ndr2_sLg[s], Y_L, R_v, rnablaY_Lv, r2HessianYL_Lvv)
        gradn_svg = np.transpose(b_vsg, [1, 0, 2]).copy()
        gradn_svg[:, :, 1:] /= rgd.r_g[1:]
        gradn_svg[:, :, 0] = gradn_svg[:, :, 1]
        gradn_svg += np.einsum(a_sg, [0, 1], R_v, [2], [0, 2, 1])
        theta_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg = process_theta(
            rgd, n_sg, sigma_xg, gradn_svg, hessian_svvg, mode=self.theta_mode
        )
        gt_sg = process_switchf(
            theta_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg, mode=self.switchf, a=self.a)
        if debug:
            self.theta_sg = theta_sg
            self.gt_sg = gt_sg
        nspins = len(n_sg)
        dedn_sg = rgd.empty(nspins)
        dedgt_sg = rgd.empty(nspins)
        self.kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg,
                              dedsigma_xg, gt_sg, dedgt_sg)
        if self.radial_only:
            add_radial_paw_correction(
                rgd, R_v, a_sg, dedgt_sg, dg_dgradn_svg, dg_dhessian_svvg, dedn_sg)
        dedn_sg += dedgt_sg * dg_dn_sg + \
            gga_radial_gradient_correction(rgd, sigma_xg, dedsigma_xg, a_sg)
        return (e_g, dedn_sg, b_vsg, dedsigma_xg, dedgt_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg)


class ThetaMGGARadialExpansion:
    def __init__(self, rcalc, r2HessianYL_nLvv, correction_method=1, *args):
        self.rcalc = rcalc
        self.r2HessianYL_nLvv = r2HessianYL_nLvv
        if correction_method == 1:
            self.calculate_theta_correction = self.calculate_theta_correction1
        elif correction_method == 2:
            self.calculate_theta_correction = self.calculate_theta_correction2
        self.args = args

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg
        dndr_sLg = np.empty_like(n_sLg)
        d2ndr2_sLg = np.empty_like(n_sLg)
        for n_Lg, dndr_Lg, d2ndr2_Lg in zip(n_sLg, dndr_sLg, d2ndr2_sLg):
            for n_g, dndr_g, d2ndr2_g in zip(n_Lg, dndr_Lg, d2ndr2_Lg):
                rgd.derivative(n_g, dndr_g)
                radial_second_derivative(rgd, n_g, d2ndr2_g)
        nspins, Lmax, nq = D_sLq.shape
        dEdD_sqL = np.zeros((nspins, nq, Lmax))
        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]
            R_v = R_nv[n]
            rnablaY_Lv = rnablaY_nLv[n, :Lmax]
            r2HessianYL_Lvv = self.r2HessianYL_nLvv[n, :Lmax]
            (e_g, dedn_sg, b_vsg, dedsigma_xg, dedgt_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg,) = self.rcalc(
                rgd, n_sLg, dndr_sLg, d2ndr2_sLg, R_v, Y_L, rnablaY_Lv, r2HessianYL_Lvv)
            dEdD_sqL += self.calculate_theta_correction(
                rgd, n_qg, Y_L, w, R_v, rnablaY_Lv, r2HessianYL_Lvv, dedgt_sg, dg_dgradn_svg, dg_dhessian_svvg,)
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            dedsigma_xg *= rgd.dr_g
            B_vsg = dedsigma_xg[::2] * b_vsg
            if nspins == 2:
                B_vsg += 0.5 * dedsigma_xg[1] * b_vsg[:, ::-1]
            B_vsq = np.dot(B_vsg, n_qg.T)
            dEdD_sqL += 8 * pi * w * np.inner(rnablaY_Lv, B_vsq.T).T
            E += w * rgd.integrate(e_g)
        return E, dEdD_sqL

    def calculate_theta_correction2(
        self,
        rgd,
        n_qg,
        Y_L,
        w,
        R_v,
        rnablaY_Lv,
        r2HessianYL_Lvv,
        dedgt_sg,
        dg_dgradn_svg,
        dg_dhessian_svvg,
    ):
        """
        Slower but more memory efficient version of the theta correction
        """
        dndr_qg = np.empty_like(n_qg)
        d2ndr2_qg = np.empty_like(n_qg)
        for n_g, dndr_g, d2ndr2_g in zip(n_qg, dndr_qg, d2ndr2_qg):
            rgd.derivative(n_g, dndr_g)
            radial_second_derivative(rgd, n_g, d2ndr2_g)
        rgradn_qvg = radial_gradient_zvg(
            dndr_qg, R_v) * rgd.r_g[np.newaxis, np.newaxis]
        hessian_qvvg = radial_hessian_zvvg(d2ndr2_qg, dndr_qg, rgd.r_g, R_v)
        # First correciton
        dg_dgradn_svg *= (
            w * 4.0 * pi * dedgt_sg *
            rgd.r_g[np.newaxis] * rgd.dr_g[np.newaxis]
        )[:, np.newaxis]  # w * 4 *pi * r * (de / dVn_q) dr
        dEdD_sqL = np.einsum("qg,Lv,svg->sqL", n_qg, rnablaY_Lv, dg_dgradn_svg)
        dEdD_sqL += np.einsum(
            "qvg,L,svg->sqL",
            rgradn_qvg,
            Y_L,
            dg_dgradn_svg,
        )
        # Second correction
        dg_dhessian_svvg *= (4.0 * pi * w * dedgt_sg * rgd.dr_g[np.newaxis])[
            :, np.newaxis, np.newaxis
        ]  # 4 * pi * w * (de / dHn_q) * dr
        # x = r^2 * Vn_q * (VY_L)^T
        x = np.einsum("qig,Lj->qLijg", rgradn_qvg, rnablaY_Lv)
        # matrix reduction [refer to eq (1.70-1.72)]
        dEdD_sqL += np.einsum("qLijg, sijg->sqL", x +
                              x.swapaxes(2, 3), dg_dhessian_svvg)
        dEdD_sqL += np.einsum("qg,Lij,sijg->sqL", n_qg,
                              r2HessianYL_Lvv, dg_dhessian_svvg)
        dEdD_sqL += np.einsum("qijg,L,sijg->sqL", hessian_qvvg * (rgd.r_g**2)
                              [np.newaxis, np.newaxis, np.newaxis], Y_L, dg_dhessian_svvg)
        return dEdD_sqL

    def calculate_theta_correction1(self, rgd, n_qg, Y_L, w, R_v, rnablaY_Lv, r2HessianYL_Lvv, dedgt_sg, dg_dgradn_svg, dg_dhessian_svvg,):
        """
        Faster but more memory consuming version of the theta correction
        """
        dndr_qg = np.empty_like(n_qg)
        d2ndr2_qg = np.empty_like(n_qg)
        for n_g, dndr_g, d2ndr2_g in zip(n_qg, dndr_qg, d2ndr2_qg):
            rgd.derivative(n_g, dndr_g)
            radial_second_derivative(rgd, n_g, d2ndr2_g)
        gradn_qvg = radial_gradient_zvg(dndr_qg, R_v)
        hessian_qvvg = radial_hessian_zvvg(d2ndr2_qg, dndr_qg, rgd.r_g, R_v)
        # First correciton
        # indices for einsum
        v = 0
        q = 1
        L = 2
        g = 3
        s = 4
        vp = 5
        temp_vqLg = np.einsum(n_qg, [q, g], rnablaY_Lv, [L, v], [v, q, L, g])
        temp_vqLg += np.einsum(gradn_qvg * rgd.r_g[np.newaxis, np.newaxis], [
                               q, v, g], Y_L, [L], [v, q, L, g])
        temp_vqLg = np.einsum(
            temp_vqLg, [v, q, L, g], rgd.r_g, [g], [v, q, L, g])
        dg_dgradn_svg *= (dedgt_sg * rgd.dr_g[np.newaxis, :])[:, np.newaxis]
        dEdD_sqL = (w * 4.0 * pi * np.einsum(temp_vqLg,
                    [v, q, L, g], dg_dgradn_svg, [s, v, g], [s, q, L]))
        # Second correction
        gradn_qvg *= (rgd.r_g[np.newaxis, np.newaxis, :]
                      * rgd.dr_g[np.newaxis, np.newaxis, :] * 4.0 * pi)
        temp_qLvvg = np.einsum(gradn_qvg, [q, v, g], rnablaY_Lv, [
                               L, vp], [q, L, v, vp, g])
        temp_qLvvg += np.transpose(temp_qLvvg, [0, 1, 3, 2, 4])
        temp_qLvvg += np.einsum(4.0 * pi * n_qg * rgd.dr_g[np.newaxis, :], [
                                q, g], r2HessianYL_Lvv, [L, v, vp], [q, L, v, vp, g])
        temp_qLvvg += np.einsum(hessian_qvvg * rgd.dv_g[np.newaxis, np.newaxis, np.newaxis], [
                                q, v, vp, g], Y_L, [L], [q, L, v, vp, g])
        dg_dhessian_svvg *= dedgt_sg[:, np.newaxis, np.newaxis, :]
        dEdD_sqL += w * \
            np.einsum(temp_qLvvg, [q, L, v, vp, g],
                      dg_dhessian_svvg, [s, v, vp, g], [s, q, L])
        return dEdD_sqL


class ThetaMGGA(XCFunctional):
    orbital_dependent = False

    def __init__(self, kernel, stencil=2, stencil_second=3, hessian_mode="mix", a=3.08,
                 theta_mode="dsigma", switchf="cauchy", b=100.0, setup_name="PBE",):
        XCFunctional.__init__(self, kernel.name, kernel.type)
        self.kernel = kernel
        self.stencil = stencil
        self.stencil_second = stencil_second
        self.hessian_mode = hessian_mode
        self.theta_mode = theta_mode
        self.a = np.array(a, dtype=float)
        if switchf == "fermi" and self.a.size == 1:
            self.a = np.asarray([a, b], dtype=float)
        self.switchf = switchf
        self.radial_initialized = False
        self.setup_name = setup_name

    def todict(self):
        a = self.a[0]
        b = float("nan")
        if self.switchf == "fermi":
            b = self.a[1]
        return {"name": self.name, "switchf": self.switchf, "a": a, "b": b, "theta_mode": self.theta_mode, "hessian_mode": self.hessian_mode, "stencil": self.stencil, "stencil_second": self.stencil_second,
                "setup_name": self.setup_name,
                }

    def set_grid_descriptor(self, gd):
        XCFunctional.set_grid_descriptor(self, gd)
        self.grad_v = get_gradient_ops(gd, self.stencil, np)
        self.hessian_vv = get_hessian_ops(
            gd, self.stencil_second, mode=self.hessian_mode)

    def get_setup_name(self):
        return self.setup_name

    def get_description(self):
        return "{} with {} nearest neighbor stencil".format(self.name, self.stencil)

    def init_radial(self):
        self.r2HessianYL_nLvv = np.empty((*Y_nL.shape, 3, 3))
        for r2HessianYL_Lvv, Y_L, R_v, rnablaY_Lv in zip(
            self.r2HessianYL_nLvv, Y_nL, R_nv, rnablaY_nLv
        ):
            for l in range(5):
                Hrl_vv = hessianrl(l, R_v)
                for L in range(l**2, (l + 1) ** 2):
                    X_vv = np.outer(R_v, rnablaY_Lv[L])
                    r2HessianYL_Lvv[L] = (hessianrlYL(
                        L, R_v) - Hrl_vv * Y_L[L] - l * (X_vv + X_vv.T))
        self.radial_initialized = True

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        nspins = len(n_sg)
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)
        hessian_svvg = self.calculate_hessian(gd, n_sg)
        theta_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg = process_theta(
            gd, n_sg, sigma_xg, gradn_svg, hessian_svvg, mode=self.theta_mode
        )
        if debug:
            self.hessian_svvg = hessian_svvg.copy()
            self.gradn_svg = gradn_svg.copy()
        gt_sg = process_switchf(
            theta_sg,
            dg_dn_sg,
            dg_dgradn_svg,
            dg_dhessian_svvg,
            mode=self.switchf,
            a=self.a,
        )
        if debug:
            self.n_sg = n_sg.copy()
            self.sigma_xg = sigma_xg.copy()
            self.gt_sg = gt_sg.copy()
            self.theta_sg = theta_sg.copy()
        dedgt_sg = gd.zeros(nspins)
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg,
                              dedsigma_xg, gt_sg, dedgt_sg)
        self.calculate_functional_derivative(
            gradn_svg, sigma_xg, dedsigma_xg, dedgt_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg, v_sg)

    def calculate_functional_derivative(
        self,
        gradn_svg,
        sigma_xg,
        dedsigma_xg,
        dedgt_sg,
        dg_dn_sg,
        dg_dgradn_svg,
        dg_dhessian_svvg,
        v_sg,
    ):
        r"""Add gradient correction to potential.

        ::

                        __   /    de(r)   __        dg    de(r)    \
            v  (r) +=  -\/. |  2-------   \/ n(r) + ---- --------   |
            xc               \   dσ(r)              d\/n(r) dg(r) /

                     __(2)      de(r)      dg   de(r)
                   + \/.     ---------- + ---- --------
                                __(2)     dn(r) dg(r)
                               d\/{n(r)}
        """
        nspins = len(v_sg)
        # buffer
        vv_g = sigma_xg[0]
        dg_dhessian_svvg *= dedgt_sg[:, np.newaxis, np.newaxis]
        dedhessian_svvg = dg_dhessian_svvg
        for s in range(nspins):
            # derivative wrt denisty
            axpy(1.0, dedgt_sg[s] * dg_dn_sg[s], v_sg[s])
            for v in range(3):
                self.grad_v[v](
                    dedsigma_xg[2 * s] * gradn_svg[s, v],
                    vv_g,
                )
                axpy(-2.0, vv_g, v_sg[s])
                if nspins == 2:
                    self.grad_v[v](dedsigma_xg[1] * gradn_svg[s, v], vv_g)
                    axpy(-1.0, vv_g, v_sg[1 - s])
                # Theta + Second order correction
                self.grad_v[v](dedgt_sg[s] * dg_dgradn_svg[s, v], vv_g)
                axpy(-1.0, vv_g, v_sg[s])
                self.hessian_vv[v, v](dedhessian_svvg[s, v, v], vv_g)
                axpy(1.0, vv_g, v_sg[s])
            self.hessian_vv[0, 1](dedhessian_svvg[s, 0, 1], vv_g)
            axpy(2.0, vv_g, v_sg[s])
            self.hessian_vv[0, 2](dedhessian_svvg[s, 0, 2], vv_g)
            axpy(2.0, vv_g, v_sg[s])
            self.hessian_vv[1, 2](dedhessian_svvg[s, 1, 2], vv_g)
            axpy(2.0, vv_g, v_sg[s])

    def calculate_hessian(self, gd, n_sg):
        """
        Calculates the Hessian matrix of the density
        """
        nspins = n_sg.shape[0]
        hessian_svvg = gd.empty((nspins, 3, 3))
        for s in range(nspins):
            for v in range(3):
                self.hessian_vv[v, v](n_sg[s], hessian_svvg[s, v, v])
            self.hessian_vv[0, 1](n_sg[s], hessian_svvg[s, 0, 1])
            self.hessian_vv[0, 2](n_sg[s], hessian_svvg[s, 0, 2])
            self.hessian_vv[1, 2](n_sg[s], hessian_svvg[s, 1, 2])
            hessian_svvg[s, 1, 0] = hessian_svvg[s, 0, 1]
            hessian_svvg[s, 2, 0] = hessian_svvg[s, 0, 2]
            hessian_svvg[s, 2, 1] = hessian_svvg[s, 1, 2]
        return hessian_svvg

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        if not self.radial_initialized:
            self.init_radial()
        rcalc = ThetaMGGARadialKernel(
            self.kernel, a=self.a, theta_mode=self.theta_mode, switchf=self.switchf)
        expansion = ThetaMGGARadialExpansion(rcalc, self.r2HessianYL_nLvv)
        result = calculate_paw_correction(
            expansion, setup, D_sp, dEdD_sp, addcoredensity, a=a)
        return result

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        dndr_sg = np.empty_like(n_sg)
        d2ndr2_sg = np.empty_like(n_sg)
        for n_g, dndr_g, d2ndr2_g in zip(n_sg, dndr_sg, d2ndr2_sg):
            rgd.derivative(n_g, dndr_g)
            radial_second_derivative(rgd, n_g, d2ndr2_g)
        if e_g is None:
            e_g = rgd.empty()

        rcalc = ThetaMGGARadialKernel(
            self.kernel, a=self.a, theta_mode=self.theta_mode, switchf=self.switchf, radial_only=True)

        e_g[:], dedn_sg, *_ = rcalc(rgd, n_sg[:, np.newaxis], dndr_sg[:, np.newaxis], d2ndr2_sg[:, np.newaxis],
                                    np.ones(3, "d") / sqrt(3.0), [1.0], np.zeros((1, 3), "d"), np.zeros((1, 3, 3), "d"))
        v_sg[:] = dedn_sg
        if debug:
            self.rcalc = rcalc

        return rgd.integrate(e_g)

    def stress_tensor_contribution(self, n_sg, skip_sum=False):
        gd = self.gd
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)
        hessian_svvg = self.calculate_hessian(gd, n_sg)
        theta_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg = process_theta(
            gd, n_sg, sigma_xg, gradn_svg, hessian_svvg, mode=self.theta_mode
        )
        gt_sg = process_switchf(
            theta_sg, dg_dn_sg, dg_dgradn_svg, dg_dhessian_svvg, mode=self.switchf, a=self.a,)
        nspins = len(n_sg)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        dedgt_sg = gd.zeros(nspins)
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg,
                              dedsigma_xg, gt_sg, dedgt_sg)
        v_sg += dedgt_sg * dg_dn_sg

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        for v_g, n_g in zip(v_sg, n_sg):
            P -= integrate(v_g, n_g)
        for sigma_g, dedsigma_g in zip(sigma_xg, dedsigma_xg):
            P -= 2 * integrate(sigma_g, dedsigma_g)

        # switching function contributions to the gradient
        dedgradn_g_svg = dedgt_sg[:, np.newaxis] * dg_dgradn_svg
        dedhessian_svvg = dedgt_sg[:, np.newaxis,
                                   np.newaxis] * dg_dhessian_svvg

        stress_vv = P * np.eye(3)
        for v1 in range(3):
            for v2 in range(3):
                stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                               gradn_svg[0, v2],
                                               dedsigma_xg[0]) * 2
                # through the switching function
                stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                               dedgradn_g_svg[0, v2])
                if nspins == 2:
                    stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[1]) * 2
                    stress_vv[v1, v2] -= integrate(gradn_svg[1, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[2]) * 2
                    stress_vv[v1, v2] -= integrate(gradn_svg[1, v1] *
                                                   dedgradn_g_svg[1])
                # hessian dependence
                for v3 in range(3):
                    for s in range(len(hessian_svvg)):
                        stress_vv[v1, v2] += integrate(
                            hessian_svvg[s, v1, v3], dedhessian_svvg[0, v2, v3])

        if not skip_sum:
            self.gd.comm.sum(stress_vv)
        return stress_vv


# General Hessian-level Meta-GGA functionals


def process_gradient_projected_hessian(gradn_svg, hessian_svvg):
    """
               _    __    _
    Calculates Vn . VVn . Vn and the partial derivatives wrt the gradient and the Hessian matrix
    """
    Vsigma_svg = np.einsum("sij..., sj...->si...", hessian_svvg, gradn_svg)
    xi_sg = np.einsum("sv..., sv...->s...", gradn_svg, Vsigma_svg)
    dxidgradn_svg = 2 * Vsigma_svg
    dxidhessian_svvg = np.einsum("si..., sj...->sij...", gradn_svg, gradn_svg)
    dxidhessian_svvg += dxidhessian_svvg.swapaxes(
        1, 2) - msvvg_diag(dxidhessian_svvg)
    return xi_sg, dxidgradn_svg, dxidhessian_svvg


class HLMGGARadialKernel:
    def __init__(self, kernel, radial_only=False):
        self.kernel = kernel
        self.radial_only = radial_only

    def __call__(
        self, rgd, n_sLg, dndr_sLg, d2ndr2_sLg, R_v, Y_L, rnablaY_Lv, r2HessianYL_Lvv
    ):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg) = radial_gga_vars(
            rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv
        )
        nspins = len(n_sg)
        # TODO: only work with the upper triangle of the Hessian
        hessian_svvg = rgd.zeros((nspins, 3, 3))
        for s in range(nspins):
            hessian_svvg[s] = radialxYL_hessian_vvg(
                rgd, n_sLg[s], dndr_sLg[s], d2ndr2_sLg[s], Y_L, R_v, rnablaY_Lv, r2HessianYL_Lvv,)
        gradn_svg = np.transpose(b_vsg, [1, 0, 2]).copy()
        gradn_svg[:, :, 1:] /= rgd.r_g[1:]
        gradn_svg[:, :, 0] = gradn_svg[:, :, 1]
        gradn_svg += np.einsum(a_sg, [0, 1], R_v, [2], [0, 2, 1])
        xi_sg, dxidgradn_svg, dxidhessian_svvg = process_gradient_projected_hessian(
            gradn_svg, hessian_svvg)
        dedn_sg = rgd.empty(nspins)
        dedxi_sg = rgd.empty(nspins)
        self.kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg,
                              dedsigma_xg, xi_sg, dedxi_sg)
        if self.radial_only:
            self.add_radial_paw_correction(
                rgd, R_v, a_sg, dedxi_sg, dxidgradn_svg, dxidhessian_svvg, dedn_sg)
        dedn_sg += gga_radial_gradient_correction(
            rgd, sigma_xg, dedsigma_xg, a_sg)
        return (e_g, dedn_sg, b_vsg, dedsigma_xg, dedxi_sg, dxidgradn_svg, dxidhessian_svvg,)

    def add_radial_paw_correction(
        self, rgd, R_v, a_sg, dedxi_sg, dxidgradn_svg, dxidhessian_svvg, dedn_sg
    ):
        vv_g1 = rgd.zeros()
        vv_g2 = rgd.zeros()
        vv_gt = rgd.zeros()
        nspins = len(a_sg)
        dedhessian_svvg = dxidhessian_svvg * \
            dedxi_sg[:, np.newaxis, np.newaxis]
        for s in range(nspins):
            for v in range(3):
                # First order correction
                rgd.derivative2(
                    rgd.dv_g * dedxi_sg[s] * dxidgradn_svg[s, v] * R_v[v], vv_g1,)
                vv_gt -= vv_g1
                # Second order correction
                # diag
                rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, v, v], vv_g1)
                vv_g1[1:] /= rgd.r_g[1:]
                vv_g1[0] = vv_g1[1]
                radial_second_derivative(
                    rgd, rgd.dv_g * dedhessian_svvg[s, v, v], vv_g2)
                vv_gt += (vv_g2 - vv_g1) * R_v[v] ** 2 + vv_g1
            # xy
            rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, 0, 1], vv_g1)
            vv_g1[1:] /= rgd.r_g[1:]
            vv_g1[0] = vv_g1[1]
            radial_second_derivative(
                rgd, rgd.dv_g * dedhessian_svvg[s, 0, 1], vv_g2)
            vv_gt += 2.0 * (vv_g2 - vv_g1) * R_v[0] * R_v[1]
            # xz
            rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, 0, 2], vv_g1)
            vv_g1[1:] /= rgd.r_g[1:]
            radial_second_derivative(
                rgd, rgd.dv_g * dedhessian_svvg[s, 0, 2], vv_g2)
            vv_gt += 2.0 * (vv_g2 - vv_g1) * R_v[0] * R_v[2]
            # yz
            rgd.derivative2(rgd.dv_g * dedhessian_svvg[s, 1, 2], vv_g1)
            vv_g1[1:] /= rgd.r_g[1:]
            radial_second_derivative(
                rgd, rgd.dv_g * dedhessian_svvg[s, 1, 2], vv_g2)
            vv_gt += 2.0 * (vv_g2 - vv_g1) * R_v[1] * R_v[2]
            vv_gt[1:] /= rgd.dv_g[1:]
            vv_gt[0] = vv_gt[1]
            dedn_sg[s] += vv_gt


class HLMGGARadialExpansion:
    def __init__(self, rcalc, r2HessianYL_nLvv, correction_method=1, *args):
        self.rcalc = rcalc
        self.r2HessianYL_nLvv = r2HessianYL_nLvv

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg
        dndr_sLg = np.empty_like(n_sLg)
        d2ndr2_sLg = np.empty_like(n_sLg)
        for n_Lg, dndr_Lg, d2ndr2_Lg in zip(n_sLg, dndr_sLg, d2ndr2_sLg):
            for n_g, dndr_g, d2ndr2_g in zip(n_Lg, dndr_Lg, d2ndr2_Lg):
                rgd.derivative(n_g, dndr_g)
                radial_second_derivative(rgd, n_g, d2ndr2_g)
        nspins, Lmax, nq = D_sLq.shape
        dEdD_sqL = np.zeros((nspins, nq, Lmax))
        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]
            R_v = R_nv[n]
            rnablaY_Lv = rnablaY_nLv[n, :Lmax]
            r2HessianYL_Lvv = self.r2HessianYL_nLvv[n, :Lmax]
            (e_g, dedn_sg, b_vsg, dedsigma_xg, dedxi_sg, dxidgradn_svg, dxidhessian_svvg,) = self.rcalc(
                rgd, n_sLg, dndr_sLg, d2ndr2_sLg, R_v, Y_L, rnablaY_Lv, r2HessianYL_Lvv,)
            dEdD_sqL += self.calculate_hessian_correction(
                rgd, n_qg, Y_L, w, R_v, rnablaY_Lv, r2HessianYL_Lvv, dedxi_sg, dxidgradn_svg, dxidhessian_svvg,)
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            dedsigma_xg *= rgd.dr_g
            B_vsg = dedsigma_xg[::2] * b_vsg
            if nspins == 2:
                B_vsg += 0.5 * dedsigma_xg[1] * b_vsg[:, ::-1]
            B_vsq = np.dot(B_vsg, n_qg.T)
            dEdD_sqL += 8 * pi * w * np.inner(rnablaY_Lv, B_vsq.T).T
            E += w * rgd.integrate(e_g)
        return E, dEdD_sqL

    def calculate_hessian_correction(self, rgd, n_qg, Y_L, w, R_v, rnablaY_Lv, r2HessianYL_Lvv,
                                     dedxi_sg, dxidgradn_svg, dxidhessian_svvg,):
        dndr_qg = np.empty_like(n_qg)
        d2ndr2_qg = np.empty_like(n_qg)
        for n_g, dndr_g, d2ndr2_g in zip(n_qg, dndr_qg, d2ndr2_qg):
            rgd.derivative(n_g, dndr_g)
            radial_second_derivative(rgd, n_g, d2ndr2_g)
        rgradn_qvg = radial_gradient_zvg(
            dndr_qg, R_v) * rgd.r_g[np.newaxis, np.newaxis]
        hessian_qvvg = radial_hessian_zvvg(d2ndr2_qg, dndr_qg, rgd.r_g, R_v)
        # First correciton
        dxidgradn_svg *= (
            w * 4.0 * pi * dedxi_sg *
            rgd.r_g[np.newaxis] * rgd.dr_g[np.newaxis]
        )[:, np.newaxis]  # w * 4 *pi * r * (de / dVn_q) dr => dedgradn_svg
        dEdD_sqL = np.einsum("qg,Lv,svg->sqL", n_qg, rnablaY_Lv, dxidgradn_svg)
        dEdD_sqL += np.einsum("qvg,L,svg->sqL",
                              rgradn_qvg, Y_L, dxidgradn_svg,)
        # Second correction
        dxidhessian_svvg *= (4.0 * pi * w * dedxi_sg * rgd.dr_g[np.newaxis])[
            :, np.newaxis, np.newaxis]  # 4 * pi * w * (de / dHn_q) * dr
        # tmp = r^2 * Vn_q * (VY_L)^T
        tmp_qLvvg = np.einsum("qig,Lj->qLijg", rgradn_qvg, rnablaY_Lv)
        dEdD_sqL += np.einsum("qLijg, sijg->sqL", tmp_qLvvg +
                              tmp_qLvvg.swapaxes(2, 3), dxidhessian_svvg)
        dEdD_sqL += np.einsum("qg,Lij,sijg->sqL", n_qg,
                              r2HessianYL_Lvv, dxidhessian_svvg)
        dEdD_sqL += np.einsum("qijg,L,sijg->sqL", hessian_qvvg * (rgd.r_g**2)
                              [np.newaxis, np.newaxis, np.newaxis], Y_L, dxidhessian_svvg,)
        return dEdD_sqL


class HLMGGA(XCFunctional):
    orbital_dependent = False

    def __init__(self, kernel, stencil=2, stencil_second=None, hessian_mode="mix", setup_name='PBE'):
        XCFunctional.__init__(self, kernel.name, kernel.type)
        self.kernel = kernel
        self.stencil = stencil
        if stencil_second is None:
            self.stencil_second = stencil
        else:
            self.stencil_second = stencil_second
        self.hessian_mode = hessian_mode
        self.setup_name = setup_name
        self.init_radial()

    def todict(self):
        return {"name": self.name, "stencil": self.stencil, "stencil_second": self.stencil_second,
                "hessian_mode": self.hessian_mode, "setup_name": self.setup_name}

    def set_grid_descriptor(self, gd):
        XCFunctional.set_grid_descriptor(self, gd)
        self.grad_v = get_gradient_ops(gd, self.stencil, np)
        self.hessian_vv = get_hessian_ops(
            gd, self.stencil_second, mode=self.hessian_mode
        )

    def get_setup_name(self):
        return self.setup_name

    def get_description(self):
        return "{} with {} nearest neighbor stencil and {} nearest neighbors for the second derivatives".format(
            self.name, self.stencil, self.stencil_second
        )

    def init_radial(self):
        self.r2HessianYL_nLvv = np.empty((*Y_nL.shape, 3, 3))
        for r2HessianYL_Lvv, Y_L, R_v, rnablaY_Lv in zip(
            self.r2HessianYL_nLvv, Y_nL, R_nv, rnablaY_nLv
        ):
            for l in range(5):
                Hrl_vv = hessianrl(l, R_v)
                for L in range(l**2, (l + 1) ** 2):
                    X_vv = np.outer(R_v, rnablaY_Lv[L])
                    r2HessianYL_Lvv[L] = (
                        hessianrlYL(L, R_v) - Hrl_vv *
                        Y_L[L] - l * (X_vv + X_vv.T)
                    )

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        nspins = len(n_sg)
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)
        hessian_svvg = self.calculate_hessian(gd, n_sg)
        # TODO: only work with the upper triangle of the hessian matrix
        xi_sg, dxidgradn_svg, dxidhessian_svvg = process_gradient_projected_hessian(
            gradn_svg, hessian_svvg
        )
        self.xi_sg = xi_sg.copy()
        self.sigma_xg = sigma_xg.copy()
        dedxi_sg = gd.zeros(nspins)
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg,
                              dedsigma_xg, xi_sg, dedxi_sg)

        self.calculate_functional_derivative(
            gradn_svg,
            sigma_xg,
            dedsigma_xg,
            dedxi_sg,
            dxidgradn_svg,
            dxidhessian_svvg,
            v_sg,
        )

    def calculate_functional_derivative(
        self,
        gradn_svg,
        sigma_xg,
        dedsigma_xg,
        dedxi_sg,
        dxidgradn_svg,
        dxidhessian_svvg,
        v_sg,
    ):
        r"""Add gradient correction to potential.

        ::

                        __   /    de(r)   __        dxi    de(r)    \
            v  (r) +=  -\/. |  2-------   \/ n(r) + ---- --------   |
            xc               \  dsigma(r)           d\/n(r) dg(r) /

                     __(2)      de(r)
                   + \/.     -----------
                                __(2)
                               d\/{n(r)}
        """
        nspins = len(v_sg)
        # buffer
        vv_g = sigma_xg[0]
        dedgradn_svg = dxidgradn_svg
        dedgradn_svg *= dedxi_sg[:, None]
        dedhessian_svvg = dxidhessian_svvg
        dedhessian_svvg *= dedxi_sg[:, None, None]
        for s in range(nspins):
            for v in range(3):
                self.grad_v[v](dedsigma_xg[2 * s] * gradn_svg[s, v], vv_g)
                axpy(-2.0, vv_g, v_sg[s])
                if nspins == 2:
                    self.grad_v[v](dedsigma_xg[1] * gradn_svg[s, v], vv_g)
                    axpy(-1.0, vv_g, v_sg[1 - s])
                # Theta + Second order correction
                self.grad_v[v](dedgradn_svg[s, v], vv_g)
                axpy(-1.0, vv_g, v_sg[s])
                self.hessian_vv[v, v](dedhessian_svvg[s, v, v], vv_g)
                axpy(1.0, vv_g, v_sg[s])
            self.hessian_vv[0, 1](dedhessian_svvg[s, 0, 1], vv_g)
            axpy(2.0, vv_g, v_sg[s])
            self.hessian_vv[0, 2](dedhessian_svvg[s, 0, 2], vv_g)
            axpy(2.0, vv_g, v_sg[s])
            self.hessian_vv[1, 2](dedhessian_svvg[s, 1, 2], vv_g)
            axpy(2.0, vv_g, v_sg[s])

    def calculate_hessian(self, gd, n_sg):
        """
        Calculates the Hessian matrix of the density
        """
        # TODO: work with the upper triangle of the Hessian matrix
        nspins = n_sg.shape[0]
        hessian_svvg = gd.empty((nspins, 3, 3))
        for s in range(nspins):
            for v in range(3):
                self.hessian_vv[v, v](n_sg[s], hessian_svvg[s, v, v])
            self.hessian_vv[0, 1](n_sg[s], hessian_svvg[s, 0, 1])
            self.hessian_vv[0, 2](n_sg[s], hessian_svvg[s, 0, 2])
            self.hessian_vv[1, 2](n_sg[s], hessian_svvg[s, 1, 2])
            hessian_svvg[s, 1, 0] = hessian_svvg[s, 0, 1]
            hessian_svvg[s, 2, 0] = hessian_svvg[s, 0, 2]
            hessian_svvg[s, 2, 1] = hessian_svvg[s, 1, 2]
        return hessian_svvg

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        rcalc = HLMGGARadialKernel(self.kernel)
        expansion = HLMGGARadialExpansion(rcalc, self.r2HessianYL_nLvv)
        return calculate_paw_correction(
            expansion, setup, D_sp, dEdD_sp, addcoredensity, a=a
        )

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        dndr_sg = np.empty_like(n_sg)
        d2ndr2_sg = np.empty_like(n_sg)
        for n_g, dndr_g, d2ndr2_g in zip(n_sg, dndr_sg, d2ndr2_sg):
            rgd.derivative(n_g, dndr_g)
            radial_second_derivative(rgd, n_g, d2ndr2_g)
        if e_g is None:
            e_g = rgd.empty()

        rcalc = HLMGGARadialKernel(self.kernel, radial_only=True)

        e_g[:], dedn_sg, *_ = rcalc(
            rgd,
            n_sg[:, np.newaxis],
            dndr_sg[:, np.newaxis],
            d2ndr2_sg[:, np.newaxis],
            np.ones(3, "d") / sqrt(3.0),
            [1.0],
            np.zeros((1, 3), "d"),
            np.zeros((1, 3, 3), "d"),
        )
        v_sg[:] = dedn_sg

        return rgd.integrate(e_g)

    def stress_tensor_contribution(self, n_sg, skip_sum=False):
        raise NotImplementedError(
            "Stress tensor contribution not implemented HLMGGA"
        )  # TODO
