from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from gpaw.utilities import unpack_hermitian
from gpaw.iks.tools import calculate_diff_overlap_matrix
if TYPE_CHECKING:
    from gpaw.iks import iKS
    from gpaw.old.kpoint import KPoint
    from gpaw.iks.tools import DegeneracyDescriptor


class iKSLinearSolver:

    def __init__(self, iks: iKS, dd: DegeneracyDescriptor, method: str, ortho: bool = False, **ls_kwargs):
        assert method == 'minres' or method is None or method.lower() == "none"
        self.iks = iks
        self.dd = dd
        self.method = method
        self.ortho = ortho
        self.ls_kwargs = ls_kwargs
        self.nocc = len(dd.occ)
        self.kpt = dd.kpt
        self.p_nG = None
        self.Pp_ani = None

    def run(self, p0_nG) -> None:
        self.p_nG = self.calculate_p(p0_nG)
        self.Pp_ani = self.calculate_projections(self.kpt, self.p_nG)

    def calculate_projections(self, kpt, p_nG):
        wfs = self.iks.wfs
        P_ani = wfs.pt.dict((self.nocc,))
        wfs.pt.integrate(p_nG, P_ani, kpt.q)
        return P_ani

    def apply_occupation_correction(self, dfde_n, eta):
        for bm in self.dd.blocks_occ:
            for m in bm:
                for n in bm:
                    coeff = self.dd.vdiff_nn[n, m]
                    if n == m:
                        coeff += eta
                    self.p_nG[m] += 0.5 * coeff * \
                        dfde_n[n] * self.kpt.psit_nG[n]
        # recalculate the projections
        self.Pp_ani = self.calculate_projections(self.kpt, self.p_nG)

    def orthogonalize(self, kpt, p_nG, psit_nG):
        wfs = self.iks.wfs
        Pp_ani = self.calculate_projections(kpt, p_nG)
        P_ani = kpt.P_ani
        p2_nG = p_nG.copy()
        for n in range(len(p2_nG)):
            II = np.zeros(1, dtype=p2_nG.dtype)
            for a, Pp_ni in Pp_ani.items():
                II += np.einsum("i,ij,j", Pp_ni[n].conj(),
                                wfs.setups[a].dO_ii, P_ani[a][n])
            self.iks.gd.comm.sum(II)
            alpha = II + wfs.integrate(p_nG[n], psit_nG[n])
            p2_nG[n] -= alpha * psit_nG[n]
        return p2_nG

        wfs = self.iks.wfs
        P_ani = wfs.pt.dict((self.nocc,))
        wfs.pt.integrate(p_nG, P_ani, kpt.q)
        return P_ani

    def apply_lhs(self, p_nG):
        iks = self.iks
        Hp_nG = np.zeros_like(p_nG)
        wfs = iks.wfs
        kpt = self.kpt
        ham = iks.hamiltonian
        wfs.apply_pseudo_hamiltonian(kpt, ham, p_nG, Hp_nG)
        P_ani = wfs.pt.dict((self.nocc,))
        wfs.pt.integrate(p_nG, P_ani, kpt.q)
        c_ani = wfs.pt.dict((self.nocc,))
        for a, P_ni in P_ani.items():
            dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
            c_ani[a] = np.dot(P_ni, dH_ii)
        wfs.pt.add(Hp_nG, c_ani, kpt.q)
        Sp_nG = p_nG.copy()
        for n, eps in enumerate(kpt.eps_n[self.dd.occ]):
            Sp_nG[n] *= eps

        for a, P_ni in P_ani.items():
            c_ani[a] = kpt.eps_n[self.dd.occ, np.newaxis] * \
                np.dot(wfs.setups[a].dO_ii, P_ni.T).T
        wfs.pt.add(Sp_nG, c_ani, q=kpt.q)
        assert np.isfinite(Hp_nG).all() and np.isfinite(Sp_nG).all()
        return Hp_nG - Sp_nG

    def calculate_rhs(self):
        iks = self.iks
        wfs = iks.wfs
        dd = self.dd
        kpt = dd.kpt
        vdiff_G = 2.0 * iks.weights_sG[kpt.s] * iks.dn_sG[kpt.s]
        psit_nG = kpt.psit_nG[dd.occ]
        if wfs.mode == "pw":
            psit_nG = np.asarray([wfs.pd.ifft(psit_G, q=kpt.q)
                                 for psit_G in psit_nG])
        b_nG = np.zeros_like(psit_nG)
        for f, psit_G, b_G in zip(kpt.f_n[dd.occ], psit_nG, b_nG):
            b_G[:] = f * vdiff_G * psit_G
        if wfs.mode == "pw":
            b_nR = b_nG.copy()
            b_nG = wfs.empty(self.nocc, q=kpt.q)
            for b_G, b_R in zip(b_nG, b_nR):
                b_G[:] = wfs.pd.fft(b_R, q=kpt.q)

        for bm in self.dd.blocks_occ:
            for m in bm:
                for n in bm:
                    b_nG[m] -= kpt.f_n[m] * dd.vdiff_nn[n, m] * kpt.psit_nG[n]

        self.apply_rhs_paw_correction(b_nG)
        return b_nG

    def apply_rhs_paw_correction(self, b_nG):
        pt = self.iks.wfs.pt
        kpt = self.kpt
        iks = self.iks
        dp_ani = pt.dict((self.nocc,))
        for a, dp_ni in dp_ani.items():
            w_ng = iks.weights_asng[a][kpt.s]
            if "radial_" in iks.atomic_weights:
                w_ng = w_ng[np.newaxis, ...]
            Phi_ii = calculate_diff_overlap_matrix(
                iks.setups[a],
                w_ng,
                iks.atomic_weights,
                iks.dD_asp[a][kpt.s],
                iks.mix_paw_error,
            )
            dp_ni[:] = np.einsum(
                'n,ij,nj->ni', 2.0 * kpt.f_n[self.dd.occ], Phi_ii, kpt.P_ani[a][self.dd.occ])
            for bm in self.dd.blocks_occ:
                for m in bm:
                    for n in bm:
                        dp_ni[m] -= kpt.f_n[m] * self.dd.vdiff_nn[n, m] * \
                            iks.wfs.setups[a].dO_ii @ kpt.P_ani[a][n]
        pt.add(b_nG, dp_ani, q=kpt.q)

    def calculate_p(self, p0_nG):
        if self.method is None or self.method.lower() == "none":
            self.p_nG = p0_nG
            return self.p_nG

        wfs = self.iks.wfs
        kpt = self.kpt

        rhs = self.calculate_rhs()
        shape = p0_nG.shape

        def integrate(a, b):
            a = a.reshape(shape)
            b = b.reshape(shape)
            II = 0.0
            for n in range(len(p0_nG)):
                II += wfs.integrate(a[n], b[n])
            return II

        def apply(a):
            return self.apply_lhs(a.reshape(shape)).ravel()

        p_nG, iters = minres_iks(
            apply,
            rhs.ravel(),
            p0_nG.ravel(),
            integrate,
            None,
            **self.ls_kwargs,
        )
        p_nG = p_nG.reshape(shape)
        self.iks.fmt.log(f"minres finished in {iters} iterations.")
        if self.ortho:
            p_nG = self.orthogonalize(kpt, p_nG, kpt.psit_nG[self.dd.occ])
        return p_nG

    def calculate_jacobian_one_k(self):
        p_nG = self.p_nG
        if self.iks.wfs.mode == "pw":
            p_nG = np.asarray(
                [self.iks.wfs.pd.ifft(p_G, q=self.kpt.q) for p_G in self.p_nG]
            )
        return 2.0 * np.sum((p_nG.conj() * self.iks.get_realspace_psit(self.kpt)[self.dd.occ]).real, axis=0)


def minres_iks(apply_lhs, rhs, x0, innerp, prec=None, rtol=1e-8, maxiter=1000):
    """copy of scipy.sparse.linalg.minres to work with custom inner product and complex values
    https://github.com/scipy/scipy/blob/25eb651a190cca323fde466b4aa47475ef1024ef/scipy/sparse/linalg/_isolve/minres.py
    """
    if not prec:

        def prec(x):
            return x

    istop = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0
    b = rhs
    if x0 is None:
        r1 = b.copy()
        x = np.zeros_like(b)
    else:
        r1 = rhs - apply_lhs(x0)
        x = x0.copy()
    xtype = x.dtype
    eps = np.finfo(xtype).eps
    y = prec(r1)
    beta1 = np.real(innerp(r1, y))
    if beta1 < 0:
        raise ValueError("indefinite preconditioner")
    elif beta1 == 0:
        return (x, 0)

    bnorm = innerp(b, b).real
    if bnorm == 0:
        x = b
        return (x, 0)

    beta1 = np.sqrt(beta1)
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = np.finfo(xtype).max
    cs = -1
    sn = 0
    w = np.zeros_like(x)
    w2 = np.zeros_like(x)
    r2 = r1
    itn = 0

    while itn < maxiter:
        itn += 1

        s = 1.0 / beta
        v = s * y

        y = apply_lhs(v)
        shift = 0
        y = y - shift * v

        if itn >= 2:
            y = y - (beta / oldb) * r1

        alfa = innerp(v, y)
        y = y - (alfa / beta) * r2
        r1 = r2
        r2 = y
        y = prec(r2)
        oldb = beta
        beta = np.real(innerp(r2, y))
        if beta < 0:
            raise ValueError("non-symmetric matrix")
        beta = np.sqrt(beta)
        tnorm2 += alfa**2 + oldb**2 + beta**2

        if itn == 1:
            if beta / beta1 <= 10 * eps:
                istop = -1  # Terminate later

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta = cs * dbar + sn * alfa  # delta1 = 0         deltak
        gbar = sn * dbar - cs * alfa  # gbar 1 = alfa1     gbar k
        epsln = sn * beta  # epsln2 = 0         epslnk+1
        dbar = -cs * beta  # dbar 2 = beta2     dbar k+1
        root = np.linalg.norm([gbar, dbar])
        # Arnorm = phibar * root

        # Compute the next plane rotation Qk

        gamma = np.linalg.norm([gbar, beta])  # gammak
        gamma = max(gamma, eps)
        cs = gbar / gamma  # ck
        sn = beta / gamma  # sk
        phi = cs * phibar  # phik
        phibar = sn * phibar  # phibark+1

        # Update  x.

        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x = x + phi * w
        # Go round again.
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z
        # Estimate various norms and test for convergence.
        Anorm = np.sqrt(tnorm2.real)
        ynorm = np.sqrt(innerp(x, x))
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        # epsr = Anorm * ynorm * rtol
        diag = gbar
        if diag == 0:
            diag = epsa
        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = np.inf
        else:
            test1 = rnorm / (Anorm * ynorm)  # ||r||  / (||A|| ||x||)
        if Anorm == 0:
            test2 = np.inf
        else:
            test2 = root / Anorm  # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q @ H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        Acond = gmax / gmin
        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above (Abar = const*I).

        if istop == 0:
            t1 = 1 + test1  # These tests work if rtol < eps
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1

            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            # if rnorm <= epsx   : istop = 2
            # if rnorm <= epsr   : istop = 1
            if test2 <= rtol:
                istop = 2
            if test1 <= rtol:
                istop = 1
        if istop != 0:
            break
    return (x, itn)
