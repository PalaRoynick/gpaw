from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from gpaw.utilities import unpack_hermitian
from gpaw.iks.tools import calculate_diff_overlap_matrix
from scipy.sparse.linalg import (
    LinearOperator, bicg, bicgstab, cg, cgs, gcrotmk, gmres, lgmres, minres, qmr, tfqmr,)

scipy_linear_solvers = {
    "minres-scipy": minres,
    "cg": cg,
    "gmres": gmres,
    "lgmres": lgmres,
    "bicg": bicg,
    "bicgstab": bicgstab,
    "cgs": cgs,
    "qmr": qmr,
    "tfqmr": tfqmr,
    "gcrotmk": gcrotmk,
}
if TYPE_CHECKING:
    from gpaw.iks import iKS
    from gpaw.old.kpoint import KPoint
    from gpaw.typing import Array1D


class iKSLinearSolver:
    mwork = 1
    mstop = -1

    def __init__(self, iks: iKS, method: str, ortho: bool = False, **ls_kwargs):
        self.iks = iks
        self.msg_stop = np.array(self.mstop, dtype=int)
        self.msg_work = np.array(self.mwork, dtype=int)
        self.comm = iks.gd.comm
        self.method = method
        self.ortho = ortho
        self.ls_kwargs = ls_kwargs
        self.reset()

    def run(self, kpt, p0_nG, u_m) -> None:
        self.reset()
        self.update_kpt(kpt)
        self.p_nG = self.calculate_p(kpt, p0_nG, u_m, **self.ls_kwargs)
        self.Pp_ani = None  # in the apply correction or externally
        self.u_m = u_m

    def apply_occupation_correction(self, dfde_m, eta):
        alpha_m = -0.5 * dfde_m * (self.u_m / self.f_n - eta)
        self.p_nG[:] += np.einsum('m,m...->m...', alpha_m, self.psit_nG)
        self.Pp_ani = self.calculate_projections(self.kpt, self.p_nG)

    def reset(self):
        self.kpt: KPoint | None = None
        self.p_nG = None
        self.Pp_ani = None
        self.eps_n = None
        self.f_n = None
        self.nG = None
        self.psit_nG = None
        self.dp_ani = None

    def calculate_projections(self, kpt, p_nG):
        wfs = self.iks.wfs
        P_ani = wfs.pt.dict((self.nG[0],))
        wfs.pt.integrate(p_nG, P_ani, kpt.q)
        return P_ani

    def apply_lhs_one_band(self, n, p_G):
        iks = self.iks
        p_xG = p_G[None]
        Hp_xG = np.zeros_like(p_xG)
        wfs = iks.wfs
        kpt = self.kpt
        ham = iks.hamiltonian
        wfs.apply_pseudo_hamiltonian(kpt, ham, p_xG, Hp_xG)
        P_axi = wfs.pt.dict((1,))
        wfs.pt.integrate(p_xG, P_axi, kpt.q)
        c_axi = wfs.pt.dict((1,))
        for a, P_xi in P_axi.items():
            dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
            c_axi[a] = np.dot(P_xi, dH_ii)
        wfs.pt.add(Hp_xG, c_axi, kpt.q)
        eps = self.eps_n[n]
        Sp_xG = p_xG * eps
        for a, P_xi in P_axi.items():
            c_axi[a][:] = eps * np.dot(P_xi[0], wfs.setups[a].dO_ii)
        wfs.pt.add(Sp_xG, c_axi, q=kpt.q)
        assert np.isfinite(Hp_xG).all() and np.isfinite(Sp_xG).all()
        lhs_G = (Hp_xG - Sp_xG)[0]
        return lhs_G

    def apply_lhs(self, p_nG):
        iks = self.iks
        Hp_nG = np.zeros_like(p_nG)
        wfs = iks.wfs
        kpt = self.kpt
        ham = iks.hamiltonian
        wfs.apply_pseudo_hamiltonian(kpt, ham, p_nG, Hp_nG)
        P_ani = wfs.pt.dict((self.nG[0],))
        wfs.pt.integrate(p_nG, P_ani, kpt.q)
        c_ani = wfs.pt.dict((self.nG[0],))
        for a, P_ni in P_ani.items():
            dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
            c_ani[a] = np.dot(P_ni, dH_ii)
        wfs.pt.add(Hp_nG, c_ani, kpt.q)
        Sp_nG = p_nG.copy()
        for n, eps in enumerate(self.eps_n):
            Sp_nG[n] *= eps

        for a, P_ni in P_ani.items():
            c_ani[a] = self.eps_n[:, np.newaxis] * \
                np.dot(wfs.setups[a].dO_ii, P_ni.T).T
        wfs.pt.add(Sp_nG, c_ani, q=kpt.q)
        assert np.isfinite(Hp_nG).all() and np.isfinite(Sp_nG).all()
        return Hp_nG - Sp_nG

    def apply_lhs_scipy(self, param: Array1D | None = None):
        data_nG = self.iks.wfs.empty(
            self.nG[0], q=self.kpt.q, global_array=True)
        if self.comm.rank == 0:
            data_nG[:] = param.reshape(self.nG)
            self.comm.broadcast(self.msg_work, 0)
        p_nG = self.dist_nG(data_nG)
        lhs_nG = self.apply_lhs(p_nG)
        lhs_nG = self.collect_nG(lhs_nG)
        return np.ravel(lhs_nG)

    # def Mop(self, param: Array1D):
    #     # TODO
    #     # prec = self.wfs.make_preconditioner()
    #     # R_nG = v.reshape(nG)
    #     # e_kin = prec.calculate_kinetic_energy(currentp_nG, kpt)
    #     # pR_nG = np.zeros_like(R_nG)
    #     # for R_G, pR_G, ek in zip(R_nG, pR_nG, e_kin):
    #     #     pR_G[:] = prec(R_G, kpt, ek)
    #     # return pR_nG.ravel()
    #     return None

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
            self.iks.hamiltonian.world.sum(II)
            alpha = II + wfs.integrate(p_nG[n], psit_nG[n])
            p2_nG[n] -= alpha * psit_nG[n]
        return p2_nG

    def collect_nG(self, array_nG):
        wfs = self.iks.wfs
        if wfs.mode == "pw":
            b_nG = wfs.empty(self.nG[0], q=self.kpt.q, global_array=True)
            for b_G, array_G in zip(b_nG, array_nG):
                b_G[:] = wfs.pd.gather(array_G, self.kpt.q)
        else:
            b_nG = wfs.gd.collect(array_nG)
        return b_nG

    def dist_nG(self, array_nG):
        wfs = self.iks.wfs
        p_nG = wfs.empty(self.nG[0], q=self.kpt.q)
        if wfs.mode == "pw":
            for n in range(self.nG[0]):
                a_G = None
                if wfs.pd.comm.rank == 0:
                    a_G = array_nG[n]
                p_nG[n] = wfs.pd.scatter(a_G, self.kpt.q)
        else:
            wfs.gd.distribute(array_nG, p_nG)
        return p_nG

    def calculate_rhs(self, kpt, u_m):
        iks = self.iks
        wfs = iks.wfs
        h_G = 2.0 * iks.weights_sG[kpt.s] * iks.dn_sG[kpt.s]
        psit_nG = self.psit_nG
        if wfs.mode == "pw":
            psit_nG = iks.get_psit(kpt)[self.occupied_m]
        b_nG = np.zeros_like(psit_nG)
        for f, psit_G, b_G in zip(self.f_n, psit_nG, b_nG):
            b_G[:] = f * h_G * psit_G
        if wfs.mode == "pw":
            b_nR = b_nG.copy()
            b_nG = wfs.empty(self.nG[0], q=kpt.q)
            for b_G, b_R in zip(b_nG, b_nR):
                b_G[:] = wfs.pd.fft(b_R, q=kpt.q)
        wfs.pt.add(b_nG, self.dp_ani, q=kpt.q)
        # -2 mu_n S psi_n
        if iks.calculate_lagrange_mu:
            muSpsi_nG = np.einsum("n,n...->n...", u_m, self.psit_nG)
            c_ani = wfs.pt.dict(self.nG[0])
            for a, c_ni in c_ani.items():
                c_ni[:] = (
                    u_m[:, np.newaxis]
                    * np.dot(wfs.setups[a].dO_ii, kpt.P_ani[a][self.occupied_m].T).T
                )
            wfs.pt.add(muSpsi_nG, c_ani, q=kpt.q)
            b_nG -= muSpsi_nG
        return b_nG

    def update_paw(self):
        iks = self.iks
        N = self.nG[0]
        kpt = self.kpt
        dp_ani = iks.wfs.pt.dict((N,))
        for a, dD_sp in iks.dD_asp.items():
            Phi_ii = calculate_diff_overlap_matrix(
                iks.setups[a], iks.weights_asng[a][kpt.s], iks.atomic_weights, dD_sp[self.kpt.s], iks.mix_paw_error
            )
            if np.isnan(Phi_ii).any():
                # ae setups
                dp_ani[a][:] = 0.0
            else:
                dp_ani[a] = np.einsum(
                    "ij, nj,n->ni",
                    Phi_ii,
                    kpt.P_ani[a][self.occupied_m],
                    2.0 * self.f_n,
                )
        self.dp_ani = dp_ani

    def update_kpt(self, kpt):
        wfs = self.iks.wfs
        bd = wfs.bd
        self.kpt = kpt
        f_n = bd.collect(kpt.f_n, broadcast=True)
        occupied_m = self.occupied_m = np.flatnonzero(f_n > 1e-10)
        f_n = self.f_n = f_n[occupied_m]
        self.psit_nG = psit_nG = bd.collect(
            kpt.psit_nG, broadcast=True)[occupied_m]
        eps_n = wfs.bd.collect(kpt.eps_n, broadcast=True)[occupied_m]
        self.eps_n = eps_n
        if wfs.mode == "pw":
            nG = (len(psit_nG), wfs.pd.ng_q[kpt.q])
        else:
            nG = (len(psit_nG), *wfs.gd.get_size_of_global_array())
        self.nG = nG
        self.update_paw()

    def calculate_p(self, kpt: KPoint, p0_nG, u_m, prec: bool = False, **ls_kwargs):
        if self.method is None or self.method.lower() == "none":
            self.p_nG = p0_nG
            return self.p_nG
        elif self.method in scipy_linear_solvers:
            return self.calculate_p_scipy(kpt, p0_nG, u_m, prec, **ls_kwargs)
        elif self.method == "minres":
            wfs = self.iks.wfs

            if prec:
                raise NotImplementedError
            rhs = self.calculate_rhs(kpt, u_m)
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
                **ls_kwargs,
            )
            p_nG = p_nG.reshape(shape)
            self.iks.fmt.log(f"minres finished in {iters} iterations.")
            if self.ortho:
                p_nG = self.orthogonalize(kpt, p_nG, self.psit_nG)
            return p_nG
        else:
            raise ValueError(f"Unkown linear solver {self.method}")

    def calculate_p_scipy(self, kpt, p0_nG, u_m, prec, **ls_kwargs):
        iks = self.iks
        npts = np.prod(self.nG)
        lhs_op = LinearOperator(
            (npts, npts), matvec=self.apply_lhs_scipy, dtype=iks.wfs.dtype
        )
        # Mop = LinearOperator((npts, npts), matvec=preconditioner, dtype=iks.wfs.dtype)
        if prec:
            raise NotImplementedError(
                "preconditioer not implemented for scipy linear solvers"
            )
        Mop = None

        rhs = np.ravel(self.collect_nG(self.calculate_rhs(kpt, u_m)))
        # initial guess as the orbital expansion
        p0_nG = self.collect_nG(p0_nG)
        p_nG = None
        if self.comm.rank == 0:
            try:
                method = scipy_linear_solvers[self.method]
                p_nG, info = method(
                    lhs_op, rhs, x0=p0_nG.ravel(), M=Mop, **ls_kwargs)
                # iks.fmt.log(f"scipy {self.method} finished with info: {info}")
                # iks.fmt.flush()
                if info < 0:
                    raise RuntimeError(f"minres breakdown {info=}")
                p_nG = p_nG.reshape(self.nG)
            finally:
                self.comm.broadcast(self.msg_stop, 0)
        else:
            msg = np.empty(1, dtype=int)
            while True:
                self.comm.broadcast(msg, 0)
                m = msg.item()
                if m == self.mwork:
                    self.apply_lhs_scipy()
                elif m == self.mstop:
                    break

        p_nG = self.dist_nG(p_nG)
        if self.ortho:
            p_nG = self.orthogonalize(kpt, p_nG, self.psit_nG)
        self.p_nG = p_nG
        return p_nG

    def calculate_jacobian_one_k(self):
        p_nG = self.p_nG
        if self.iks.wfs.mode == "pw":
            p_nG = np.asarray(
                [self.iks.wfs.pd.ifft(p_G, q=self.kpt.q) for p_G in self.p_nG]
            )
        return 2.0 * np.sum((p_nG.conj() * self.iks.get_psit(self.kpt)[self.occupied_m]).real, axis=0)


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
