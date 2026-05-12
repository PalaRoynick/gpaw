from __future__ import annotations

from typing import TYPE_CHECKING
from collections import deque

import numpy as np
from scipy.optimize import minimize

from ase.parallel import DummyMPI
from ase.utils.abc import Optimizable
from gpaw.xc.numericalxc import broadcast_arraydict_data, dist_arraydict_data
from gpaw.sphere.lebedev import weight_n

if TYPE_CHECKING:
    from gpaw.typing import Array1D, Array4D, ArrayND
    from gpaw.old.grid_descriptor import GridDescriptor
    from gpaw.old.arraydict import ArrayDict
    from typing import Any
    from collections import MutableMapping
    from gpaw.iks import iKS
    from scipy.optimize import OptimizeResult
    NaN = float


def adam_step(
    alpha: float,
    vxc: ArrayND,
    J: ArrayND,
    m: ArrayND,
    mm: ArrayND,
    t: int,
    beta1: float,
    beta2: float,
) -> None:
    epsilon = 1e-8
    # Update biased first and second moment estimate
    m[:] = beta1 * m + (1.0 - beta1) * J
    mm[:] = beta2 * mm + (1.0 - beta2) * J**2
    # bias corrected
    m_hat = m / (1.0 - beta1**t)
    mm_hat = mm / (1.0 - beta2**t)
    # Update vxc
    vxc[:] = vxc - alpha * m_hat / (np.sqrt(mm_hat) + epsilon)


def iks_adam_opt(
    iks: iKS,
    maxiter: int,
    eta_pseudo: float,
    eta_paw: float,
    Ltol: float,
    LDtol: float,
    Jtol: float,
    beta1: float,
    beta2: float,
) -> list[float]:
    assert iks.xc.vxc_sg is not None
    if iks.use_coarse_grid:
        gd = iks.gd
        vxc_sg = gd.zeros(iks.nspins)
        iks.fine2coarse(iks.xc.vxc_sg, vxc_sg)
    else:
        gd = iks.finegd
        vxc_sg = iks.xc.vxc_sg
    m_sg = np.zeros_like(vxc_sg)
    mm_sg = np.zeros_like(m_sg)
    m_xasz = []
    mm_xasz = []
    for v_asz in iks.xc.vxc_xasz:
        m_xasz.append({a: np.zeros_like(vxc) for a, vxc in v_asz.items()})
        mm_xasz.append({a: np.zeros_like(vxc) for a, vxc in v_asz.items()})

    for t in range(1, maxiter + 1):
        iks.niter = t
        vxc_xasz = iks.xc.vxc_xasz
        # step
        iks.iterate()
        L1, L2, LD1, LD2 = iks.lagrangian()
        L = L2
        LD = LD2
        iks.L_iter.append(L)
        iks.LD_iter.append(LD)
        J_sg = iks.jacobian()
        iks.atomic_jacobian()
        # update pseudo
        adam_step(eta_pseudo, vxc_sg,
                  J_sg, m_sg, mm_sg, t, beta1, beta2)
        if iks.use_coarse_grid:
            iks.coarse2fine(vxc_sg, iks.xc.vxc_sg)
        else:
            iks.xc.vxc_sg[:] = vxc_sg
        # update paw
        for vxc_asz, J_asz, m_asz, mm_asz in zip(vxc_xasz, iks.J_xasz, m_xasz, mm_xasz):
            J_asz = iks.hamiltonian.atomdist.to_work(J_asz)
            for a in vxc_asz:
                adam_step(
                    eta_paw, vxc_asz[a], J_asz[a], m_asz[a], mm_asz[a], t, beta1, beta2
                )
        # log
        iks.logiter(t, L1, L2, LD1, LD2, J_sg)
        iks.calc.call_observers(iks.niter)
        # check convergence
        if iks.converged(L, LD, J_sg, Ltol, LDtol, Jtol):
            iks.fmt.log("OECO-iKS converged!")
            break
    return iks.L_iter


class iKSObj:
    """OECO objective for scipy optimization"""

    mwork = 1
    mstop = -1

    def __init__(self, iks: iKS, scale: float = 1.0):
        self.iks = iks
        self.msg_stop = np.array(self.mstop, dtype=int)
        self.msg_work = np.array(self.mwork, dtype=int)
        self.xc = iks.xc
        assert self.xc.vxc_sg is not None
        self.scale = scale

        if iks.use_coarse_grid:
            self.gd = iks.gd
        else:
            self.gd = iks.finegd
        self.result: OptimizeResult | None = None

    def collect_array(self, gd: GridDescriptor, source: Array4D, v: Array1D | None):
        # collect the original
        if gd.comm.rank == 0:
            vf = gd.empty(source.shape[:-3], global_array=True)
        else:
            vf = None
        gd.collect(source, vf)
        if gd.comm.rank == 0:
            if source.ndim == 3:
                nspins = 1
            else:
                nspins = len(source)
            size = np.prod(gd.N_c) * nspins
            assert v is not None
            vf[:] = v[:size].real
            # the atomic parts
            vf = np.append(vf.ravel(), v[size:])
        return vf

    def pack_jac(self, J_sg: Array4D, J_xasz: list[ArrayDict]) -> Array1D | NaN:
        J_sg = self.gd.collect(J_sg)
        if self.gd.comm.rank == 0:
            J = [J_sg.ravel()]
        for J_asz in J_xasz:
            data_asz = broadcast_arraydict_data(J_asz)
            if self.gd.comm.rank == 0:
                J.append(data_asz)
        if self.gd.comm.rank == 0:
            return np.concatenate(J)
        else:
            return np.nan

    def packed_coarse2fine(self, v: Array1D | None) -> Array1D | NaN:
        # this function is only called when iks.use_coarse_grid=True
        iks = self.iks
        if iks.gd.comm.rank == 0:
            size = np.prod(iks.gd.N_c) * iks.nspins
            data_sG = iks.gd.zeros(iks.nspins, global_array=True)
            data_sG[:] = v[:size].real
        else:
            size = 0
            data_sG = None
        v_sG = iks.gd.empty(iks.nspins)
        iks.gd.distribute(data_sG, v_sG)
        v_sg = iks.finegd.empty(iks.nspins)
        iks.coarse2fine(v_sG, v_sg)
        v_sg = iks.finegd.collect(v_sg)
        if iks.gd.comm.rank == 0:
            return np.concatenate([v_sg.ravel(), v[size:]])
        else:
            return np.nan

    def calculate_error_and_jac(
        self, v: Array1D | None = None
    ) -> tuple[float, Array1D | NaN]:
        iks = self.iks
        if iks.use_coarse_grid:
            v = self.packed_coarse2fine(v)
        v = self.collect_array(iks.finegd, self.xc.vxc_sg, v)
        self.xc.set_pots(iks.finegd, v)
        # step
        iks.niter += 1
        iks.iterate()
        # L and J
        L1, L2, LD1, LD2 = iks.lagrangian()
        L = L2
        LD = LD2
        J_sg = iks.jacobian()
        iks.atomic_jacobian()
        J_xasz = [iks.hamiltonian.atomdist.to_work(
            J_asz) for J_asz in iks.J_xasz]
        J = self.pack_jac(J_sg, J_xasz)
        # log
        iks.L_iter.append(L)
        iks.LD_iter.append(LD)
        iks.logiter(iks.niter, L1, L2, LD1, LD2, J_sg)
        iks.calc.call_observers(iks.niter)
        return self.scale * (L + LD), self.scale * J

    def __call__(self, param: Array1D | None):
        if self.gd.comm.rank == 0:
            self.gd.comm.broadcast(self.msg_work, 0)
        return self.calculate_error_and_jac(param)

    def run(self, method: str, options: dict[str, Any], **opt_kwargs):
        iks = self.iks
        comm = self.gd.comm
        if iks.use_coarse_grid:
            v_sg = self.gd.empty(iks.nspins)
            iks.fine2coarse(self.xc.vxc_sg, v_sg)
        else:
            v_sg = self.xc.vxc_sg.copy()
        v0 = self.pack_jac(v_sg, self.xc.vxc_xasz)

        vopt: Array1D | None = None
        if comm.rank == 0:
            try:
                self.result = minimize(
                    self, v0, method=method, options=options, **opt_kwargs, jac=True
                )
                iks.fmt.log("Optimization result:")
                iks.fmt.log(self.result)
                iks.fmt.flush()
                vopt = self.result.x
            finally:
                comm.broadcast(self.msg_stop, 0)

        else:
            msg = np.empty(1, dtype=int)
            while True:
                comm.broadcast(msg, 0)
                m = msg.item()
                if m == self.mwork:
                    self.calculate_error_and_jac()
                elif m == self.mstop:
                    break
        if iks.use_coarse_grid:
            vopt = self.packed_coarse2fine(vopt)
        vopt = self.collect_array(iks.finegd, self.xc.vxc_sg, vopt)
        self.xc.set_pots(iks.finegd, vopt)


class iKSMixOpt:
    mstop = -1
    mcalc = 1
    mstep = 2

    def __init__(self, iks: iKS):
        self.iks = iks
        self.msg_stop = np.array(self.mstop, dtype=int)
        self.msg_calc = np.array(self.mcalc, dtype=int)
        self.msg_step = np.array(self.mstep, dtype=int)
        if iks.use_coarse_grid:
            self.gd = iks.gd
        else:
            self.gd = iks.finegd
        self.alpha_paw: float | None = None
        self.method: str | None = None
        self.L_eval = []
        self.LD_eval = []
        self.n_eval = 0

    def dist_vxc_sg(self, v, v_sg=None):
        gd = self.gd
        nspins = self.iks.nspins
        if v_sg is None:
            v_sg = gd.empty(nspins)
        if gd.comm.rank == 0:
            v = v.reshape((nspins, *gd.get_size_of_global_array(pad=False)))
        gd.distribute(v, v_sg)
        if self.iks.use_coarse_grid:
            v_sG = v_sg.copy()
            v_sg = self.iks.finegd.empty(len(v_sG))
            self.iks.coarse2fine(v_sG, v_sg)
        return v_sg

    def calculate_lagrangian_and_jacobian(self, v=None):
        self.n_eval += 1
        iks = self.iks
        v_sg = self.dist_vxc_sg(v)
        iks.xc.vxc_sg[:] = v_sg
        # step
        iks.fmt.log(
            f"{self.method} evaluation #{self.n_eval}, iter={iks.niter}:")
        iks.iterate()
        # L and J
        L1, L2, LD1, LD2 = iks.lagrangian()
        J_sg = iks.jacobian()
        iks.atomic_jacobian()  # for logging
        # log
        iks.logiter(iks.niter, L1, L2, LD1, LD2, J_sg, sep="")
        J = self.gd.collect(J_sg)
        self.L_eval.append(L2)
        self.LD_eval.append(LD2)
        if self.gd.comm.rank == 0:
            return (L2 + LD2), J.ravel()

    def _calculate_lagrangian_and_jacobian(self, param: Array1D | None):
        if self.gd.comm.rank == 0:
            self.gd.comm.broadcast(self.msg_calc, 0)
        return self.calculate_lagrangian_and_jacobian(param)

    def step(self, v=None, apply_grads=True):
        iks = self.iks
        iks.niter += 1
        v_sg = self.dist_vxc_sg(v)
        iks.xc.vxc_sg[:] = v_sg
        if apply_grads:
            iks.fmt.log("Updating the atomic potentials:")
            iks.iterate()
            J_xasz = iks.atomic_jacobian()
            for J_asz, v_asz in zip(J_xasz, iks.xc.vxc_xasz):
                J_a = iks.hamiltonian.atomdist.to_work(J_asz)
                for a, J in J_a.items():
                    v_asz[a][:] -= self.alpha_paw * J
        iks.fmt.log(f"{self.method} optimization step iter={iks.niter}:")
        iks.iterate()
        L1, L2, LD1, LD2 = iks.lagrangian()
        J_sg = iks.jacobian()
        iks.atomic_jacobian()
        iks.L_iter.append(L2)
        iks.LD_iter.append(LD2)
        iks.logiter(iks.niter, L1, L2, LD1, LD2, J_sg)
        iks.calc.call_observers(iks.niter)

    def _step(self, param: Array1D | None):
        if self.gd.comm.rank == 0:
            self.gd.comm.broadcast(self.msg_step, 0)
        return self.step(param)

    def run(self, alpha_paw: float = 1.0, method: str = "L-BFGS-B", **opt_kwargs):
        self.alpha_paw = alpha_paw
        self.method = method
        iks = self.iks
        comm = self.gd.comm
        if iks.use_coarse_grid:
            v_sg = self.gd.empty(iks.nspins)
            iks.fine2coarse(iks.xc.vxc_sg, v_sg)
        else:
            v_sg = iks.xc.vxc_sg.copy()
        v0 = self.gd.collect(v_sg)

        vopt: Array1D | None = None
        if comm.rank == 0:
            try:
                self.result = minimize(
                    self._calculate_lagrangian_and_jacobian,
                    v0.ravel(),
                    method=method,
                    **opt_kwargs,
                    jac=True,
                    callback=self._step,
                )
                iks.fmt.log("Optimization result:")
                iks.fmt.log(self.result)
                iks.fmt.flush()
                vopt = self.result.x
            finally:
                comm.broadcast(self.msg_stop, 0)

        else:
            msg = np.empty(1, dtype=int)
            while True:
                comm.broadcast(msg, 0)
                m = msg.item()
                if m == self.mcalc:
                    self.calculate_lagrangian_and_jacobian()
                elif m == self.mstep:
                    self.step()
                elif m == self.mstop:
                    break
        self.step(vopt, apply_grads=False)


class PotState:
    def __init__(self, data_sg: Array4D, data_xasz: list[MutableMapping]):
        self.data_sg = data_sg
        self.data_xasz = data_xasz
        self.gd = None
        self.setups = None
        self.comm = None

    def initialize(self, gd, comm, setups):
        self.gd = gd
        self.comm = comm
        self.setups = setups

    def __add__(self, other: "PotState") -> "PotState":
        if not isinstance(other, PotState):
            raise NotImplementedError
        new_xasz = []
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            new_xasz.append({a: data_asz[a] + other_asz[a] for a in data_asz})
        return PotState(self.data_sg + other.data_sg, new_xasz)

    def __sub__(self, other: "PotState") -> "PotState":
        if not isinstance(other, PotState):
            raise NotImplementedError
        new_xasz = []
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            new_xasz.append({a: data_asz[a] - other_asz[a] for a in data_asz})
        return PotState(self.data_sg - other.data_sg, new_xasz)

    def __neg__(self) -> "PotState":
        new_xasz = []
        for data_asz in self.data_xasz:
            new_xasz.append({a: -data_asz[a] for a in data_asz})
        return PotState(-self.data_sg, new_xasz)

    def __mul__(self, other: "PotState" | float) -> "PotState":
        if isinstance(other, PotState):
            new_xasz = []
            for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
                new_xasz.append(
                    {a: data_asz[a] * other_asz[a] for a in data_asz})
            return PotState(self.data_sg * other.data_sg, new_xasz)
        else:
            new_xasz = []
            for data_asz in self.data_xasz:
                new_xasz.append({a: v * other for a, v in data_asz.items()})
            return PotState(self.data_sg * other, new_xasz)

    def copy(self) -> "PotState":
        copy_xasz = []
        for array_asz in self.data_xasz:
            copy_xasz.append({a: v.copy() for a, v in array_asz.items()})
        x = PotState(self.data_sg.copy(), copy_xasz)
        x.initialize(self.gd, self.comm, self.setups)
        return x

    def inner(self, other):
        # ignoring the coupling between atomic and pseudo quantities
        assert isinstance(other, PotState)
        assert self.gd is not None and self.setups is not None
        II = np.zeros(1, dtype=float)
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            for a in data_asz:
                data_sng = data_asz[a]
                other_sng = other_asz[a]
                rgd = self.setups[a].xc_correction.rgd
                II += np.einsum("n,sng,sng,g", weight_n,
                                data_sng, other_sng, rgd.dv_g)
        self.comm.sum(II)  # over all atoms
        return self.gd.integrate(self.data_sg * other.data_sg, global_integral=True).sum() + II.item()

    def norm(self):
        return np.sqrt(self.inner(self))

    def __iadd__(self, other: "PotState") -> "PotState":
        self.data_sg += other.data_sg
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            for a, v in data_asz.items():
                v += other_asz[a]
        return self

    def __isub__(self, other: "PotState") -> "PotState":
        self.data_sg -= other.data_sg
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            for a, v in data_asz.items():
                v -= other_asz[a]
        return self

    def __imul__(self, other: "PotState | float") -> "PotState":
        if isinstance(other, PotState):
            self.data_sg *= other.data_sg
            for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
                for a, v in data_asz.items():
                    v *= other_asz[a]
        else:
            self.data_sg *= other
            for data_asz in self.data_xasz:
                for v in data_asz.values():
                    v *= other
        return self

    def add_scaled(self, other: "PotState", factor: float) -> None:
        self.data_sg += factor * other.data_sg
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            for a, v in data_asz.items():
                v += factor * other_asz[a]


class iKSLBFGS:
    def __init__(
        self,
        iks: iKS,
        m: int = 10,
        max_ls=10,
        maxiter: int = 300,
        Ltol: float = 1e-6,
        LDtol: float = 1e-4,
        Jtol: float = 1e-9,
        step_scale: float = 1.0,
    ):
        self.iks = iks
        if iks.use_coarse_grid:
            raise NotImplementedError(
                '`use_coarse_grid=True` is not implemented for iKSLBFGS')
        self.m = m
        self.maxiter = maxiter
        self.max_ls = max_ls
        self.Ltol = Ltol
        self.LDtol = LDtol
        self.Jtol = Jtol
        self.step_scale = step_scale
        self.world = iks.hamiltonian.world

        self.s_hist: deque[PotState] = deque(maxlen=m)
        self.y_hist: deque[PotState] = deque(maxlen=m)
        self.rho_hist: deque[float] = deque(maxlen=m)

        self.L_iter: list[float] = []
        self.LD_iter: list[float] = []

    def apply_lbfgs(self, gk: PotState) -> PotState:
        if not self.s_hist:
            return -gk * self.step_scale
        q = gk.copy()
        alpha_list: list[float] = []
        for s_i, y_i, rho_i in zip(reversed(self.s_hist),
                                   reversed(self.y_hist),
                                   reversed(self.rho_hist)):
            s_i.initialize(self.iks.finegd, self.world, self.iks.setups)
            alpha_i = rho_i * s_i.inner(q)
            alpha_list.append(alpha_i)
            q.add_scaled(y_i, -alpha_i)
        s_last = self.s_hist[-1]
        y_last = self.y_hist[-1]
        y_last.initialize(self.iks.finegd, self.world, self.iks.setups)

        sy = y_last.inner(s_last)
        yy = y_last.inner(y_last)
        if yy > 1e-14:
            gamma = sy / yy
            if gamma < 0:
                gamma = 1.0
        else:
            gamma = 1.0
        r = q
        r *= gamma
        alpha_list.reverse()
        for alpha_i, (s_i, y_i, rho_i) in zip(alpha_list,
                                              zip(self.s_hist, self.y_hist, self.rho_hist)):
            s_i.initialize(self.iks.finegd, self.world, self.iks.setups)
            beta_i = rho_i * y_i.inner(r)
            r.add_scaled(s_i, (alpha_i - beta_i))
        return -r

    def line_search(self, x, gk, pk, L0, LD0, phi0, dphi0, c1=1e-4, tau=0.5, max_ls=10):
        """
        Backtracking Armijo line search with quadratic interpolation.
        """
        iks = self.iks
        xc = iks.xc
        alpha = 1.0
        L_new = L0
        LD_new = LD0
        g_new = gk
        accepted = False

        best_phi = phi0
        best_state = None  # (x_best, g_best, L_best, LD_best, alpha_best)

        for il in range(max_ls):
            xc.vxc_sg[:] = x.data_sg + alpha * pk.data_sg
            for vxc_asz, newvxc_asz, p_asz in zip(
                xc.vxc_xasz, x.data_xasz, pk.data_xasz
            ):
                for a in vxc_asz.keys():
                    vxc_asz[a][:] = newvxc_asz[a] + alpha * p_asz[a]
                vxc_asz.check_consistency()
            iks.iterate()
            self.L1, L_new, self.LD1, LD_new = iks.lagrangian()
            J_sg = iks.jacobian()
            J_xasz = iks.atomic_jacobian()
            Jwork_xasz = [iks.hamiltonian.atomdist.to_work(
                J_asz) for J_asz in J_xasz]
            iks.fmt.log(f"line search {il}")
            iks.logiter(il, self.L1, L_new, self.LD1,
                        LD_new, g_new.data_sg, sep="")
            g_new = PotState(J_sg, Jwork_xasz)
            phi_new = L_new + LD_new

            # track the best phi we have seen along this line
            if phi_new < best_phi:
                best_phi = phi_new
                x_best = PotState(xc.vxc_sg, xc.vxc_xasz).copy()
                g_best = g_new.copy()
                best_state = (x_best, g_best, L_new, LD_new, alpha)

            if phi_new <= phi0 + c1 * alpha * dphi0:
                accepted = True
                break
            # quadratic interpolation
            denom = 2.0 * (phi_new - phi0 - dphi0 * alpha)
            if denom > 0:
                alpha_quad = - (dphi0 * alpha**2) / denom
                # don't reduce alpha too slowly or too fast
                alpha_new = max(0.1 * alpha, min(0.9 * alpha, alpha_quad))
                iks.fmt.log(
                    f"  iKSLBFGS LS Interpolation: {alpha:.4e} -> {alpha_new:.4e}")
                alpha = alpha_new
            else:
                # fallback to simple backtracking if curvature is negative
                alpha *= tau
        if accepted:
            x_new = PotState(xc.vxc_sg, xc.vxc_xasz).copy()
            return x_new, g_new, L_new, LD_new, alpha, True

        # no Armijo, but we found some alpha with phi_new < phi0
        if best_state is not None and best_phi < phi0:
            self.iks.fmt.log(
                "L-BFGS line search: Armijo failed, but found decreasing lagrangian; "
                "using best lagrangian step along direction."
            )
            x_best, g_best, L_best, LD_best, alpha_best = best_state
            # make sure xc matches x_best
            xc.vxc_sg[:] = x_best.data_sg
            for vxc_asz, vbest_asz in zip(xc.vxc_xasz, x_best.data_xasz):
                for a in vxc_asz.keys():
                    vxc_asz[a][:] = vbest_asz[a]

            return x_best, g_best, L_best, LD_best, alpha_best, True
        # nothing along this direction improves phi
        self.iks.fmt.log(
            "Warning: L-BFGS line search: no step reduced Lagrangian; rejecting direction.")
        # restore potentials to x
        xc.vxc_sg[:] = x.data_sg
        for vxc_asz, newvxc_asz in zip(xc.vxc_xasz, x.data_xasz):
            for a in vxc_asz.keys():
                vxc_asz[a][:] = newvxc_asz[a]
        return x.copy(), gk, L0, LD0, 0.0, False

    def run(self) -> list[float]:
        iks = self.iks

        if not iks.initialized:
            iks.initialize()

        opt_kwargs = {
            "memory": self.m,
            "max_ls": self.max_ls,
            "maxiter": self.maxiter,
            "smooth Lagrangian tolerance": self.Ltol,
            "atomic Lagrangian tolerance": self.LDtol,
            "gradients-tolerance": self.Jtol,
        }
        iks.preopt(
            "L-BFGS optimizer for OECO-iKS procedure (with line search)", opt_kwargs
        )

        self.iks.L_iter = []
        self.iks.LD_iter = []

        iks.niter = 0
        iks.iterate()
        L1, L, LD1, LD = iks.lagrangian()
        phi = L + LD
        J_sg = iks.jacobian()
        J_xasz = iks.atomic_jacobian()
        Jwork_xasz = []
        for J_asz in J_xasz:
            Jwork_xasz.append(iks.hamiltonian.atomdist.to_work(J_asz))
        gk = PotState(J_sg, Jwork_xasz).copy()
        x = PotState(iks.xc.vxc_sg, iks.xc.vxc_xasz).copy()

        x_prev = None
        g_prev = None

        for t in range(1, self.maxiter + 1):
            iks.niter = t

            # convergence check with current (L, LD, gk) at x
            if iks.converged(L, LD, gk.data_sg, self.Ltol, self.LDtol, self.Jtol):
                iks.fmt.log("OECO-iKS L-BFGS converged!")
                break

            #  update L-BFGS history using previous step
            if x_prev is not None and g_prev is not None:
                s_k = x - x_prev
                y_k = gk - g_prev
                s_k.initialize(iks.finegd, self.world, iks.setups)
                y_k.initialize(iks.finegd, self.world, iks.setups)

                sy = s_k.inner(y_k)
                ss = s_k.inner(s_k)
                yy = y_k.inner(y_k)

                # tiny epsilon relative to scales to avoid numerical junk
                eps = 1e-10
                if sy > eps * np.sqrt(ss * yy):
                    rho_k = 1.0 / sy
                    self.s_hist.append(s_k)
                    self.y_hist.append(y_k)
                    self.rho_hist.append(rho_k)
                else:
                    # skip update if curvature information is unreliable
                    iks.fmt.log(
                        "Skipping L-BFGS update: bad curvature (s^T y too small or non-positive)"
                    )
            # search direction
            pk = self.apply_lbfgs(gk)
            pk.initialize(iks.finegd, self.world, iks.setups)
            dphi0 = pk.inner(gk)
            norm_p = pk.norm()
            gk.initialize(iks.finegd, self.world, iks.setups)
            norm_g = gk.norm()
            cosine = -pk.inner(gk) / (norm_p * norm_g)
            if cosine < 1e-4:
                # direction is almost orthogonal to gradient -> reset history
                self.s_hist.clear()
                self.y_hist.clear()
                self.rho_hist.clear()
                pk = -gk * self.step_scale
                pk.initialize(iks.finegd, self.world, iks.setups)
                dphi0 = pk.inner(gk)

            # line search along pk
            x_new, g_new, L_new, LD_new, alpha, accepted = self.line_search(
                x, gk, pk, L, LD, phi, dphi0, max_ls=self.max_ls
            )

            if not accepted:
                # reset LBFGS and try a steepest descent step once.
                self.s_hist.clear()
                self.y_hist.clear()
                self.rho_hist.clear()

                pk = -gk * self.step_scale
                pk.initialize(iks.finegd, self.world, iks.setups)
                dphi0 = pk.inner(gk)
                x_new, g_new, L_new, LD_new, alpha, accepted = self.line_search(
                    x, gk, pk, L, LD, phi, dphi0, max_ls=self.max_ls
                )
                if not accepted:
                    iks.fmt.log(
                        "L-BFGS: SD line search also failed to reduce lagrangian; stopping to avoid stagnation.")
                    break
            # if we got here, we have an accepted step
            phi = L_new + LD_new
            self.iks.L_iter.append(L_new)
            self.iks.LD_iter.append(LD_new)
            iks.logiter(t, self.L1, L_new, self.LD1, LD_new, g_new.data_sg)
            iks.calc.call_observers(t)
            x_prev, g_prev = x, gk
            x, gk, L, LD = x_new, g_new, L_new, LD_new

        iks.afteropt()
        return self.L_iter


class iKSOptimizable(Optimizable):
    # There is some issue with the k-point parallelization in iKSOptimizable
    def __init__(self, iks: iKS):
        self.iks = iks
        self.world = iks.hamiltonian.world
        self.pool_code = {
            -1: 'stop',
            0: 'idle',
            1: 'get_x',
            2: 'set_x',
            3: 'get_gradient',
            4: 'get_value',
            5: 'gradient_norm',
            6: 'converged',
        }
        self.code = {value: key for key, value in self.pool_code.items()}
        self.initialized = False

    def initialize(self, title='iKSOptimizable', opt_kwargs={}):
        self.iks.preopt(title, opt_kwargs)
        self.iks.iterate()
        self.t = 0
        self.pool_started = False
        self.initialized = True

    def start_pool(self):
        self.pool_started = True
        if self.world.rank != 0:
            msg = np.zeros(1, dtype=int)
            while True:
                self.world.broadcast(msg, 0)
                m = msg.item()
                if m == -1:
                    break
                elif m == 0:
                    continue
                else:
                    method = getattr(self, self.pool_code[m])
                    method()

    def finish_pool(self):
        if self.world.rank == 0:
            self.signal_pool(self.code['stop'])

    def signal_pool(self, mc):
        msg = np.array(mc, dtype=int)
        self.world.broadcast(msg, 0)

    def pack_jac(self, J_sg: Array4D, J_xasz: list[ArrayDict]) -> Array1D | NaN:
        assert not self.iks.use_coarse_grid
        gd = self.iks.finegd
        comm = self.world
        J_sg = gd.collect(J_sg)
        if comm.rank == 0:
            J = [J_sg.ravel()]
        for J_asz in J_xasz:
            data_asz = broadcast_arraydict_data(J_asz)
            if comm.rank == 0:
                J.append(data_asz)
        if comm.rank == 0:
            return np.concatenate(J)
        else:
            return np.nan

    def unpack_jac(self, gd: GridDescriptor, J: Array1D | None) -> None:
        comm = self.world
        Jg_sg = None
        J_sg = gd.empty(self.iks.nspins)
        fsize = 0
        if comm.rank == 0:
            Jg_sg = gd.empty(self.iks.nspins, global_array=True)
            fsize = Jg_sg.size
            Jg_sg.ravel()[:] = J[:fsize].real
        gd.distribute(Jg_sg, out=J_sg)
        i = fsize

        J_xasz = [self.iks.atompart.arraydict(
            v_asz.shapes_a, dtype=float) for v_asz in self.iks.xc.vxc_xasz]
        for J_asz in J_xasz:
            shapes_a = J_asz.shapes_a
            sizea = sum([np.prod(shape) for shape in shapes_a])
            data_asz = np.empty(sizea, dtype=float)
            if comm.rank == 0:
                data_asz[:] = J[i: i + sizea].real
            i += sizea
            # broadcast so we don't have to worry about the atomic partition
            comm.broadcast(data_asz, 0)
            dist_arraydict_data(J_asz, data_asz)
        return J_sg, J_xasz

    def ndofs(self) -> int:
        return self.iks.get_xcpotential_space_size()

    def get_x(self) -> np.ndarray:
        if self.world.rank == 0:
            self.signal_pool(self.code['get_x'])
        v = self.iks.xc.pack_pots(self.iks.finegd)
        return v

    def set_x(self, x=None) -> None:
        if self.world.rank == 0:
            self.signal_pool(self.code['set_x'])
        self.iks.xc.set_pots(self.iks.finegd, x)
        self.iks.iterate()

    def get_gradient(self):
        if self.world.rank == 0:
            self.signal_pool(self.code['get_gradient'])
        J_sg = self.iks.jacobian()
        J_xasz = self.iks.atomic_jacobian()
        J = self.pack_jac(J_sg, J_xasz)
        return J

    def get_value(self) -> float:
        if self.world.rank == 0:
            self.signal_pool(self.code['get_value'])
        self.t += 1
        L1, L, LD1, LD = self.iks.lagrangian()
        self.iks.logiter(self.t, L1, L, LD1, LD, self.iks.J_sg,)
        return L + LD

    def iterimages(self):
        raise NotImplementedError

    def converged(self, gradient=None, fmax=None) -> bool:
        if self.world.rank == 0:
            self.signal_pool(self.code['converged'])
        norm = self.gradient_norm(gradient, signal=False)
        if self.world.rank == 0:
            return norm < fmax
        else:
            return False

    def gradient_norm(self, gradient=None, signal=True):
        if self.world.rank == 0 and signal:
            self.signal_pool(self.code['gradient_norm'])
        J_sg, J_xasz = self.unpack_jac(self.iks.finegd, gradient)
        gk = PotState(J_sg, J_xasz)
        gk.initialize(self.iks.finegd,
                      self.world, self.iks.setups)
        norm = gk.norm()
        return norm

    def run(self, opt_cls, maxiter=312, Jmax=1e-10, opt_run_kwargs={}, **kwargs):
        opt = None
        opt_run_kwargs.setdefault('steps', maxiter)
        opt_run_kwargs.setdefault('fmax', Jmax)
        if not self.initialized:
            opt_kwargs = kwargs.copy()
            opt_kwargs.update(opt_run_kwargs)
            self.initialize(
                f'ASE {opt_cls.__name__} optimizer for iKS', opt_kwargs)
        self.start_pool()
        if self.world.rank == 0:
            opt = opt_cls(self, **kwargs, comm=DummyMPI())
            opt.run(**opt_run_kwargs)
        self.finish_pool()
        self.iks.afteropt()
