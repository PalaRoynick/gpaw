from __future__ import annotations
from typing import TYPE_CHECKING
from collections import deque

import numpy as np
from scipy.optimize import minimize

from ase.parallel import DummyMPI
from ase.utils.abc import Optimizable
from gpaw.xc.numericalxc import broadcast_arraydict_data, dist_arraydict_data
# from gpaw.xc.numericalxc import lebedev2angular, angular2lebedev
from gpaw.sphere.lebedev import weight_n, Y_nL

if TYPE_CHECKING:
    from gpaw.typing import Array1D, Array4D, ArrayND
    from gpaw.old.grid_descriptor import GridDescriptor
    from gpaw.old.arraydict import ArrayDict
    from typing import Any
    from collections import MutableMapping
    from gpaw.iks import iKS
    from scipy.optimize import OptimizeResult
    NaN = float


def adam_step(alpha: float, vxc: ArrayND, J: ArrayND, m: ArrayND,
              mm: ArrayND, t: int, beta1: float, beta2: float) -> None:
    epsilon = 1e-8
    # Update biased first and second moment estimate
    m[:] = beta1 * m + (1.0 - beta1) * J
    mm[:] = beta2 * mm + (1.0 - beta2) * J**2
    # bias corrected
    m_hat = m / (1.0 - beta1**t)
    mm_hat = mm / (1.0 - beta2**t)
    # Update vxc
    vxc[:] = vxc - alpha * m_hat / (np.sqrt(mm_hat) + epsilon)


def iks_adam_opt(iks: iKS, maxiter: int, eta_pseudo: float, eta_paw: float,
                 Ltol: float, LDtol: float, Jtol: float,
                 beta1: float, beta2: float,) -> list[float]:
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


class PotState:
    def __init__(self, data_sg: Array4D, data_xasz: list[MutableMapping], cvbar_a: MutableMapping,
                 gd=None, comm=None, setups=None, angular=False, kpt_par=False, cvbar_inner: callable = None):
        self.data_sg = data_sg
        self.data_xasz = data_xasz
        self.cvbar_a = cvbar_a
        self.gd = gd
        self.comm = comm
        self.setups = setups
        self.angular = angular
        self.kpt_par = kpt_par
        self.cvbar_inner = cvbar_inner

    @property
    def args(self):
        return (self.gd, self.comm, self.setups, self.angular, self.kpt_par, self.cvbar_inner)

    def __add__(self, other: "PotState") -> "PotState":
        if not isinstance(other, PotState):
            raise NotImplementedError
        new_xasz = []
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            new_xasz.append({a: data_asz[a] + other_asz[a] for a in data_asz})
        newc_a = {a: self.cvbar_a[a] +
                  other.cvbar_a[a] for a in self.cvbar_a}
        return PotState(self.data_sg + other.data_sg, new_xasz, newc_a, *self.args)

    def __sub__(self, other: "PotState") -> "PotState":
        if not isinstance(other, PotState):
            raise NotImplementedError
        new_xasz = []
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            new_xasz.append({a: data_asz[a] - other_asz[a] for a in data_asz})
        newc_a = {a: self.cvbar_a[a] -
                  other.cvbar_a[a] for a in self.cvbar_a}
        return PotState(self.data_sg - other.data_sg, new_xasz, newc_a, *self.args)

    def __neg__(self) -> "PotState":
        new_xasz = []
        for data_asz in self.data_xasz:
            new_xasz.append({a: -data_asz[a] for a in data_asz})
        newc_a = {a: -self.cvbar_a[a] for a in self.cvbar_a}
        return PotState(-self.data_sg, new_xasz, newc_a, *self.args)

    def __mul__(self, other: "PotState" | float) -> "PotState":
        if isinstance(other, PotState):
            new_xasz = []
            for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
                new_xasz.append(
                    {a: data_asz[a] * other_asz[a] for a in data_asz})
            newc_a = {a: self.cvbar_a[a] *
                      other.cvbar_a[a] for a in self.cvbar_a}
            return PotState(self.data_sg * other.data_sg, new_xasz, newc_a, *self.args)
        else:
            new_xasz = []
            for data_asz in self.data_xasz:
                new_xasz.append({a: v * other for a, v in data_asz.items()})
            newc_a = {a: c * other for a, c in self.cvbar_a.items()}
            return PotState(self.data_sg * other, new_xasz, newc_a, *self.args)

    def __rmul__(self, other: float) -> "PotState":
        return self.__mul__(other)

    def copy(self) -> "PotState":
        copy_xasz = []
        for array_asz in self.data_xasz:
            copy_xasz.append({a: v.copy() for a, v in array_asz.items()})
        cvbar_a = {a: v.copy() for a, v in self.cvbar_a.items()}
        x = PotState(self.data_sg.copy(), copy_xasz, cvbar_a, *self.args)
        return x

    def inner(self, other):
        # ignoring the coupling between atomic and pseudo quantities
        assert isinstance(other, PotState)
        assert self.gd is not None and self.setups is not None and self.cvbar_inner is not None
        II = np.zeros(1, dtype=float)
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            assert tuple(data_asz.keys()) == tuple(other_asz.keys())
            assert tuple(data_asz.keys()) == tuple(self.cvbar_a.keys())
            for a in data_asz:
                data_sng = data_asz[a]
                other_sng = other_asz[a]
                xcc = self.setups[a].xc_correction
                if xcc is None:
                    continue
                rgd = xcc.rgd
                if self.angular:
                    Lmax = data_sng.shape[1]
                    data_sng = np.einsum(
                        'nL,sLg->sng', Y_nL[:, :Lmax], data_sng)
                    other_sng = np.einsum(
                        'nL,sLg->sng', Y_nL[:, :Lmax], other_sng)
                II += np.einsum("n,sng,sng,g", weight_n,
                                data_sng, other_sng, rgd.dv_g)
        self.comm.sum(II)  # over all atoms
        # return self.gd.integrate(self.data_sg * other.data_sg, global_integral=True).sum() + II.item()
        It = self.gd.integrate(
            self.data_sg * other.data_sg, global_integral=True).sum()
        Ic = self.cvbar_inner(self.cvbar_a, other.cvbar_a)
        total = np.array([It + II.item() + Ic], dtype=float)
        if self.kpt_par:
            # prevent divergence in case of kpt-parallelization
            self.comm.broadcast(total, 0)
        return total.item()

    def norm(self):
        return np.sqrt(self.inner(self))

    def __iadd__(self, other: "PotState") -> "PotState":
        self.data_sg += other.data_sg
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            for a, v in data_asz.items():
                v += other_asz[a]
        for a in self.cvbar_a:
            self.cvbar_a[a] += other.cvbar_a[a]
        return self

    def __isub__(self, other: "PotState") -> "PotState":
        self.data_sg -= other.data_sg
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            for a, v in data_asz.items():
                v -= other_asz[a]
        for a in self.cvbar_a:
            self.cvbar_a[a] -= other.cvbar_a[a]
        return self

    def __imul__(self, other: "PotState | float") -> "PotState":
        if isinstance(other, PotState):
            self.data_sg *= other.data_sg
            for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
                for a in data_asz.keys():
                    data_asz[a] *= other_asz[a]
            for a in self.cvbar_a:
                self.cvbar_a[a] *= other.cvbar_a[a]
        else:
            self.data_sg *= other
            for data_asz in self.data_xasz:
                for a in data_asz.keys():
                    data_asz[a] *= other
            for a in self.cvbar_a:
                self.cvbar_a[a] *= other
        return self

    def add_scaled(self, other: "PotState", factor: float) -> None:
        self.data_sg += factor * other.data_sg
        for data_asz, other_asz in zip(self.data_xasz, other.data_xasz):
            for a in data_asz.keys():
                data_asz[a] += factor * other_asz[a]
        for a in self.cvbar_a:
            self.cvbar_a[a] += factor * other.cvbar_a[a]


class iKSLBFGS:
    def __init__(self, iks: iKS, m: int = 10, max_ls=15,
                 maxiter: int = 300, Ltol: float = 1e-6, LDtol: float = 1e-4,
                 Jtol: float = 1e-9, step_scale: float = 1.0):
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

    @property
    def potstate_args(self):
        iks = self.iks
        return (iks.finegd, iks.hamiltonian.world, iks.setups,
                iks.xc.paw_potmode.name == 'angular', iks.wfs.kptband_comm.size > 1, self.cvbar_inner)

    def from_workdict(self, w_a):
        iks = self.iks
        wpart = iks.xc.vxc_xasz[1].partition
        c_a = wpart.arraydict(
            [(1,) for a in range(len(iks.setups))])
        c_a.update(w_a)
        return iks.hamiltonian.atomdist.from_work(c_a)

    def cvbar_inner(self, c1_a, c2_a) -> float:
        II = 0.0

        assert set(c1_a) == set(c2_a)

        for a in c1_a:
            c1 = np.asarray(c1_a[a])
            c2 = np.asarray(c2_a[a])
            assert c1.shape == (1,), c1.shape
            assert c2.shape == (1,), c2.shape
            II += c1[0] * c2[0]

        value = np.array([II], dtype=float)
        self.iks.hamiltonian.world.sum(value)

        return float(value[0])

    def apply_lbfgs(self, gk: PotState) -> PotState:
        if not self.s_hist:
            return -gk * self.step_scale
        q = gk.copy()
        alpha_list: list[float] = []
        for s_i, y_i, rho_i in zip(reversed(self.s_hist),
                                   reversed(self.y_hist),
                                   reversed(self.rho_hist)):
            alpha_i = rho_i * s_i.inner(q)
            alpha_list.append(alpha_i)
            q.add_scaled(y_i, -alpha_i)
        s_last = self.s_hist[-1]
        y_last = self.y_hist[-1]

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
            beta_i = rho_i * y_i.inner(r)
            r.add_scaled(s_i, (alpha_i - beta_i))
        return -r * self.step_scale

    def set_potentials_and_iterate(self, x: PotState):
        iks = self.iks
        xc = iks.xc
        xc.vxc_sg[:] = x.data_sg + iks.lambda_vbar * \
            iks.get_vbar(self.from_workdict(x.cvbar_a))
        for a in x.data_xasz[0].keys():
            c = x.cvbar_a[a]
            v_sz = x.data_xasz[0][a]
            vt_sz = x.data_xasz[1][a]
            E_g = iks.sw_ag[a]
            xc.vxc_xasz[1][a][:] = vt_sz + iks.lambda_vbar * \
                c * iks.vbar_ag[a][None, None, :]
            xc.vxc_xasz[0][a][:] = v_sz * E_g + vt_sz * (1.0 - E_g)
        for vxc_asz in xc.vxc_xasz:
            vxc_asz.check_consistency()
        iks.iterate()

    def get_cvbar_grad(self, J_sg, Jt_asz):
        iks = self.iks
        Jc_a = {}
        partition = Jt_asz.partition
        for a in range(partition.natoms):
            Jc = 0.0
            coeff_a = {a_: np.zeros(1) for a_ in partition.my_indices}
            if a in partition.my_indices:
                coeff_a[a][:] = 1.0
            vbar_g = iks.get_vbar(self.from_workdict(coeff_a))
            for s in range(iks.nspins):
                Jc += iks.finegd.integrate(vbar_g * J_sg[s])
            if a in partition.my_indices:
                xcc = iks.setups[a].xc_correction
                for s in range(iks.nspins):
                    Jt_ng = Jt_asz[a][s]
                    for n, weight in enumerate(weight_n):
                        Jc += weight * \
                            xcc.rgd.integrate(
                                Jt_ng[n] * iks.vbar_ag[a])
                Jc_a[a] = iks.lambda_vbar * np.array([Jc])
        return Jc_a

    def transform_gradients(self, J_xasz: list[ArrayDict]) -> list[ArrayDict]:
        partition = J_xasz[0].partition
        JT_xasz: list[ArrayDict] = [
            partition.arraydict(J_asz.shapes_a) for J_asz in J_xasz]
        for a in partition.my_indices:
            E_g = self.iks.sw_ag[a]
            J_sz = J_xasz[0][a]
            Jt_sz = J_xasz[1][a]
            JT_xasz[0][a][:] = J_sz * E_g
            JT_xasz[1][a][:] = Jt_sz + J_sz * (1.0 - E_g)
        for JT_asz in JT_xasz:
            JT_asz.check_consistency()
        return JT_xasz

    def line_search(self, x: PotState, gk: PotState, pk: PotState,
                    L0: float, LD0: float, phi0: float, dphi0: float,
                    c1: float = 1e-4, c2: float = 0.9, tau: float = 0.5,
                    max_ls: int = 15
                    ) -> tuple[PotState, PotState, float, float, float, bool]:
        """Line search satisfying the strong Wolfe conditions."""

        iks = self.iks

        if not 0.0 < c1 < c2 < 1.0:
            raise ValueError(
                f'Require 0 < c1 < c2 < 1, got c1={c1}, c2={c2}')

        if not 0.0 < tau < 1.0:
            raise ValueError(
                f'Require 0 < tau < 1, got tau={tau}')

        if not np.isfinite(dphi0) or dphi0 >= 0.0:
            iks.fmt.log(
                'Strong-Wolfe line search received a non-descent direction: '
                f'dphi0={dphi0:.5e}')

            self.set_potentials_and_iterate(x)

            return x.copy(), gk.copy(), L0, LD0, 0.0, False

        alpha = 1.0
        alpha_max = 32.0

        # Wolfe bracket:
        alpha_lo = 0.0  # step satisfying Armijo
        alpha_hi = None  # alpha_hi is an upper bracket

        phi_lo = phi0

        g_new = gk
        L_new = L0
        LD_new = LD0

        for il in range(max_ls):
            x_new = x + alpha * pk
            self.set_potentials_and_iterate(x_new)

            self.L1, L_new, self.LD1, LD_new = iks.lagrangian()
            phi_new = float(L_new + LD_new)

            J_xasz = iks.atomic_jacobian()
            JT_xasz = self.transform_gradients(J_xasz)
            J_sg = iks.jacobian()

            atomdist = iks.hamiltonian.atomdist

            Jwork_xasz = [
                atomdist.to_work(J_asz)
                for J_asz in JT_xasz
            ]

            # Use the raw physical pseudo gradient for c^a.
            Jc_a = self.get_cvbar_grad(
                J_sg,
                atomdist.to_work(J_xasz[1])
            )

            g_new = PotState(
                J_sg,
                Jwork_xasz,
                Jc_a,
                *self.potstate_args
            )

            dphi_new = float(pk.inner(g_new))

            iks.fmt.log(
                f'line search {il}: '
                f'alpha={alpha:.5e}, '
                f'phi={phi_new:.5e}, '
                f'dphi={dphi_new:.5e}'
            )
            self.log_cvbar(x_new)

            iks.logiter(il, self.L1, L_new, self.LD1,
                        LD_new, g_new.data_sg, sep='')

            if not np.isfinite(phi_new) or not np.isfinite(dphi_new):
                # Treat a non-finite result as an excessively large step.
                alpha_hi = alpha

            else:
                armijo = (phi_new <= phi0 + c1 * alpha * dphi0)

                strong_curvature = (abs(dphi_new) <= c2 * abs(dphi0))

                if armijo and strong_curvature:
                    iks.fmt.log(
                        'Strong-Wolfe conditions satisfied: '
                        f'alpha={alpha:.5e}'
                    )

                    return (x_new.copy(), g_new.copy(), L_new, LD_new, alpha, True)

                if not armijo or (
                    alpha_lo > 0.0 and phi_new >= phi_lo
                ):
                    # The step is too large or the objective has started increasing
                    alpha_hi = alpha

                elif dphi_new >= 0.0:
                    # The derivative crossed zero
                    alpha_hi = alpha

                else:
                    # Armijo passed, but the derivative is still too negative.
                    alpha_lo = alpha
                    phi_lo = phi_new

            if alpha_hi is None:
                # No upper bracket yet. Expand the trial step.
                alpha_new = min(alpha / tau, alpha_max)

                iks.fmt.log(
                    '  Strong-Wolfe expansion: '
                    f'{alpha:.5e} -> {alpha_new:.5e}'
                )
            else:
                # A bracket exists. Zoom by bisection.
                alpha_new = 0.5 * (alpha_lo + alpha_hi)

                iks.fmt.log(
                    '  Strong-Wolfe zoom: '
                    f'[{alpha_lo:.5e}, {alpha_hi:.5e}] '
                    f'-> {alpha_new:.5e}'
                )

            # avoid repeating the same step
            if abs(alpha_new - alpha) <= (
                    1e-14 * max(1.0, abs(alpha))):
                iks.fmt.log(
                    'Strong-Wolfe line search stagnated: '
                    f'alpha={alpha:.5e}')
                break

            alpha = alpha_new

        iks.fmt.log('Strong-Wolfe line search failed; restoring current state.')

        self.set_potentials_and_iterate(x)

        return (x.copy(), gk.copy(), L0, LD0, 0.0, False)

    def log_cvbar(self, x):
        iks = self.iks
        atomdist = iks.hamiltonian.atomdist
        cvbar = atomdist.work_partition.arraydict(shapes=lambda a: (1,))
        cvbar.update({a: x.cvbar_a[a].copy() for a in x.cvbar_a})
        cvbar = atomdist.from_work(cvbar)
        cvbar.redistribute(cvbar.partition.as_serial())
        iks.fmt.log(f"cvbar: {dict(cvbar)}")

    def line_search_armijo_only(self, x: PotState, gk: PotState, pk: PotState,
                                L0: float, LD0: float, phi0: float, dphi0: float,
                                c1: float = 1e-4, c2: float = 0.9, tau: float = 0.5,
                                max_ls: int = 15) -> tuple[PotState, PotState, float, float, float, bool]:
        iks = self.iks
        alpha = 1.0
        accepted = False
        best_phi = phi0
        best_state = None
        g_new = gk
        L_new = L0
        LD_new = LD0

        for il in range(max_ls):
            x_new = x + alpha * pk
            self.set_potentials_and_iterate(x_new)

            self.L1, L_new, self.LD1, LD_new = iks.lagrangian()
            phi_new = L_new + LD_new

            J_xasz = iks.atomic_jacobian()
            JT_xasz = self.transform_gradients(J_xasz)
            J_sg = iks.jacobian()
            Jwork_xasz = [iks.hamiltonian.atomdist.to_work(
                J_asz) for J_asz in JT_xasz]
            Jc_a = self.get_cvbar_grad(
                J_sg, iks.hamiltonian.atomdist.to_work(J_xasz[1]))
            g_new = PotState(J_sg, Jwork_xasz, Jc_a, *self.potstate_args)

            iks.fmt.log(f"line search {il}")
            iks.logiter(il, self.L1, L_new, self.LD1,
                        LD_new, g_new.data_sg, sep="")

            if phi_new < best_phi:
                best_phi = phi_new
                x_best = x_new.copy()
                g_best = g_new.copy()
                best_state = (x_best, g_best, L_new, LD_new, alpha)

            if phi_new <= phi0 + c1 * alpha * dphi0:
                dphi_new = pk.inner(g_new)
                if dphi_new >= c2 * dphi0:
                    accepted = True
                    break
                else:
                    self.iks.fmt.log(
                        "  Warning: Armijo passed but Wolfe curvature failed.")
                    accepted = True
                    break

            denom = 2.0 * (phi_new - phi0 - dphi0 * alpha)
            if denom > 0:
                alpha_quad = - (dphi0 * alpha**2) / denom
                alpha_new = max(0.1 * alpha, min(0.9 * alpha, alpha_quad))
                iks.fmt.log(
                    f"  iKSLBFGS LS Interpolation: {alpha:.5e} -> {alpha_new:.5e}")
                alpha = alpha_new
            else:
                alpha *= tau

        if accepted:
            return x_new.copy(), g_new.copy(), L_new, LD_new, alpha, True

        if best_state is not None and best_phi < phi0:
            self.iks.fmt.log(
                "L-BFGS line search: Armijo failed, using best lagrangian step.")
            x_best, g_best, L_best, LD_best, alpha_best = best_state
            self.set_potentials_and_iterate(x_best)
            return x_best, g_best, L_best, LD_best, alpha_best, True

        self.iks.fmt.log(
            "Warning: L-BFGS line search: no step reduced Lagrangian; rejecting direction.")

        # restore
        self.set_potentials_and_iterate(x)
        return x.copy(), gk, L0, LD0, 0.0, False

    def run(self) -> list[float]:
        iks = self.iks

        if not iks.initialized:
            iks.initialize()

        opt_kwargs = {
            "memory": self.m,
            "max_ls": self.max_ls,
            "maxiter": self.maxiter,
            "step_scale": self.step_scale,
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
        J_xasz = iks.atomic_jacobian()
        JT_xasz = self.transform_gradients(J_xasz)
        J_sg = iks.jacobian()
        Jwork_xasz = []
        atomdist = iks.hamiltonian.atomdist
        for J_asz in JT_xasz:
            Jwork_xasz.append(atomdist.to_work(J_asz))
        Jc_a = self.get_cvbar_grad(J_sg, atomdist.to_work(J_xasz[1]))
        gk = PotState(J_sg, Jwork_xasz, Jc_a, *self.potstate_args).copy()
        # x = PotState(iks.xc.vxc_sg, iks.xc.vxc_xasz, *self.potstate_args).copy()
        xAE_asz = {}
        xPS_asz = {}
        c_a = {}
        for a in iks.xc.vxc_xasz[0].keys():
            if iks.setups[a].xc_correction is None:
                continue
            v_AE = iks.xc.vxc_xasz[0][a]
            v_PS = iks.xc.vxc_xasz[1][a]
            xPS_asz[a] = v_PS.copy()
            E_g = iks.sw_ag[a]
            x_AE_init = v_PS.copy()
            mask = E_g > 1e-8
            x_AE_init[..., mask] = v_PS[..., mask] + \
                (v_AE[..., mask] - v_PS[..., mask]) / E_g[mask]
            xAE_asz[a] = x_AE_init
            c_a[a] = np.zeros(1)

        x = PotState(iks.xc.vxc_sg, [xAE_asz, xPS_asz], c_a,
                     *self.potstate_args).copy()
        x_prev = None
        g_prev = None

        for t in range(1, self.maxiter + 1):
            iks.niter = t
            val_core, reset_core = iks.coresched.step(t)
            val_reg, reset_reg = iks.regsched.step(t)
            if reset_core or reset_reg:
                iks.lambdacore = val_core
                iks.lambdareg = val_reg
                iks.fmt.log(f'Regularization scheduler: '
                            f'lambdareg={val_reg:.5e}, lambdacore={val_core:.5e}')
                # reset the history right after the regularization has been turned off
                self.s_hist.clear()
                self.y_hist.clear()
                self.rho_hist.clear()

                # Reevaluate the current state under the new objective.
                self.set_potentials_and_iterate(x)
                self.L1, L, self.LD1, LD = iks.lagrangian()
                phi = L + LD

                J_xasz = iks.atomic_jacobian()
                JT_xasz = self.transform_gradients(J_xasz)
                J_sg = iks.jacobian()
                atomdist = iks.hamiltonian.atomdist
                Jwork_xasz = [atomdist.to_work(J_asz) for J_asz in JT_xasz]
                Jc_a = self.get_cvbar_grad(J_sg, atomdist.to_work(J_xasz[1]))

                gk = PotState(
                    J_sg,
                    Jwork_xasz,
                    Jc_a,
                    *self.potstate_args).copy()

            # convergence check with current (L, LD, gk) at x
            if iks.converged(L, LD, gk.data_sg, self.Ltol, self.LDtol, self.Jtol):
                iks.fmt.log("OECO-iKS L-BFGS converged!")
                break

            #  update L-BFGS history using previous step
            if x_prev is not None and g_prev is not None:
                s_k = x - x_prev
                y_k = gk - g_prev

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
            dphi0 = pk.inner(gk)
            norm_p = pk.norm()
            norm_g = gk.norm()
            cosine = -pk.inner(gk) / max(norm_p * norm_g, 1e-16)
            if cosine < 1e-4:
                # direction is almost orthogonal to gradient -> reset history
                self.s_hist.clear()
                self.y_hist.clear()
                self.rho_hist.clear()
                pk = -gk * self.step_scale
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
        return self.iks.L_iter


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

    @property
    def potstate_args(self):
        iks = self.iks
        return (iks.gd, iks.hamiltonian.world, iks.setups)

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
        gk = PotState(J_sg, J_xasz, *self.potstate_args)
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


class StaircaseScheduler:
    def __init__(
        self,
        start_val: float,
        decay_rate: float = 0.5,
        min_val: float = 1e-8,
        start_iter: int = 0,
        step_size: int = 50,
        n_regularization: int | None = None
    ):
        self.start_val = start_val
        self.decay_rate = decay_rate
        self.min_val = min(start_val, min_val)
        self.start_iter = start_iter
        self.step_size = step_size
        self.n_regularization = n_regularization
        self.current_val = start_val

    def todict(self):
        return dict(
            start_val=self.start_val,
            decay_rate=self.decay_rate,
            min_val=self.min_val,
            start_iter=self.start_iter,
            step_size=self.step_size,
            n_regularization=self.n_regularization,
            current_val=self.current_val,
        )

    def step(self, current_iter: int) -> tuple[float, bool]:
        if self.n_regularization is not None and current_iter > self.n_regularization:
            self.current_val = 0.0
            return 0.0, False
        if self.current_val == 0.0:
            return 0.0, False
        if current_iter <= self.start_iter:
            return self.start_val, False
        is_drop_step = (current_iter - self.start_iter) % self.step_size == 0
        if is_drop_step:
            new_val = max(self.min_val, self.current_val * self.decay_rate)
            if new_val < self.current_val:
                self.current_val = new_val
                return self.current_val, True
        return self.current_val, False
