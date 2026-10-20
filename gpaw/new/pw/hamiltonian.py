from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

from ase.units import Ha, Bohr

from gpaw.core.arrays import XArray
from gpaw.core.plane_waves import PWArray
from gpaw.core.uniform_grid import UGArray
from gpaw.gpu import cupy as cp
from gpaw.new import trace, zips
from gpaw.new.c import pw_precond
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.utilities.tools import cutoff2gridspacing


class PWHamiltonian(Hamiltonian):
    def __init__(self, grid, dtype, xp=np):
        self.blocksize = 2 if xp is np else 8
        self.grid = grid
        self.grid_local = grid.new(comm=None, dtype=dtype)

        self.plan = self.grid_local.fft_plans(xp=xp,
                                              blocksize=self.blocksize)
        # It's a bit too expensive to create all the local PW-descriptors
        # for all the k-points every time we apply the Hamiltonian, so we
        # cache them:
        self.pw_cache = {}

    @trace
    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nG: XArray,
                              out: XArray) -> None:
        assert isinstance(psit_nG, PWArray)
        assert isinstance(out, PWArray)
        out_nG = out
        xp = psit_nG.xp
        pw = psit_nG.desc
        vt_R = vt_R.gather(broadcast=True)

        plan = self.plan
        tmp_R = plan.tmp_R  # self.grid_local.empty(xp=xp)
        tmp_Q = plan.tmp_Q
        if pw.comm.size == 1:
            pw_local = pw
        else:
            key = tuple(pw.kpt_c)
            pw_local = self.pw_cache.get(key)
            if pw_local is None:
                pw_local = pw.new(comm=None)
                self.pw_cache[key] = pw_local
        psit_xG = pw_local.empty(self.blocksize, xp=xp)
        e_kin_G = xp.asarray(pw_local.ekin_G)
        Q_G = pw_local.indices(tmp_Q.shape[-3:])

        # Apply the potential
        for y, psit_yG in psit_nG.iter_by_band(psit_xG, out_nG):
            if len(y) == 0:
                # This rank got no data
                continue

            if self.blocksize == 1:
                tmp_yR = tmp_R
                tmp_yQ = tmp_Q
                plan.ifft_sphere(psit_yG.data[0], pw_local)
            else:
                tmp_yR = tmp_R[:len(y)]
                tmp_yQ = tmp_Q[:len(y)]
                plan.ifft_sphere(psit_yG.data, pw_local)

            tmp_yR *= vt_R.data

            plan.fft()
            vtpsit_yG = tmp_yQ.reshape(len(y), -1)[:, Q_G]
            vtpsit_yG *= len(y) / tmp_yR.size
            psit_yG.data *= e_kin_G
            psit_yG.data += vtpsit_yG

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nG: XArray,
                   vt_nG: XArray) -> None:
        pw = psit_nG.desc
        xp = psit_nG.xp
        dpsit_R = dedtaut_R.desc.new(dtype=pw.dtype).empty(xp=xp)
        Gplusk1_Gv = pw.reciprocal_vectors(xp)
        tmp_G = pw.empty(xp=xp)

        for psit_G, vt_G in zips(psit_nG, vt_nG):
            for v in range(3):
                tmp_G.data[:] = psit_G.data
                tmp_G.data *= 1j * Gplusk1_Gv[:, v]
                tmp_G.ifft(out=dpsit_R)
                dpsit_R.data *= dedtaut_R.data
                dpsit_R.fft(out=tmp_G)
                vt_G.data -= 0.5j * Gplusk1_Gv[:, v] * tmp_G.data

    def create_preconditioner(self,
                              blocksize: int,
                              xp=np
                              ) -> Callable[[PWArray,
                                             PWArray,
                                             PWArray], None]:
        return precondition

    def calculate_kinetic_energy(self, wfs, skip_sum=False):
        e_kin = 0.0
        for f, psit_G in zip(wfs.myocc_n, wfs.psit_nX):
            if f > 1.0e-10:
                e_kin += f * psit_G.norm2('kinetic', skip_sum=skip_sum)
        if not skip_sum:
            e_kin = psit_G.desc.comm.sum_scalar(e_kin)
            e_kin = wfs.band_comm.sum_scalar(e_kin)
        return e_kin * wfs.spin_degeneracy


@trace
def precondition(psit_nG: PWArray,
                 residual_nG: PWArray,
                 out: PWArray,
                 ekin_n=None) -> None:
    """Preconditioner for KS equation.

    From:

      Teter, Payne and Allen, Phys. Rev. B 40, 12255 (1989)

    as modified by:

      Kresse and Furthmüller, Phys. Rev. B 54, 11169 (1996)
    """
    xp = psit_nG.xp
    G2_G = xp.asarray(psit_nG.desc.ekin_G * 2)
    if ekin_n is None:
        ekin_n = psit_nG.norm2('kinetic')

    if xp is np:
        for r_G, o_G, ekin in zips(residual_nG.data,
                                   out.data,
                                   ekin_n):
            pw_precond(G2_G, r_G, ekin, o_G)
    else:
        out.data[:] = gpu_prec(ekin_n[:, np.newaxis],
                               G2_G[np.newaxis],
                               residual_nG.data)
    return ekin_n


@trace(gpu=True)
@cp.fuse()
def gpu_prec(ekin, G2, residual):
    x = 1 / ekin / 3 * G2
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    return -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * residual


def spinor_precondition(psit_nsG, residual_nsG, out):
    G2_G = psit_nsG.desc.ekin_G * 2
    for r_sG, o_sG, ekin in zips(residual_nsG.data,
                                 out.data,
                                 psit_nsG.norm2('kinetic').sum(1)):
        for r_G, o_G in zips(r_sG, o_sG):
            pw_precond(G2_G, r_G, ekin, o_G)


class SpinorPWHamiltonian(Hamiltonian):
    def __init__(self, qspiral_v):
        super().__init__()
        self.qspiral_v = qspiral_v

    def apply(self,
              vt_xR: UGArray,
              dedtaut_xR: UGArray | None,
              ibzwfs,
              D_asii,
              psit_nsG: XArray,
              out: XArray,
              spin: int,
              calculate_energy: bool = False) -> XArray:
        assert dedtaut_xR is None
        out_nsG = out
        pw = psit_nsG.desc

        if self.qspiral_v is None:
            np.multiply(pw.ekin_G, psit_nsG.data, out_nsG.data)
        else:
            for s, sign in enumerate([1, -1]):
                ekin_G = 0.5 * ((pw.G_plus_k_Gv +
                                 0.5 * sign * self.qspiral_v)**2).sum(1)
                np.multiply(ekin_G, psit_nsG.data[:, s], out_nsG.data[:, s])

        grid = vt_xR.desc.new(dtype=complex)

        v, x, y, z = vt_xR.data
        iy = y * 1j

        f_sR = grid.empty(2)
        g_R = grid.empty()

        for p_sG, o_sG in zips(psit_nsG, out_nsG):
            p_sG.ifft(out=f_sR)
            a, b = f_sR.data
            g_R.data = a * (v + z) + b * (x - iy)
            o_sG.data[0] += g_R.fft(pw=pw).data
            g_R.data = a * (x + iy) + b * (v - z)
            o_sG.data[1] += g_R.fft(pw=pw).data

        return out_nsG

    def create_preconditioner(self, blocksize, xp):
        return spinor_precondition
