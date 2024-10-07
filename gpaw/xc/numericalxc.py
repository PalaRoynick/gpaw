from __future__ import annotations

from math import pi, sqrt
from typing import Dict, Union, Optional

import numpy as np
from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.fd_operators import Gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.new import trace
from gpaw.typing import Array3D, ArrayND
from gpaw.xc.lda import LDA

# from gpaw.arraydict import ArrayDict
# from gpaw import debug
from gpaw.lfc import LocalizedFunctionsCollection

# from gpaw.setup import Setup
debug = True


class NumericalRadialExpansion:
    def __init__(self, rcalc, collinear=True):
        self.rcalc = rcalc
        self.collinear = collinear

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg):
        n_sLg = np.dot(D_sLq, n_qg)
        if self.collinear:
            n_sLg[:, 0] += nc0_sg
        else:
            n_sLg[0, 0] += 4 * nc0_sg[0]
        s, L, q = D_sLq.shape
        dEdD_sqL = np.zeros((s, q, L), dtype=D_sLq.dtype)
        self.rcalc.kernel.rgd = rgd
        Lmax = n_sLg.shape[1]
        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]
            self.rcalc.kernel.n = n
            e_g, dedn_sg = self.rcalc(rgd, n_sLg, Y_L)
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg, n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            E += w * rgd.integrate(e_g)
        return E, dEdD_sqL


@trace
def calculate_paw_correction_numerical(
    expansion, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
):
    xcc = setup.xc_correction
    if xcc is None:
        return 0.0

    rgd = xcc.rgd
    nspins = len(D_sp)

    if addcoredensity:
        nc0_sg = rgd.empty(nspins)
        nct0_sg = rgd.empty(nspins)
        nc0_sg[:] = sqrt(4 * pi) / nspins * xcc.nc_g
        nct0_sg[:] = sqrt(4 * pi) / nspins * xcc.nct_g
        if xcc.nc_corehole_g is not None and nspins == 2:
            nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
            nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
    else:
        nc0_sg = 0
        nct0_sg = 0

    D_sLq = np.inner(D_sp, xcc.B_pqL.T)
    # set state to real
    expansion.rcalc.kernel.paw_state = "R"
    e, dEdD_sqL = expansion(rgd, D_sLq, xcc.n_qg, nc0_sg)
    # set state to pseudo
    expansion.rcalc.kernel.paw_state = "T"
    et, dEtdD_sqL = expansion(rgd, D_sLq, xcc.nt_qg, nct0_sg)

    if dEdD_sp is not None:
        dEdD_sp += np.inner(
            (dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
            xcc.B_pqL.reshape((len(xcc.B_pqL), -1)),
        )

    if addcoredensity:
        return e - et - xcc.e_xc0
    else:
        return e - et


class NumericalRadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, rgd, n_sLg, Y_L):
        nspins = len(n_sLg)
        n_sg = np.dot(Y_L, n_sLg)
        e_g = rgd.empty()
        dedn_sg = rgd.zeros(nspins)
        self.kernel.calculate(e_g, n_sg, dedn_sg)
        return e_g, dedn_sg


class NumericalXCKernel:
    def __init__(
        self,
        name: str = "Numerical",
    ):
        self.name = name
        self.type = "LDA"
        self.vxc_sg: Union[ArrayND, None] = None
        self.gd = None
        self.grad_v = None
        self.r_vg = None
        self.rgd = None
        self.a = 0
        self.vxc_asg = None
        self.n = 0
        self.vxc_ansg = None
        self.vxct_ansg = None
        self.paw_state = "R"

    def set_grid_descriptor(self, gd: GridDescriptor, nn: int) -> None:
        self.gd = gd
        self.grad_v = [Gradient(gd, v, n=nn, xp=np).apply for v in range(3)]
        self.r_vg = gd.get_grid_point_coordinates()

    def set_rgd(self, rgd: RadialGridDescriptor, a: Optional[int] = None) -> None:
        self.rgd = rgd
        if a is not None:
            self.a = a

    def update_vxc(self, vxc_sg: ArrayND) -> None:
        if self.vxc_sg is not None:
            self.vxc_sg[:] = vxc_sg
        else:
            self.vxc_sg = np.copy(vxc_sg)

    def calculate(
        self,
        e_g,
        n_sg,
        v_sg,
        sigma_xg=None,
        dedsigma_xg=None,
        tau_sg=None,
        dedtau_sg=None,
    ) -> None:
        if len(n_sg.shape[1:]) == 1:
            self.rcalc(e_g, n_sg, v_sg)
        else:
            if not np.any(self.vxc_sg):
                raise ValueError("vxc_sg is not set")
            self._calculate(e_g, n_sg, v_sg)

    def _calculate(self, e_g: ArrayND, n_sg: ArrayND, v_sg: ArrayND) -> None:
        v_sg += self.vxc_sg
        gradv_svg = self.gd.empty((len(v_sg), 3))
        for gradv_vg, v_g in zip(gradv_svg, self.vxc_sg):
            for v in range(3):
                self.grad_v[v](v_g, gradv_vg[v])
        e_g[:] -= np.sum(
            n_sg * np.einsum("vxyz,svxyz->sxyz", self.r_vg, gradv_svg), axis=0
        )

    def update_vxc_a(self, vxc_ansg: Dict[int, Array3D]) -> None:
        if self.vxc_as is None:
            self.vxc_ansg = {a: vxc_sg for a, vxc_sg in vxc_ansg.items()}
        else:
            for a, vxc_sg in vxc_ansg.items():
                self.vxc_ansg[a][:] = vxc_sg

    def update_vxct_a(self, vxct_ansg: Dict[int, Array3D]) -> None:
        if self.vxct_asng is None:
            self.vxct_ansg = {a: vxct_nsg for a, vxct_nsg in vxct_ansg.items()}
        else:
            for a, vxct_nsg in vxct_ansg.items():
                self.vxct_ansg[a][:] = vxct_nsg

    def rcalc(self, e_g: ArrayND, n_sg: ArrayND, v_sg: ArrayND) -> None:
        if self.paw_state == "R":
            vxc_ansg = self.vxc_ansg
        elif self.paw_state == "T":
            vxc_ansg = self.vxct_ansg
        else:
            raise ValueError('paw_state must be either "R" or "T"')
        if vxc_ansg is None:
            if self.vxc_asg is None:
                return
            vxc_ansg = {self.a: self.vxc_asg[self.a][np.newaxis]}
        assert n_sg.shape == vxc_ansg[self.a][self.n].shape
        v_sg[:] += vxc_ansg[self.a][self.n]
        dvdr_sg = self.rgd.zeros((len(v_sg),))
        for v_g, dvdr_g in zip(vxc_ansg[self.a][self.n], dvdr_sg):
            self.rgd.derivative(v_g, dvdr_g)
        e_g[:] -= np.sum(n_sg * self.rgd.r_g[np.newaxis] * dvdr_sg, axis=0)


class NumericalXC(LDA):
    def __init__(self, kernel: NumericalXCKernel, stencil: int = 3):
        self.stencil = stencil
        super().__init__(kernel)

    @property
    def vxc_sg(self):
        return self.kernel.vxc_sg

    @property
    def vxc_asg(self):
        return self.kernel.vxc_asg

    @property
    def vxc_ansg(self):
        return self.kernel.vxc_ansg

    @property
    def vxct_ansg(self):
        return self.kernel.vxc_ansg

    def set_grid_descriptor(self, gd: GridDescriptor):
        self.gd = gd
        self.kernel.set_grid_descriptor(gd, self.stencil)

    def set_rgd(self, rgd: RadialGridDescriptor, a: int = 0):
        self.kernel.set_rgd(rgd, a)

    def update_vxc(self, vxc_sg: ArrayND) -> None:
        self.kernel.update_vxc(vxc_sg)

    def update_vxc_a(self, vxc_ansg: Dict[int, Array3D]) -> None:
        self.kernel.update_vxc_a(vxc_ansg)

    def update_vxct_a(self, vxct_ansg: Dict[int, Array3D]) -> None:
        self.kernel.update_vxct_a(vxct_ansg)

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        xcc = setup.xc_correction
        if debug:
            if xcc is None:
                print("No XC correction")
        if xcc is not None:
            self.kernel.set_rgd(xcc.rgd, a or 0)
        rcalc = NumericalRadialCalculator(self.kernel)
        expansion = NumericalRadialExpansion(rcalc, True)
        return calculate_paw_correction_numerical(
            expansion, setup, D_sp, dEdD_sp, addcoredensity, a
        )

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None) -> float:
        self.kernel.set_rgd(rgd, 0)
        self.kernel.n = 0
        return super().calculate_spherical(rgd, n_sg, v_sg, e_g)
