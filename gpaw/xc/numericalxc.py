from __future__ import annotations

from math import pi, sqrt
from typing import List, Optional

import numpy as np
from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.fd_operators import Gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.typing import ArrayND
from gpaw.xc.lda import LDA

# from gpaw import debug
debug = True


class NXCRadialCalculator:
    def __init__(self, vxc_szg, use_lebedev=True):
        self.vxc_szg = vxc_szg
        self.use_lebedev = use_lebedev
        self.n = 0

    def __call__(self, rgd, n_sLg, Y_L):
        nspins = len(n_sLg)
        n_sg = np.dot(Y_L, n_sLg)
        if self.use_lebedev:
            dedn_sg = self.vxc_szg[:, self.n]
        else:
            dedn_sg = np.dot(Y_L, self.vxc_szg)
        dvdr_sg = rgd.zeros((nspins,))
        # virial theorem
        for v_g, dvdr_g in zip(dedn_sg, dvdr_sg):
            rgd.derivative(v_g, dvdr_g)
        # TODO: The energy now has terms including the gradient of spherical harmonics?
        # This needs overhaul
        e_g = np.sum(n_sg * rgd.r_g[np.newaxis] * dvdr_sg, axis=0)
        return e_g, dedn_sg


class NXCRadialExpansion:
    def __init__(self, rcalc, collinear=True):
        self.rcalc = rcalc
        self.collinear = collinear

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg):
        n_sLg = np.dot(D_sLq, n_qg)
        if self.collinear:
            n_sLg[:, 0] += nc0_sg
        else:
            n_sLg[0, 0] += 4 * nc0_sg[0]

        dEdD_sqL = np.zeros_like(np.transpose(D_sLq, (0, 2, 1)))

        Lmax = n_sLg.shape[1]
        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            self.rcalc.n = n
            w = weight_n[n]
            e_g, dedn_sg = self.rcalc(rgd, n_sLg, Y_L)
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg, n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            E += w * rgd.integrate(e_g)
        return E, dEdD_sqL


class NumericalXCKernel:
    def __init__(
        self,
        name: str = "Numerical",
        vxc_sg: Optional[ArrayND] = None,
        vxc_aszg: Optional[ArrayND] = None,
        vxct_aszg: Optional[ArrayND] = None,
        dEdD_sqL: Optional[ArrayND] = None,
        use_lebedev: bool = True,
    ):
        self.name = name
        self.type = "LDA"
        self.gd: GridDescriptor
        self.grad_v: List[Gradient]
        self.r_vg: ArrayND
        self.rgd: RadialGridDescriptor
        self.a: int = 0
        self.vxc_sg = vxc_sg
        self.vxc_aszg = vxc_aszg
        self.vxct_aszg = vxct_aszg
        self.dEdD_sqL = dEdD_sqL
        self.use_lebedev = use_lebedev

    def set_grid_descriptor(self, gd: GridDescriptor, nn: int = 2) -> None:
        self.gd = gd
        self.grad_v = [Gradient(gd, v, n=nn, xp=np).apply for v in range(3)]
        self.origin = 0.5 * np.linalg.norm(gd.cell_cv, axis=0)
        self.r_vg = gd.get_grid_point_distance_vectors(self.origin)

    def set_rgd(self, rgd: RadialGridDescriptor, a: Optional[int] = None) -> None:
        self.rgd = rgd
        if a is not None:
            self.a = a

    def calculate(
        self,
        e_g,
        n_sg,
        v_sg,
        sigma_xg=None,
        dedsigma_xg=None,
        tau_sg=None,
        dedtau_sg=None,
    ):
        if len(n_sg.shape[1:]) == 1:
            self.rcalc(e_g, n_sg, v_sg)
        else:
            assert self.vxc_sg is not None
            vxc_sg = self.vxc_sg
            grad_v = self.grad_v
            r_vg = self.r_vg
            v_sg[:] += vxc_sg
            gradv_svg = self.gd.zeros((len(vxc_sg), 3))
            # virial theorem
            for gradv_vg, v_g in zip(gradv_svg, vxc_sg):
                for v in range(3):
                    grad_v[v](v_g, gradv_vg[v])
            e_g[:] = 0.0
            for s in range(len(n_sg)):
                e_g[:] -= n_sg[s] * np.einsum(
                    "v...,v...->...",
                    r_vg,
                    gradv_svg[s],
                )

    def rcalc(self, e_g: ArrayND, n_sg: ArrayND, v_sg: ArrayND, z: int = 0) -> None:
        # for calculate_spherical method
        if self.vxc_aszg is None:
            raise ValueError("No atomic XC potential available!")
        vxc_sg = self.vxc_aszg[self.a][z]
        v_sg[:] += vxc_sg
        dvdr_sg = self.rgd.zeros((len(v_sg),))
        for v_g, dvdr_g in zip(vxc_sg, dvdr_sg):
            self.rgd.derivative(v_g, dvdr_g)
        e_g[:] = -np.sum(n_sg * self.rgd.r_g[np.newaxis] * dvdr_sg, axis=0)


class NumericalXC(LDA):
    def __init__(self, kernel: NumericalXCKernel, stencil: int = 3):
        self.stencil = stencil
        super().__init__(kernel)

    def set_grid_descriptor(self, gd: GridDescriptor):
        self.kernel.set_grid_descriptor(gd, self.stencil)

    def _calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=0
    ):
        xcc = setup.xc_correction
        if xcc is None:
            return 0.0

        rcalc = NXCRadialCalculator(self.kernel.vxc_aszg[a], self.kernel.use_lebedev)
        expansion = NXCRadialExpansion(rcalc)
        rcalct = NXCRadialCalculator(self.kernel.vxct_aszg[a], self.kernel.use_lebedev)
        expansiont = NXCRadialExpansion(rcalct)
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

        e, dEdD_sqL = expansion(rgd, D_sLq, xcc.n_qg, nc0_sg)
        et, dEtdD_sqL = expansiont(rgd, D_sLq, xcc.nt_qg, nct0_sg)
        if dEdD_sp is not None:
            dEdD_sp += np.inner(
                (dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
                xcc.B_pqL.reshape((len(xcc.B_pqL), -1)),
            )
        if addcoredensity:
            return e - et - xcc.e_xc0
        else:
            return e - et

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        if a is None or self.kernel.vxc_aszg is None or self.kernel.vxct_aszg is None:
            print("No atomic XC potential available!")
            return 0.0
        return self._calculate_paw_correction(setup, D_sp, dEdD_sp, addcoredensity, a)

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None) -> float:
        self.kernel.set_rgd(rgd, 0)
        return super().calculate_spherical(rgd, n_sg, v_sg, e_g)
