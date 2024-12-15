from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.fd_operators import Gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import BasisFunctions
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.typing import ArrayND
from gpaw.utilities import pack_density, unpack_density
from gpaw.utilities.tools import tri2full
from gpaw.xc.lda import LDA

# from gpaw import debug
debug = True


class NumericalXCKernel:
    STATE_PSEUDO = "pseudo"
    STATE_PAWTRUE = "true_paw"
    STATE_PAWPSEUDO = "pseudo_paw"

    def __init__(
        self,
        name: str = "Numerical",
    ):
        self.name = name
        self.type = "LDA"
        self.vxc_sg: Optional[ArrayND] = None
        self.gd: Optional[GridDescriptor] = None
        self.grad_v: List[Gradient] = None
        self.r_vg: Optional[ArrayND] = None
        self.rgd: Optional[RadialGridDescriptor] = None
        self.a: Optional[int] = 0
        self.vxc_asG: Optional[ArrayND] = None
        self.vxct_asG: Optional[ArrayND] = None
        self.state: str = self.STATE_PSEUDO

    def set_grid_descriptor(self, gd: GridDescriptor, nn: int = 2) -> None:
        self.gd = gd
        self.grad_v = [Gradient(gd, v, n=nn, xp=np).apply for v in range(3)]
        self.r_vg = gd.get_grid_point_coordinates()

    def set_grid_descriptor_coarse(self, cgd: GridDescriptor, nn: int = 2) -> None:
        self.cgd = cgd
        self.cgrad_v = [Gradient(cgd, v, n=nn, xp=np).apply for v in range(3)]
        self.r_vG = cgd.get_grid_point_coordinates()

    def set_rgd(self, rgd: RadialGridDescriptor, a: Optional[int] = None) -> None:
        self.rgd = rgd
        if a is not None:
            self.a = a

    def update_vxc(self, vxc_sg: ArrayND) -> None:
        self.vxc_sg = vxc_sg

    def update_vxc_a(self, vxc_asG: ArrayND) -> None:
        self.vxc_asG = vxc_asG

    def update_vxct_a(self, vxct_asG: ArrayND) -> None:
        self.vxct_asG = vxct_asG

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
            self.state = self.STATE_PAWTRUE
            self.rcalc(e_g, n_sg, v_sg)
        else:
            assert self.vxc_sg is not None
            self.state = self.STATE_PSEUDO
            self._calculate(self.gd, e_g, n_sg, v_sg, self.state)

    def _calculate(
        self,
        gd: GridDescriptor,
        e_g: ArrayND,
        n_sg: ArrayND,
        v_sg: Optional[ArrayND] = None,
        state: str = STATE_PSEUDO,
    ) -> None:
        if state == self.STATE_PSEUDO:
            vxc_sg = self.vxc_sg
            grad_v = self.grad_v
            r_vg = self.r_vg
        elif state == self.STATE_PAWTRUE:
            vxc_sg = self.vxc_asG[self.a]
            grad_v = self.cgrad_v
            r_vg = self.r_vG
        elif state == self.STATE_PAWPSEUDO:
            vxc_sg = self.vxct_asG[self.a]
            grad_v = self.cgrad_v
            r_vg = self.r_vG
        else:
            raise ValueError("Invalid state!")
        if v_sg is not None:
            v_sg[:] += vxc_sg
        gradv_svg = gd.zeros((len(vxc_sg), 3))
        # virial theorem
        for gradv_vg, v_g in zip(gradv_svg, vxc_sg):
            for v in range(3):
                grad_v[v](v_g, gradv_vg[v])
        e_g[:] -= np.sum(n_sg * np.einsum("v...,sv...->s...", r_vg, gradv_svg), axis=0)

    def rcalc(self, e_g: ArrayND, n_sg: ArrayND, v_sg: ArrayND) -> None:
        if self.state == self.STATE_PAWTRUE or self.state == self.STATE_PSEUDO:
            vxc_asg = self.vxc_asG
        elif self.state == self.STATE_PAWPSEUDO:
            vxc_asg = self.vxct_asG
        else:
            raise ValueError("Invalid state!")

        v_sg[:] += vxc_asg[self.a]
        dvdr_sg = self.rgd.zeros((len(v_sg),))
        for v_g, dvdr_g in zip(vxc_asg[self.a], dvdr_sg):
            self.rgd.derivative(v_g, dvdr_g)
        e_g[:] -= np.sum(n_sg * self.rgd.r_g[np.newaxis] * dvdr_sg, axis=0)


class NumericalXC(LDA):
    def __init__(self, kernel: NumericalXCKernel, stencil: int = 3):
        self.stencil = stencil
        self.kernel: NumericalXCKernel = kernel
        super().__init__(kernel)
        self.nc: None | LFC = None
        self.nct: None | LFC = None
        self.phi: None | BasisFunctions = None
        self.phit: None | BasisFunctions = None

    def set_grid_descriptor(self, gd: GridDescriptor):
        self.kernel.set_grid_descriptor(gd, self.stencil)

    def set_lfc(
        self,
        phi: BasisFunctions,
        phit: BasisFunctions,
        nc: LFC,
        nct: LFC,
        nspins: int,
    ):
        self.phi = phi
        self.phit = phit
        self.nc = nc
        self.nct = nct
        self.nspins = nspins
        self.c_asi = {
            a: np.zeros((self.nspins, nc.get_function_count(a)))
            for a in range(len(nc.sphere_a))
        }
        self.kernel.set_grid_descriptor_coarse(phi.gd, self.stencil)

    def _calculate_paw_correction(self, setup, D_sp, dEdD_sp, a):
        M1 = self.phi.M_a[a]
        M2 = M1 + self.phi.sphere_a[a].Mmax
        nspins = len(D_sp)
        na_sG = self.kernel.cgd.zeros(nspins)
        nat_sG = self.kernel.cgd.zeros(nspins)
        for s, D_p in enumerate(D_sp):
            rho_MM = np.zeros((self.phi.Mmax, self.phi.Mmax))
            rho_MM[M1:M2, M1:M2] = unpack_density(D_p)
            self.phi.construct_density(rho_MM, na_sG[s], -1)
            self.phit.construct_density(rho_MM, nat_sG[s], -1)
            if dEdD_sp is not None:
                # calculate atomic xc correction to the Hamiltonian
                self.phi.calculate_potential_matrix(
                    self.kernel.vxc_asG[a, s], rho_MM, -1
                )
                # make matrix hermitian since only lower triangle is calculated
                tri2full(rho_MM)
                dEdD_sp[s] += pack_density(rho_MM[M1:M2, M1:M2])
                self.phit.calculate_potential_matrix(
                    self.kernel.vxct_asG[a, s], rho_MM, -1
                )
                tri2full(rho_MM)
                dEdD_sp[s] -= pack_density(rho_MM[M1:M2, M1:M2])
        # add core density
        c_asi = self.c_asi.copy()
        c_asi[a] = np.ones_like(c_asi[a])
        self.nc.add(na_sG, c_asi)
        self.nct.add(nat_sG, c_asi)
        # calculate energy
        e_G = self.kernel.cgd.zeros()
        self.kernel.state = self.kernel.STATE_PAWTRUE
        self.kernel._calculate(self.kernel.cgd, e_G, na_sG, None, self.kernel.STATE_PAWTRUE)
        self.kernel.state = self.kernel.STATE_PAWPSEUDO
        self.kernel._calculate(self.kernel.cgd, e_G, nat_sG, None, self.kernel.STATE_PAWPSEUDO)
        return self.kernel.cgd.integrate(e_G)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        if a is None or self.kernel.vxc_asG is None or self.kernel.vxct_asG is None:
            print("FLJFFFFFFFFFFFFFFFFFFFKSJSALK")
            return 0.0
        else:
            self.kernel.a = a
            E = self._calculate_paw_correction(setup, D_sp, dEdD_sp, a)
            return E

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None) -> float:
        self.kernel.set_rgd(rgd, 0)
        return super().calculate_spherical(rgd, n_sg, v_sg, e_g)
