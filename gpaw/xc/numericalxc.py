from __future__ import annotations

import warnings
from enum import Enum
from math import pi, sqrt
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Ha
from gpaw.fd_operators import Gradient, Laplace
from gpaw.lfc import LFC, BasisFunctions
from gpaw.poisson import create_poisson_solver
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.utilities import divrl
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional
from gpaw.xc.pawcorrection import rnablaY_nLv
from gpaw.xc.gga import GGARadialCalculator

if TYPE_CHECKING:
    from typing import Any

    from gpaw.arraydict import ArrayDict
    from gpaw.atom.radialgd import RadialGridDescriptor
    from gpaw.density import Density
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.hamiltonian import Hamiltonian
    from gpaw.poisson import BasePoissonSolver
    from gpaw.setup import Setup, Setups
    from gpaw.typing import Array1D, Array2D, Array3D, Array4D, ArrayND
    from gpaw.wavefunctions.base import WaveFunctions


def dist_arraydict_data(v_asz: ArrayDict, data: Array1D) -> None:
    """writes the (broadcasted) data to the arraydict"""
    partition = v_asz.partition
    shapes_a = v_asz.shapes_a
    for a in partition.my_indices:
        M1 = sum([np.prod(shape) for shape in shapes_a[:a]])
        M2 = M1 + np.prod(shapes_a[a])
        v_asz[a].ravel()[:] = data[M1:M2]


def broadcast_arraydict_data(v_asz: ArrayDict) -> Array1D:
    atompart = v_asz.partition
    acomm = atompart.comm
    if acomm.size == 1:
        return v_asz.toarray()
    shapes_a = v_asz.shapes_a
    size = sum([np.prod(shape) for shape in shapes_a])
    data_asz = np.empty(size, dtype=float)
    for a in range(atompart.natoms):
        M1 = sum([np.prod(shape) for shape in shapes_a[:a]])
        M2 = M1 + np.prod(shapes_a[a])
        if acomm.rank == 0:
            if a in atompart.my_indices:
                data_asz[M1:M2] = v_asz[a].ravel()
            else:
                acomm.receive(data_asz[M1:M2], atompart.rank_a[a], tag=1213)
        elif a in atompart.my_indices:
            acomm.send(v_asz[a].ravel(), 0, tag=1213)
    acomm.broadcast(data_asz, 0)
    return data_asz


class PAWPotMode(Enum):
    lebedev = "lebedev"
    angular = "angular"
    direct = "direct"
    none = "none"


num_pawpots = {
    PAWPotMode.lebedev: 2,
    PAWPotMode.angular: 2,
    PAWPotMode.direct: 1,
    PAWPotMode.none: 0,
}


class NXCRadialCalculator:
    def __init__(self, vxc_szg: Array3D, mode: PAWPotMode, virial_xc_energy: bool):
        self.vxc_szg = vxc_szg
        self.mode = PAWPotMode(mode)
        assert self.mode is PAWPotMode.lebedev or self.mode is PAWPotMode.angular
        self.virial_xc_energy = virial_xc_energy
        self.n: int = 0

    def __call__(self, rgd: RadialGridDescriptor, n_sLg: Array3D, Y_L: Array1D):
        nspins = len(n_sLg)
        n_sg = np.dot(Y_L, n_sLg)
        dedn_sg = rgd.zeros(nspins)
        if self.mode is PAWPotMode.lebedev:
            dedn_sg[:] = self.vxc_szg[:, self.n]
        elif self.mode is PAWPotMode.angular:
            dedn_sg[:] = np.dot(Y_L, self.vxc_szg)
        if self.virial_xc_energy:
            dvdr_sg = rgd.zeros(nspins)
            # virial theorem
            for v_g, dvdr_g in zip(dedn_sg, dvdr_sg):
                rgd.derivative(v_g, dvdr_g)
            # The energy now has terms including the gradient of spherical harmonics?
            # This needs overhaul
            e_g = -np.sum(n_sg * rgd.r_g[np.newaxis] * dvdr_sg, axis=0)
        else:
            e_g = np.sum(n_sg * dedn_sg, axis=0)
        return e_g, dedn_sg

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        if e_g is None:
            e_g = rgd.zeros()
        e_g[:], v_sg[:] = self(rgd, n_sg[:, np.newaxis], [1.0])
        return rgd.integrate(e_g)


class NXCRadialExpansion:
    """
    same as the LDARadialExpansion but passing the lebedev point to
    calculate with the corresponding potential
    """

    def __init__(self, rcalc: NXCRadialCalculator, collinear: bool = True):
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
            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            E += w * rgd.integrate(e_g)
        return E, dEdD_sqL


def calculate_paw_correction(
    rcalc: NXCRadialCalculator,
    rcalct: NXCRadialCalculator,
    setup: Setup,
    D_sp: Array2D,
    dEdD_sp: Array2D | None,
    addcoredensity: bool,
    a: int = 0,
) -> float:
    """
    same as the LDA caculate_paw_correction with two separate expansions
    """
    xcc = setup.xc_correction
    if xcc is None:
        return 0.0

    expansion = NXCRadialExpansion(rcalc, True)
    expansiont = NXCRadialExpansion(rcalct, True)
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


class NumericalXCKernel:
    """only for the regular grid, for the radial case use NXCRadialCalculator and NXCRadialExpansion"""

    def __init__(
        self,
        name: str = "Numerical",
        vxc_sg: Array4D | None = None,
        virial_xc_energy: bool = False,
    ):
        self.name = name
        self.type = "NXC"
        self.vxc_sg = vxc_sg
        self.gd: GridDescriptor | None = None
        self.grad_v: list[Any] = []
        self.r_vg: ArrayND
        self.virial_xc_energy = virial_xc_energy

    def set_grid_descriptor(self, gd: GridDescriptor, nn: int = 2) -> None:
        self.grad_v = [
            Gradient(gd, v, n=nn, xp=np).apply for v in range(3)]
        self.lapl = Laplace(gd, n=nn + 1).apply
        self.origin = 0.5 * np.linalg.norm(gd.cell_cv, axis=0)
        self.r_vg = gd.get_grid_point_distance_vectors(self.origin)
        # self.r_vg = gd.get_grid_point_coordinates()
        self.gd = gd

    def calculate_gradient(self, vxc_sg):
        nspins = len(vxc_sg)
        gradv_svg = self.gd.zeros((nspins, 3))
        for gradv_vg, v_g in zip(gradv_svg, vxc_sg):
            for v in range(3):
                self.grad_v[v](v_g, gradv_vg[v])
        return gradv_svg

    def calculate_lapl(self, vxc_sg):
        lapl_sg = self.gd.zeros((len(vxc_sg)))
        self.lapl(vxc_sg, lapl_sg)
        return lapl_sg

    def calculate(self, e_g, n_sg, v_sg):
        assert self.vxc_sg is not None
        vxc_sg = self.vxc_sg
        v_sg[:] += vxc_sg
        if self.virial_xc_energy:
            # virial theorem
            r_vg = self.r_vg
            gradv_svg = self.calculate_gradient(self.vxc_sg)
            e_g[:] = 0.0
            for s in range(len(n_sg)):
                e_g[:] -= n_sg[s] * np.einsum(
                    "v...,v...->...",
                    r_vg,
                    gradv_svg[s],
                )
        else:
            e_g[:] = np.sum(n_sg * vxc_sg, axis=0)


class NumericalXC(XCFunctional):
    def __init__(
        self,
        kernel: NumericalXCKernel,
        setup_name: str = "Numerical",
        paw_potmode: str | PAWPotMode = PAWPotMode.lebedev,
        virial_xc_energy: bool = False,
        stencil: int = 3,
        vxc_sg: ArrayND | None = None,
        vxc_xasz: list[ArrayDict] | None = None,
        vc_xag: list[ArrayDict] = None,
        interpolate_vt=False,  # deprecated
    ):
        super().__init__(kernel.name, kernel.type)
        self.stencil = stencil
        self.kernel: NumericalXCKernel = kernel
        self.setup_name = setup_name
        self.vxc_xasz = vxc_xasz or []
        self.vc_xag = vc_xag
        self.paw_potmode = PAWPotMode(paw_potmode)
        self.virial_xc_energy = (
            False if virial_xc_energy == "False" else bool(virial_xc_energy)
        )
        self._vxc_sg = vxc_sg
        self.ham = None

    @property
    def vxc_sg(self):
        return self._vxc_sg

    @vxc_sg.setter
    def vxc_sg(self, _vxc_sg):
        self._vxc_sg = _vxc_sg
        self.kernel.vxc_sg = _vxc_sg

    @classmethod
    def from_other_xc(
        cls,
        xc: str | dict | XCFunctional,
        density: Density,
        paw_potmode: str | PAWPotMode = "lebedev",
        setup_name: None | str = None,
        virial_xc_energy: bool = False,
        stencil: int = 3,
        ham: Hamiltonian | None = None,
        wfs: WaveFunctions | None = None,
        add_core: bool = True,
    ) -> "NumericalXC":
        """
        Returns a NumericalXC object by evaluating the given functional over the density.
        paw_potmode ``lebedev`` and ``angular`` are only implemented for LDA and GGA.
        ``ham`` and ``wfs`` are only needed for MGGA and Hybrids.
        """
        paw_potmode = PAWPotMode(paw_potmode)
        if isinstance(xc, (str, dict)):
            xc = XC(xc)
        gd = density.finegd
        xc.set_grid_descriptor(gd)
        if (ham is not None) and (wfs is not None):
            # XXX add converge wfs for MGGA
            xc.initialize(density, ham, wfs)
            xc.set_positions(ham.spos_ac)
        nspins = density.nspins
        vxc_sg = gd.zeros(nspins)
        xc.calculate(gd, density.nt_sg, vxc_sg)
        D_asp = density.atomdist.to_work(density.D_asp)
        atompart = D_asp.partition
        setups = density.setups
        shapes_a = get_atomic_potential_shapes(setups, nspins, paw_potmode)
        vxc_xasz = [
            atompart.arraydict(shapes_a, dtype=float)
            for _ in range(num_pawpots[paw_potmode])
        ]

        for a, D_sp in D_asp.items():
            if paw_potmode is PAWPotMode.none:
                continue
            elif paw_potmode is PAWPotMode.direct:
                dEdD_sp = vxc_xasz[0][a]
                dEdD_sp[:] = 0.0
                xc.calculate_paw_correction(setups[a], D_sp, dEdD_sp, a=a)
            else:
                xcc = setups[a].xc_correction
                if xcc is None:
                    continue
                # angular and lebedev
                rgd = xcc.rgd
                n_sLg = calculate_angular_density(
                    xcc, D_sp, xcc.n_qg, xcc.nc_g, add_core)
                nt_sLg = calculate_angular_density(
                    xcc, D_sp, xcc.nt_qg, xcc.nct_g, add_core)
                vxc_xasz[0][a] = calculate_atomic_potentials_from_kernel(
                    rgd, xc.kernel, n_sLg, paw_potmode,
                )
                vxc_xasz[1][a] = calculate_atomic_potentials_from_kernel(
                    rgd, xc.kernel, nt_sLg, paw_potmode,
                )

        return cls(
            kernel=NumericalXCKernel(virial_xc_energy=virial_xc_energy),
            setup_name=setup_name or xc.get_setup_name(),
            paw_potmode=paw_potmode,
            stencil=stencil,
            vxc_sg=vxc_sg,
            vxc_xasz=vxc_xasz,
            virial_xc_energy=virial_xc_energy,
        )

    def todict(self) -> dict[str, str | bool | int]:
        return {
            "name": self.name,
            "setup_name": self.setup_name,
            "stencil": self.stencil,
            "paw_potmode": self.paw_potmode.value,
            "virial_xc_energy": self.virial_xc_energy,
        }

    def initialize(
        self, density: Density, hamiltonian: Hamiltonian, wfs: WaveFunctions
    ) -> None:
        # needed for reading and ae_vxc
        self.ham = hamiltonian
        # needed for get_all_electron_xc_proj for the direct mode
        self.dens = density
        self.wfs = wfs
        self.gd: GridDescriptor | None = None
        self.vt = None
        self.vt_overlap_inv = None

    def calculate_impl(
        self, gd: GridDescriptor, n_sg: Array4D, v_sg: Array4D, e_g: Array3D
    ) -> None:
        assert self.gd is self.ham.finegd
        self.kernel.vxc_sg = self._vxc_sg
        self.kernel.virial_xc_energy = self.virial_xc_energy
        self.set_grid_descriptor(gd)
        self.kernel.calculate(e_g, n_sg, v_sg)

    def get_setup_name(self) -> str:
        return self.setup_name

    def write(self, writer, natoms=None) -> None:
        assert self.gd is not None
        writer.write(vxc_sg=self.gd.collect(self.vxc_sg) * Ha,
                     vxc_xZ=self.pack_atomic_potentials(self.vxc_xasz) * Ha)
        if self.vc_xag is not None:
            writer.write(vc_xag=self.pack_atomic_potentials(self.vc_xag) * Ha)

    def read(self, reader) -> None:
        h = reader.hamiltonian.xc
        if hasattr(h, 'xcpots'):
            return self.read_old(reader)
        nspins = self.ham.nspins
        self._vxc_sg = self.ham.finegd.zeros(self.ham.nspins)
        self.ham.finegd.distribute(h.vxc_sg / reader.ha, self.vxc_sg)
        self.kernel.vxc_sg = self._vxc_sg
        setups = self.ham.setups
        shapes_a = get_atomic_potential_shapes(
            setups, nspins, self.paw_potmode)
        self.vxc_xasz = self.unpack_atomic_potentials(
            h.vxc_xZ / reader.ha, shapes_a)
        if hasattr(h, 'vc_xag'):
            corepots_shapes_a = [(s.xc_correction.rgd.N,) for s in setups]
            self.vc_xag = self.unpack_atomic_potentials(
                h.vc_xag / reader.ha, corepots_shapes_a)

    def write_old(self, writer, natoms=None) -> None:
        assert self.gd is not None
        writer.write(xcpots=self.pack_pots(self.gd) * Ha)

    def read_old(self, reader) -> None:
        h = reader.hamiltonian.xc
        nspins = self.ham.nspins
        self._vxc_sg = self.ham.finegd.zeros(self.ham.nspins)
        setups = self.ham.setups
        shapes_a = get_atomic_potential_shapes(
            setups, nspins, self.paw_potmode)
        work_partition = self.ham.atomdist.work_partition
        self.vxc_xasz = [
            work_partition.arraydict(shapes_a, dtype=float)
            for _ in range(num_pawpots[self.paw_potmode])
        ]
        self.set_pots(self.ham.finegd, h.xcpots / Ha)

    def set_grid_descriptor(self, gd: GridDescriptor) -> None:
        if self.gd is not None and self.gd is not gd:
            if self._vxc_sg is None:
                self.vxc_sg = gd.zeros(1)
            else:
                v_sg = self.gd.collect(self.vxc_sg)
                self._vxc_sg = gd.empty(len(self._vxc_sg))
                gd.distribute(v_sg, self._vxc_sg)
        self.gd = gd
        self.kernel.set_grid_descriptor(gd, self.stencil)
        if self.ham is not None:
            for v_asz in self.vxc_xasz:
                if v_asz.partition is not self.ham.atomdist.work_partition:
                    v_asz.redistribute(self.ham.atomdist.work_partition)
            if self.vc_xag is not None:
                for v_asz in self.vc_xag:
                    if v_asz.partition is not self.ham.atomdist.work_partition:
                        v_asz.redistribute(self.ham.atomdist.work_partition)

    def calculate_paw_correction(
        self,
        setup,
        D_sp: Array2D,
        dEdD_sp: Array2D | None = None,
        addcoredensity: bool = True,
        a: int = None,
    ) -> float:
        assert a is not None
        if setup.xc_correction is None:
            return 0.0
        for v_asz in self.vxc_xasz:
            assert v_asz.partition is self.ham.atomdist.work_partition
        if self.paw_potmode is PAWPotMode.direct:
            dExc_sp = self.vxc_xasz[0][a]
            if dEdD_sp is not None:
                dEdD_sp[:] += dExc_sp
            # rough estimate: needs improvement
            E = np.sum(dExc_sp * D_sp)
        elif self.paw_potmode is PAWPotMode.none:
            # do nothing
            E = 0.0
        else:
            # lebedev and angular
            v_sz = self.vxc_xasz[0][a]
            vt_sz = self.vxc_xasz[1][a]
            if self.vc_xag is not None:
                v_sz = v_sz + self.vc_xag[0][a]
                vt_sz = vt_sz + self.vc_xag[1][a]
            rcalc = NXCRadialCalculator(
                v_sz, self.paw_potmode, self.virial_xc_energy)
            rcalct = NXCRadialCalculator(
                vt_sz, self.paw_potmode, self.virial_xc_energy)
            E = calculate_paw_correction(
                rcalc, rcalct, setup, D_sp, dEdD_sp, addcoredensity, a=a
            )
        return E

    def calculate_spherical(
        self, rgd, n_sg: Array2D, v_sg: Array2D, e_g: Array1D | None = None, a: int = 0
    ) -> float:
        # not tested
        if e_g is None:
            e_g = rgd.zeros()
        if len(self.vxc_xasz) == 0:
            if TYPE_CHECKING:
                assert self.vxc_sg is not None
            vxc_sz = self.vxc_sg[:, np.newaxis]
        else:
            vxc_sz = self.vxc_xasz[0][a]
        rcalc = NXCRadialCalculator(
            vxc_sz, self.paw_potmode, self.virial_xc_energy)
        e_g[:], dedn_sg = rcalc(rgd, n_sg[:, np.newaxis], np.ones(1))
        v_sg[:] = dedn_sg
        return rgd.integrate(e_g)

    def set_pots(self, gd: GridDescriptor, v: Array1D | None) -> None:
        """sets the packed potentials from the master to their respective ranks"""
        warnings.warn(
            "set_pots() does not work properly in case of kpt parallelization")
        v_sg = None
        fsize = 0
        if gd.comm.rank == 0:
            if TYPE_CHECKING:
                assert self.vxc_sg is not None and v is not None
            v_sg = gd.empty(len(self.vxc_sg), global_array=True)
            fsize = v_sg.size
            v_sg.ravel()[:] = v[:fsize].real
        gd.distribute(v_sg, out=self.vxc_sg)
        i = fsize
        for vxc_asz in self.vxc_xasz:
            shapes_a = vxc_asz.shapes_a
            sizea = sum([np.prod(shape) for shape in shapes_a])
            data_asz = np.empty(sizea, dtype=float)
            if gd.comm.rank == 0:
                data_asz[:] = v[i: i + sizea].real
            i += sizea
            # broadcast so we don't have to worry about the atomic partition
            gd.comm.broadcast(data_asz, 0)
            dist_arraydict_data(vxc_asz, data_asz)

    def pack_pots(self, gd: GridDescriptor) -> Array1D | float:
        # does not work for kpt parallelization
        warnings.warn(
            "pack_pots() does not work properly in case of kpt parallelization")
        v_sg = gd.collect(self.vxc_sg)
        if gd.comm.rank == 0:
            v = [v_sg.ravel()]
        for vxc_asz in self.vxc_xasz:
            # broadcast so that we don't have to worry about the partition mismatch of the arraydicts and gd
            data = broadcast_arraydict_data(vxc_asz)
            if gd.comm.rank == 0:
                v.append(data)
        if gd.comm.rank == 0:
            return np.concatenate(v)
        else:
            return np.nan

    def pack_atomic_potentials(self, vxc_xaz):
        if self.paw_potmode is PAWPotMode.none:
            return np.nan
        data_xZ = []
        for v_asz in vxc_xaz:
            v_asz = self.ham.atomdist.from_work(v_asz).copy()
            v_asz.redistribute(self.ham.atom_partition.as_serial())
            if self.ham.atom_partition.comm.rank == 0:
                data_xZ.append(v_asz.toarray())
        if self.ham.atom_partition.comm.rank == 0:
            return np.vstack(data_xZ)
        else:
            return np.nan

    def unpack_atomic_potentials(self, data_xZ, shapes_a):
        vxc_xasz = []
        for x in range(num_pawpots[self.paw_potmode]):
            v_asz = self.ham.atom_partition.arraydict(shapes_a, dtype=float)
            v_asz.redistribute(self.ham.atom_partition.as_serial())
            if self.ham.atom_partition.comm.rank == 0:
                v_asz.fromarray(data_xZ[x])
            v_asz.redistribute(self.ham.atom_partition)
            vxc_xasz.append(self.ham.atomdist.to_work(v_asz))
        return vxc_xasz

    def collect_angular_atomic_potentials(self) -> list[dict[int, Array3D]]:
        assert self.paw_potmode in (PAWPotMode.lebedev, PAWPotMode.angular)
        vxc_xasz: list[dict[int, Array3D]] = [{} for _ in self.vxc_xasz]
        for a, setup in enumerate(self.ham.setups):
            for x, vxc_asz in enumerate(self.vxc_xasz):
                vxc_sz = vxc_asz.get(a)
                if self.vc_xag is not None:
                    vc_g = self.vc_xag[x].get(a)
                if vxc_sz is None:
                    vxc_sz = np.empty(vxc_asz.shapes_a[a], dtype=float)
                    if self.vc_xag is not None:
                        vc_g = np.empty(
                            self.vc_xag[x].shapes_a[a], dtype=float)
                rank = vxc_asz.partition.rank_a[a]
                vxc_asz.partition.comm.broadcast(vxc_sz, rank)
                if self.vc_xag is not None:
                    vxc_asz.partition.comm.broadcast(vc_g, rank)
                if self.paw_potmode is PAWPotMode.lebedev:
                    if self.vc_xag is None:
                        vxc_sz = lebedev2angular(
                            vxc_sz, setup.xc_correction.Lmax)
                    else:
                        vxc_sz = lebedev2angular(
                            vxc_sz + vc_g, setup.xc_correction.Lmax)

                vxc_xasz[x][a] = vxc_sz
        return vxc_xasz

    def get_all_electron_xc(
        self, gridrefinement: int = 2, proj=False, return_gd=True, lfc_points=1000
    ) -> tuple[Array4D, GridDescriptor] | Array4D:
        if self.paw_potmode is PAWPotMode.direct or proj:
            v_sg, exc = get_all_electron_xc_proj(
                self, self.dens, self.ham, self.wfs)
            if return_gd:
                return v_sg, self.ham.finegd
            else:
                return v_sg
        if gridrefinement not in (2, 4):
            raise ValueError("gridrefinement should be either 2 or 4")
        gd = self.gd or self.ham.finegd
        assert self.vxc_sg is not None
        vxc_sg = self.vxc_sg.copy()
        nspins = len(vxc_sg)
        if gridrefinement == 4:
            from gpaw.transformers import Transformer

            fgd = gd.refine()
            interpolator = Transformer(gd, fgd, 3)
            vxc_sg = fgd.empty(nspins)
            for s in range(nspins):
                interpolator.apply(self.vxc_sg[s], vxc_sg[s])
            gd = fgd
        if self.paw_potmode is PAWPotMode.none:
            if return_gd:
                return vxc_sg, gd
            else:
                return vxc_sg
        spos_ac = self.ham.spos_ac
        setups = self.ham.setups
        vxc_asLg, vxct_asLg = self.collect_angular_atomic_potentials()
        for s in range(nspins):
            v_aj = []
            vt_aj = []
            for a, id_ in enumerate(setups.id_a):
                if not hasattr(setups[a], "xc_correction"):
                    raise NotImplementedError(id_)
                v_aj.append(get_atomic_vxc_spline(
                    setups[a], vxc_asLg[a][s], points=lfc_points))
                vt_aj.append(get_atomic_vxc_spline(
                    setups[a], vxct_asLg[a][s], points=lfc_points))
            v = LFC(gd, v_aj)
            v.set_positions(spos_ac=spos_ac)
            v.add(vxc_sg[s], c_axi=1.0)
            vt = LFC(gd, vt_aj)
            vt.set_positions(spos_ac=spos_ac)
            vt.add(vxc_sg[s], c_axi=-1.0)
        if return_gd:
            return vxc_sg, gd
        else:
            return vxc_sg


def calculate_angular_density(xcc, D_sp, n_qg, nc_g, add_core=True):
    nspins = len(D_sp)
    rgd = xcc.rgd
    D_sLq = np.inner(D_sp, xcc.B_pqL.T)
    n_sLg = np.dot(D_sLq, n_qg)
    # not considering core_hole
    if add_core:
        nc0_sg = rgd.empty(nspins)
        nc0_sg[:] = sqrt(4 * pi) / nspins * nc_g
        n_sLg[:, 0] += nc0_sg
    return n_sLg


def lebedev2angular(vxc_sng, Lmax):
    vxc_sLg = 4 * np.pi * np.einsum("n,nL,sng->sLg",
                                    weight_n, Y_nL[:, :Lmax], vxc_sng)
    return vxc_sLg


def angular2lebedev(vxc_sLg):
    Lmax = vxc_sLg.shape[1]
    vxc_sng = np.einsum("nL,sLg->sng", Y_nL[:, :Lmax], vxc_sLg)
    return vxc_sng


def get_atomic_vxc_spline(setup, vxc_Lg, points=1000):
    xcc = setup.xc_correction
    rgd = xcc.rgd
    lmax = int(sqrt(xcc.Lmax))
    v_lg = rgd.zeros(lmax)
    for ll in range(lmax):
        s = rgd.zeros()
        for m in range(2 * ll + 1):
            L = ll**2 + m
            s += vxc_Lg[L]
        v_lg[ll] = s

    v_j = []

    # rcut = rgd.r_g[-1]
    rcut = max(setup.rcut_j)
    gcut = rgd.ceil(rcut)
    v_lg[:, gcut:] = 0.0
    for ll, v_g in enumerate(v_lg):
        v_j.append(rgd.spline(v_g, rcut=rcut,
                              l=ll, points=points))
    return v_j


def calculate_atomic_potentials_from_kernel(
    rgd: RadialGridDescriptor,
    kernel,
    n_sLg: Array3D,
    paw_potmode: PAWPotMode,
) -> tuple[Array3D, Array3D]:
    assert paw_potmode is PAWPotMode.lebedev or paw_potmode is PAWPotMode.angular
    assert kernel.type in ["LDA", "GGA"], f"Not impelmented for {kernel.type}"
    nspins, Lmax = n_sLg.shape[:2]
    vxc_sng = rgd.zeros((nspins, len(Y_nL)))
    if kernel.type == 'LDA':
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            n_sg = np.dot(Y_L, n_sLg)
            e_g = rgd.zeros()
            v_sg = rgd.zeros(nspins)
            kernel.calculate(e_g, n_sg, v_sg)
            vxc_sng[:, n] = v_sg
    elif kernel.type == 'GGA':
        # needs improvement for spin-polarized case
        rcalc = GGARadialCalculator(kernel)
        dndr_sLg = np.empty_like(n_sLg)
        for n_Lg, dndr_Lg in zip(n_sLg, dndr_sLg):
            for n_g, dndr_g in zip(n_Lg, dndr_Lg):
                rgd.derivative(n_g, dndr_g)
        coeff_L = []
        for L in range(Lmax):
            ll, m = L_to_lm(L)
            coeff_L.append(2 * ll * (ll + 1))
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            n_sg = np.dot(Y_L, n_sLg)
            rnablaY_Lv = rnablaY_nLv[n, :Lmax]
            e_g, dedn_sg, b_vsg, dedsigma_xg = rcalc(
                rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n)
            vv_sg = dedsigma_xg[::2] * \
                np.einsum('L,sLg,L->sg', coeff_L, n_sLg, Y_L)
            vv_sg[:, 1:] /= rgd.r_g[1:]**2
            vv_sg[:, 0] = vv_sg[:, 1]
            vxc_sng[:, n] = dedn_sg + vv_sg

    if paw_potmode is PAWPotMode.lebedev:
        return vxc_sng
    else:
        return lebedev2angular(vxc_sng, Lmax)


def L_to_lm(L: int) -> tuple[int, int]:
    """convert L index to (l, m) index"""
    ll = int(sqrt(L))
    m = L - ll**2 - ll
    return ll, m


def atomic_fa_angular(setup: Setup, D_sp: Array2D) -> tuple[Array3D, Array3D]:
    xcc = setup.xc_correction
    if xcc is None:
        return np.zeros(1), np.zeros(1)
    n_sLg = calculate_angular_density(xcc, D_sp, xcc.n_qg, xcc.nc_g)
    nt_sLg = calculate_angular_density(xcc, D_sp, xcc.nt_qg, xcc.nct_g)
    nspins, Lmax = n_sLg.shape[:2]
    rgd = xcc.rgd
    vxc_sLg = rgd.zeros((nspins, Lmax))
    vxct_sLg = rgd.zeros((nspins, Lmax))
    N_ae = rgd.integrate(n_sLg[:, 0]) / np.sqrt(4 * np.pi)
    N_ps = rgd.integrate(nt_sLg[:, 0]) / np.sqrt(4 * np.pi)
    for L in range(Lmax):
        ll, m = L_to_lm(L)
        for s in range(nspins):
            v_g = rgd.poisson(n_sLg[s, L], l=ll)
            v_g[1:] /= rgd.r_g[1:]
            v_g[0] = v_g[1]
            # N = rgd.integrate(n_sLg[s, L])
            N = N_ae[s]
            if N < 1e-20:
                N = np.inf
            vxc_sLg[s, L] = -v_g / N
            v_g = rgd.poisson(nt_sLg[s, L], l=ll)
            v_g[1:] /= rgd.r_g[1:]
            v_g[0] = v_g[1]
            # Nt = rgd.integrate(nt_sLg[s, L])
            Nt = N_ps[s]
            if Nt < 1e-20:
                Nt = np.inf
            vxct_sLg[s, L] = -v_g / Nt
    return vxc_sLg, vxct_sLg


def atomic_fa_lebedev(setup: Setup, D_sp: Array2D, mix_spin: bool = False) -> tuple[Array3D, Array3D]:
    xcc = setup.xc_correction
    if xcc is None:
        return np.zeros(1), np.zeros(1)
    # n_sLg = calculate_angular_density(xcc, D_sp, xcc.n_qg, xcc.nc_g)
    # nt_sLg = calculate_angular_density(
    #     xcc, D_sp, xcc.nt_qg, xcc.nct_g)
    # rgd = xcc.rgd
    # nspins = len(D_sp)
    # Lmax = n_sLg.shape[1]
    # vxc_sng = rgd.zeros((nspins, len(Y_nL)))
    # vxct_sng = rgd.zeros((nspins, len(Y_nL)))
    # for n, Y_L in enumerate(Y_nL[:, :Lmax]):
    #     n_sg = np.dot(Y_L, n_sLg)
    #     nt_sg = np.dot(Y_L, nt_sLg)
    #     if mix_spin:
    #         n_g = n_sg.sum(axis=0)
    #         nt_g = nt_sg.sum(axis=0)
    #         N = rgd.integrate(n_g)
    #         if N == 0.0:
    #             N = np.inf
    #         Nt = rgd.integrate(nt_g)
    #         if Nt == 0.0:
    #             Nt = np.inf
    #         vxc_sng[:, n] = -divrl(rgd.poisson(n_g), 1, rgd.r_g) / N
    #         vxct_sng[:, n] = -divrl(rgd.poisson(nt_g), 1, rgd.r_g) / Nt
    #     else:
    #         for s in range(nspins):
    #             N = rgd.integrate(n_sg[s])
    #             # what to do if the density is charge neutral!
    #             # it might never happen with a soft enough shell
    #             if N == 0.0:
    #                 N = np.inf
    #             Nt = rgd.integrate(nt_sg[s])
    #             if Nt == 0.0:
    #                 Nt = np.inf
    #             vxc_sng[s, n] = -divrl(rgd.poisson(n_sg[s]), 1, rgd.r_g) / N
    #             vxct_sng[s, n] = -divrl(rgd.poisson(nt_sg[s]), 1, rgd.r_g) / Nt

    vxc_sLg, vxct_sLg = atomic_fa_angular(setup, D_sp)
    vxc_sng = angular2lebedev(vxc_sLg)
    vxct_sng = angular2lebedev(vxct_sLg)
    return vxc_sng, vxct_sng


def atomic_fa(setup: Setup, D_sp: Array2D, mode: PAWPotMode | str, mix_spin: bool = False) -> tuple[ArrayND, ...]:
    mode = PAWPotMode(mode)
    if mode is PAWPotMode.lebedev:
        return atomic_fa_lebedev(setup, D_sp, mix_spin)
    elif mode is PAWPotMode.angular:
        return atomic_fa_angular(setup, D_sp)
    elif mode is PAWPotMode.none:
        return (np.zeros(1),)
    elif mode is PAWPotMode.direct:
        vfa_szg, vfat_szg = atomic_fa_lebedev(setup, D_sp)
        rcalc = NXCRadialCalculator(vfa_szg, PAWPotMode.lebedev, False)
        rcalct = NXCRadialCalculator(vfat_szg, PAWPotMode.lebedev, False)
        dEdD_sp = np.zeros_like(D_sp, dtype=float)
        calculate_paw_correction(rcalc, rcalct, setup, D_sp, dEdD_sp, True)
        return (dEdD_sp,)


def get_atomic_potential_shapes(
    setups: Setups, nspins: int, mode: PAWPotMode
) -> list[tuple[int, ...]]:
    shapes_a: list[tuple[int, ...]] = []
    for setup in setups:
        xcc = setup.xc_correction
        if xcc is None:
            shapes_a.append((0,))
        elif mode is PAWPotMode.lebedev:
            shapes_a.append((nspins, len(Y_nL), xcc.ng))
        elif mode is PAWPotMode.angular:
            shapes_a.append((nspins, xcc.Lmax, xcc.ng))
        elif mode is PAWPotMode.direct:
            shapes_a.append((nspins, setup.ni * (setup.ni + 1) // 2))
        elif mode is PAWPotMode.none:
            shapes_a.append((0,))
    return shapes_a


def calculate_fa_atomic_potentials(
    D_asp: ArrayDict, setups: Setups, nspins: int, mode, mix_spin=False
) -> list[ArrayDict]:
    mode = PAWPotMode(mode)
    atompart = D_asp.partition
    shapes_a = get_atomic_potential_shapes(setups, nspins, mode)
    vxc_xasz = [
        atompart.arraydict(shapes_a, dtype=float) for _ in range(num_pawpots[mode])
    ]
    for a, D_sp in D_asp.items():
        vfa_xsz = atomic_fa(setups[a], D_sp, mode, mix_spin)
        for vxc_asz, vfa_sz in zip(vxc_xasz, vfa_xsz):
            vxc_asz[a][:] = vfa_sz
    return vxc_xasz


def calculate_fa_potential(
    fapoisson: BasePoissonSolver, gd: GridDescriptor, n_sg: Array4D, mix_spin=False
) -> ArrayND:
    nspins = len(n_sg)
    vHt_sg = gd.zeros(nspins)
    if mix_spin:
        n_g = n_sg.sum(axis=0)
        N = gd.integrate(n_g)
        vH_g = np.zeros_like(n_g)
        fapoisson.solve(vH_g, n_g / N)
        vHt_sg[:] = -1.0 * vH_g
    else:
        for s in range(nspins):
            N = gd.integrate(n_sg[s])
            fapoisson.solve(vHt_sg[s], n_sg[s] / N)
            vHt_sg[s] *= -1.0
    return vHt_sg


def FermiAmaldi(
    density: Density,
    poisson=None,
    setup_name: str = "LDA",
    paw_potmode: str | PAWPotMode = PAWPotMode.lebedev,
    virial_xc_energy: bool = False,
    stencil: int = 3,
    add_core_potential: bool = False,
    mix_spin_fa: bool = False,
    **poisson_kwargs: dict[str, Any],
) -> NumericalXC:
    if poisson is None:
        poisson_kwargs.setdefault("use_charge_center", True)
        poisson_kwargs.setdefault("use_charged_periodic_corrections", True)
        poisson = create_poisson_solver(**poisson_kwargs)
        poisson.set_grid_descriptor(density.finegd)
    paw_potmode = PAWPotMode(paw_potmode)
    from gpaw.old.pw.poisson import ReciprocalSpacePoissonSolver
    if isinstance(poisson, ReciprocalSpacePoissonSolver):
        vxc_sg = density.finegd.zeros(density.nspins)
        for s in range(density.nspins):
            N = density.finegd.integrate(density.nt_sg[s])
            # probably not correct for spin polarized systems
            pd3 = density.pd3
            v_q = pd3.zeros(q=0)
            rhot_q = pd3.fft(density.nt_sg[s], q=0)
            poisson._solve(v_q, rhot_q)
            vxc_sg[s] = -pd3.ifft(v_q, q=0, distribute=True) / N
    else:
        vxc_sg = calculate_fa_potential(
            poisson, poisson.gd, density.nt_sg, mix_spin_fa)

    D_asp = density.atomdist.to_work(density.D_asp)
    vxc_xasz = calculate_fa_atomic_potentials(
        D_asp, density.setups, density.nspins, paw_potmode, mix_spin_fa
    )
    vc_xag = None
    if add_core_potential:
        assert paw_potmode is PAWPotMode.lebedev
        setup_xc = XC(setup_name)

        def core_potentials_shape(a):
            return (density.setups[a].xc_correction.rgd.N,)
        vc_ag = D_asp.partition.arraydict(
            core_potentials_shape, dtype=float)
        vct_ag = D_asp.partition.arraydict(
            core_potentials_shape, dtype=float)
        for a in vc_ag:
            vc_ag[a][:] = 0.0
            vct_ag[a][:] = 0.0
            nc_g = density.setups[a].xc_correction.nc_g
            nct_g = density.setups[a].xc_correction.nct_g
            if setup_xc.kernel.type == 'LDA':
                e_g = np.zeros_like(nc_g)
                setup_xc.kernel.calculate(
                    e_g, nc_g[np.newaxis], vc_ag[a][np.newaxis])
                setup_xc.kernel.calculate(
                    e_g, nct_g[np.newaxis], vct_ag[a][np.newaxis])
            else:
                raise NotImplementedError
            vxc_xasz[0][a] -= vc_ag[a]
            vxc_xasz[1][a] -= vct_ag[a]
        vc_xag = [vc_ag, vct_ag]

    xc = NumericalXC(
        NumericalXCKernel(virial_xc_energy=virial_xc_energy),
        setup_name,
        paw_potmode=paw_potmode,
        virial_xc_energy=virial_xc_energy,
        stencil=stencil,
        vxc_sg=vxc_sg,
        vxc_xasz=vxc_xasz,
        vc_xag=vc_xag,
    )
    return xc


def get_all_electron_xc_proj(
    xc, density: Density, hamiltonian: Hamiltonian, wfs: WaveFunctions
) -> tuple[ArrayND, float]:
    # [experimental]
    from gpaw.utilities import unpack_density

    """
    This is not the actual AE effective potential but rather the projection of
    the atomic corrections to the Hamiltonian (multipled by the projectors) +
    the xc potential over the pseudo space (contributions to the Hamiltonian)
    """
    gd = hamiltonian.finegd
    nspins = hamiltonian.nspins
    v_sg = gd.zeros(nspins)
    exc = xc.calculate(gd, density.nt_sg, v_sg)
    atomdist = hamiltonian.atomdist
    setups = hamiltonian.setups
    D_asp = atomdist.to_work(density.D_asp)
    apart = D_asp.partition
    dEdD_asp = apart.arraydict(D_asp.shapes_a, dtype=float)
    exc_atomic = 0.0
    for a in apart.my_indices:
        dEdD_asp[a][:] = 0.0
        exc_atomic += xc.calculate_paw_correction(
            setups[a], D_asp[a], dEdD_asp[a], True, a=a
        )
    exc += hamiltonian.world.sum_scalar(exc_atomic)
    if wfs.mode == "pw":
        pt = BasisFunctions(gd, wfs.pt.spline_aj, dtype=wfs.dtype)
    else:
        pt = BasisFunctions(
            gd, [sphere.spline_j for sphere in wfs.pt.sphere_a], dtype=wfs.dtype
        )
    pt.set_positions(hamiltonian.spos_ac)
    dEdD_asp = atomdist.to_aux(dEdD_asp)
    # unpack dEdD_asp
    # does construct_density function double count the atoms?
    for s in range(nspins):
        vxc_MM = np.zeros((pt.Mmax, pt.Mmax), dtype=wfs.dtype)
        M1 = 0
        for a, setup in enumerate(setups):
            ni = setup.ni
            dEdD_sp = dEdD_asp.get(a)
            if dEdD_sp is None:
                dEdD_sp = np.empty((nspins, ni * (ni + 1) // 2))
            rank = D_asp.partition.rank_a[a]
            dEdD_asp.partition.comm.broadcast(dEdD_sp, rank)
            M2 = M1 + ni
            vxc_MM[M1:M2, M1:M2] = unpack_density(dEdD_sp[s])
            M1 = M2
        pt.construct_density(vxc_MM, v_sg[s], -1)
    return v_sg, exc
