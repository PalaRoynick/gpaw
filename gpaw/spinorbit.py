from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Bohr, Ha, alpha

from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.ibz2bz import IBZ2BZMaps
from gpaw.mpi import MPIComm, Parallelization, broadcast_array, serial_comm
from gpaw.new import zips as zip
from gpaw.new.brillouin import IBZ, MonkhorstPackKPoints
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.smearing import OccupationNumberCalculator as OccNumCalc
from gpaw.new.symmetry import Symmetries, SymmetrizationPlan
from gpaw.new.wave_functions import (WaveFunctions,
                                     add_to_4component_density_matrix)
from gpaw.occupations import OccupationNumberCalculator, ParallelLayout
from gpaw.old.band_descriptor import BandDescriptor
from gpaw.old.grid_descriptor import GridDescriptor
from gpaw.old.kpoint import KPoint
from gpaw.old.kpt_descriptor import KPointDescriptor
from gpaw.old.projections import Projections
from gpaw.setup import Setup
from gpaw.typing import Array1D, Array2D, Array3D, Array4D, ArrayND
from gpaw.utilities import pack_density
from gpaw.utilities.partition import AtomPartition

if TYPE_CHECKING:
    from gpaw.dft import DFT
    from gpaw.new.ase_interface import ASECalculator
    from gpaw.old.calculator import GPAW as OldGPAW

_L_vlmm: list[list[np.ndarray]] = []  # see get_L_vlmm() below


def calculate_soc(dft: DFT,
                  *,
                  n1: int = 0,
                  n2: int | None = None,
                  scale: float = 1.0,
                  theta: float = 0.0,  # degrees
                  phi: float = 0.0,  # degrees
                  ignore_xc_potential: bool = False
                  ) -> tuple[IBZWaveFunctions, dict[int, np.ndarray]]:
    ibzwfs = dft.ibzwfs
    ibz = ibzwfs.ibz

    n2 = n2 or ibzwfs.nbands

    P_ksani = {(wfs.k, wfs.spin): wfs.P_ani for wfs in ibzwfs}
    l_aj = [setup.l_j for setup in dft.setups]

    P_Ksani, kpt_comm, atomdist = distribute_projections(
        P_ksani,
        l_aj,
        ibzwfs.rank_ks,
        ibz,
        ibzwfs.comm,
        n1, n2)

    C_ss, s_vss = basis_change(theta=theta, phi=phi)

    eig_skn = ibzwfs.get_all_eigs(broadcast=True)

    # <phi_i|dV_adr / r * L_v|phi_j>
    dVL_avii = {a: soc(dft.setups[a],
                       dft.pot_calc.xc.xc,
                       np.array([pack_density(D_ii) for D_ii in D_sii.real]),
                       ignore_xc_potential) * scale
                for a, D_sii in dft.density.D_asii.items()}

    eig_Km = {}
    P_Kamsi = {}
    v_Kmn = {}
    for K, P_sani in P_Ksani.items():
        k = ibz.bz2ibz_K[K]
        eig_sn = eig_skn[:, k, n1:n2]
        P_amsi, eig_m = to_spinors(P_sani, eig_sn)
        P_amsi, eig_m, v_mn, spin_projection_mv = diagonalize(
            P_amsi, eig_m, dVL_avii, s_vss, C_ss)
        eig_Km[K] = eig_m
        P_Kamsi[K] = P_amsi
        v_Kmn[K] = v_mn

    def create_wfs_func(spin, q, k, kpt_c, kweight):
        wfs = WaveFunctions(
            setups=dft.setups,
            nbands=2 * (n2 - n1) if ibzwfs.collinear else n2 - n1,
            relpos_ac=dft.relpos_ac,
            atomdist=atomdist,
            q=q,
            k=k,
            kpt_c=kpt_c,
            weight=kweight,
            ncomponents=4,
            dtype=complex,
            domain_band_comm=atomdist.comm)
        wfs._P_ani = P_Kamsi[k]
        wfs._eig_n = eig_Km[k]
        wfs.xp = np
        return wfs

    nosym = Symmetries(cell=ibz.symmetries.cell_cv)
    bzwfs = IBZWaveFunctions.create(
        ibz=ibz.bz.reduce(nosym, use_time_reversal=False),
        ncomponents=4,
        create_wfs_func=create_wfs_func,
        kpt_comm=kpt_comm,
        kpt_band_comm=kpt_comm,
        comm=dft.comm)

    if isinstance(ibz.bz, MonkhorstPackKPoints):
        nelectrons = (dft.density.nvalence - dft.density.charge +
                      dft.pot_calc.charge)
        occcalc = OccNumCalc(
            dft.params.occupations.params,
            dft.atoms.pbc,
            bzwfs.ibz,
            bzwfs.nbands,
            {'k': kpt_comm, 'K': atomdist.comm},
            np.zeros(3) + np.nan,
            4,
            nelectrons,
            None)
        bzwfs.calculate_occs(occcalc, nelectrons)

    return bzwfs, v_Kmn


def distribute_projections(P_ksani: dict[tuple[int, int], AtomArrays],
                           l_aj: list[list[int]],
                           rank_ks: np.ndarray,
                           ibz: IBZ,
                           comm: MPIComm,
                           n1: int,
                           n2: int) -> tuple[dict[int, list[AtomArrays]],
                                             MPIComm,
                                             AtomDistribution]:
    """Distribute projections from IBZ to BZ.
    """
    nbzkpts = len(ibz.bz)
    p = Parallelization(comm, nbzkpts)
    p.set(band=1)
    comms = p.build_communicators()
    kpt_comm = comms['k']
    domain_comm = comms['d']
    mynk = (nbzkpts + kpt_comm.size - 1) // kpt_comm.size
    print(kpt_comm.size, domain_comm.size, mynk)
    P_layout = next(iter(P_ksani.values())).layout.new(atomdist=None)
    atomdist = AtomDistribution.from_number_of_atoms(len(l_aj), domain_comm)

    to_r = defaultdict(set)
    for K in range(nbzkpts):
        k = ibz.bz2ibz_K[K]
        for spin, from_rank in enumerate(rank_ks[k]):
            to_rank = K // mynk * domain_comm.size
            to_r[from_rank].add((to_rank, k, spin))

    P2_ksani = {}
    requests = []
    for (k1, spin1), P_ani in P_ksani.items():
        P0_ani = P_ani.gathergather()
        if P0_ani is not None:
            P0_ani = P0_ani[:, n1:n2]
            for r, k, spin in to_r[comm.rank]:
                if k == k1 and spin == spin1:
                    if r == comm.rank:
                        P2_ksani[(k, spin)] = P0_ani
                    else:
                        req = comm.send(P0_ani.data, r,
                                        tag=k * 2 + spin,
                                        block=False)
                        requests.append(req)

    if domain_comm.rank == 0:
        for from_rank, to in to_r.items():
            for to_rank, k, spin in to:
                if to_rank == comm.rank and from_rank != to_rank:
                    P_ani = P_layout.empty(n2 - n1)
                    comm.receive(P_ani.data, from_rank, tag=k * 2 + spin)
                    P2_ksani[(k, spin)] = P_ani
    comm.waitall(requests)

    symplan = SymmetrizationPlan(ibz.symmetries, l_aj)
    K1 = min(kpt_comm.rank * mynk, nbzkpts)
    K2 = min((kpt_comm.rank + 1) * mynk, nbzkpts)
    nspins = rank_ks.shape[1]
    P_Ksani = {}
    print(list(P2_ksani))
    print(atomdist, K1, K2)
    for K in range(K1, K2):
        k = ibz.bz2ibz_K[K]
        P_sani = []
        for spin in range(nspins):
            if domain_comm.rank == 0:
                P0_ani = symplan.rotate_projections(P2_ksani[(k, spin)],
                                                    ibz.s_K[K],
                                                    ibz.time_reversal_K[K])
            else:
                P0_ani = None
            P_ani = P_layout.new(atomdist=atomdist).empty(P0_ani.dims)
            P_ani.scatter_from(P0_ani)
            P_sani.append(P_ani)
        P_Ksani[K] = P_sani

    return P_Ksani, kpt_comm, atomdist


def to_spinors(P_sani: list[AtomArrays],
               eig_sn: np.ndarray) -> tuple[AtomArrays, np.ndarray]:
    nspins, nbands = eig_sn.shape
    if len(P_sani[0].dims) == 2:
        # Non-collinear:
        eig_m = eig_sn[0]
        P_amsi = P_sani[0].layout.zeros((nbands, 2))
        P_amsi.data[:] = P_sani[0].data
    else:
        eig_m = np.empty(2 * nbands)
        for s in range(2):
            eig_m[s::2] = eig_sn[-s]
        P_amsi = P_sani[0].layout.zeros((2 * nbands, 2))
        P_amsi.data[::2, 0] = P_sani[0].data
        P_amsi.data[1::2, 1] = P_sani[-1].data
    return P_amsi, eig_m


def diagonalize(P_amsi: AtomArrays,
                eig_m: np.ndarray,
                dVL_avii: dict[int, Array3D],
                s_vss: list[Array2D],
                C_ss: Array2D) -> tuple[AtomArrays,
                                        np.ndarray,
                                        np.ndarray,
                                        np.ndarray]:
    """Evaluate H in a basis of S_z eigenstates and diagonalize."""
    M = len(eig_m)
    H_mm = np.zeros((M, M), complex)
    for a, P_msi in P_amsi.items():
        dVL_vii = dVL_avii[a]
        ni = dVL_vii.shape[1]
        H_ssii = np.zeros((2, 2, ni, ni), complex)
        H_ssii[0, 0] = dVL_vii[2]
        H_ssii[0, 1] = dVL_vii[0] - 1.0j * dVL_vii[1]
        H_ssii[1, 0] = dVL_vii[0] + 1.0j * dVL_vii[1]
        H_ssii[1, 1] = -dVL_vii[2]

        # Tranform to theta, phi basis
        H_ssii = np.tensordot(C_ss, H_ssii, (0, 1))
        H_ssii = np.tensordot(C_ss.T.conj(), H_ssii, (1, 1))

        for s1 in range(2):
            for s2 in range(2):
                H_ii = H_ssii[s1, s2]
                P1_mi = P_msi[:, s1]
                P2_mi = P_msi[:, s2]
                H_mm += P1_mi.conj() @ H_ii @ P2_mi.T

    domain_comm = P_amsi.layout.atomdist.comm
    domain_comm.sum(H_mm, 0)
    if domain_comm.rank == 0:
        H_mm += np.diag(eig_m)
        eig_m, v_nm = np.linalg.eigh(H_mm)
    else:
        v_nm = np.empty((M, M), complex)
    domain_comm.broadcast(v_nm, 0)

    Pout_amsi = P_amsi.new(dtype=complex)
    Pout_amsi.data[:] = (v_nm.T @
                         P_amsi.data.reshape((M, -1))).reshape((M, 2, -1))

    sx_m = []
    sy_m = []
    sz_m = []

    v_msn = v_nm.copy().reshape((M // 2, 2, M)).T.copy()
    for v_sn in v_msn:
        sx_m.append(np.trace(v_sn.T.conj().dot(s_vss[0]).dot(v_sn)))
        sy_m.append(np.trace(v_sn.T.conj().dot(s_vss[1]).dot(v_sn)))
        sz_m.append(np.trace(v_sn.T.conj().dot(s_vss[2]).dot(v_sn)))

    spin_projection_mv = np.array([sx_m, sy_m, sz_m]).real.T.copy()

    return Pout_amsi, eig_m, v_nm.T, spin_projection_mv


def basis_change(*,
                 theta: float,
                 phi: float) -> tuple[np.ndarray, list[np.ndarray]]:
    theta *= np.pi / 180
    phi *= np.pi / 180

    # Hamiltonian with SO in KS basis
    # The even indices in H_mm are spin up along hat n defined by theta, phi
    # Basis change matrix for constructing Pauli matrices in theta,phi basis:
    #     sigma_i^n = C^dag sigma_i C
    C_ss = np.array([[np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
                      -np.sin(theta / 2) * np.exp(-1.0j * phi / 2)],
                     [np.sin(theta / 2) * np.exp(1.0j * phi / 2),
                      np.cos(theta / 2) * np.exp(1.0j * phi / 2)]])

    sx_ss = np.array([[0, 1], [1, 0]], complex)
    sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
    sz_ss = np.array([[1, 0], [0, -1]], complex)
    s_vss = [C_ss.T.conj() @ sx_ss @ C_ss,
             C_ss.T.conj() @ sy_ss @ C_ss,
             C_ss.T.conj() @ sz_ss @ C_ss]

    return C_ss, s_vss


class SOCResult:
    def __init__(self,
                 bzwfs: IBZWaveFunctions,
                 v_Kmn):
        self.bzwfs = bzwfs
        self.v_Kmn = v_Kmn
        try:
            self.fermi_level = bzwfs.fermi_level * Ha
        except ValueError:
            self.fermi_level = np.nan

    def weights(self):
        return self.bzwfs.ibz.weight_k

    def eigenvalues(self):
        return self.bzwfs.get_all_eigs()[0] * Ha

    def eigenvectors(self):
        nbands = self.bzwfs.nbands
        v_Kmn = np.zeros((len(self.bzwfs.ibz), nbands, nbands), complex)
        for K, v_mn in self.v_Kmn.items():
            v_Kmn[K] = v_mn
        assert self.bzwfs.comm.size == 1
        return v_Kmn

    def get_orbital_magnetic_moments(self):
        """Returns the orbital magnetic moment vector for each atom a."""
        from gpaw.new.orbmag import calculate_orbmag_from_density
        P_ani = self.bzwfs._wfs_u[0]._P_ani
        setups = self.bzwfs._wfs_u[0].setups
        atomdist = P_ani.layout.atomdist
        D_asii = AtomArraysLayout(
            [(ni, ni) for (ni,) in P_ani.layout.shape_a],
            dtype=complex,
            atomdist=atomdist).zeros(4)
        for wfs in self.bzwfs:
            for D_sii, P_msi in zip(D_asii.values(), wfs.P_ani.values()):
                add_to_4component_density_matrix(
                    D_sii, P_msi, wfs.weight * wfs.occ_n, np)
        self.bzwfs.kpt_comm.sum(D_asii.data)
        n_aj = [setup.n_j for setup in setups]
        l_aj = [setup.l_j for setup in setups]
        return calculate_orbmag_from_density(D_asii, n_aj, l_aj)

    def __getitem__(self, K):
        return KPointData(K, self.bzwfs._wfs_u[K], self.v_Kmn[K])

    def pdos_weights(self,
                     a: int,
                     indices: list[int]
                     ) -> Array3D:
        """PDOS weights."""
        nbands = self.bzwfs.nbands
        nbzkpts = len(self.bzwfs.ibz)
        dos_kms = np.zeros((nbzkpts, nbands, 2))

        for wfs in self.bzwfs:
            dos_kms[wfs.k] = (abs(wfs.P_ani[a][:, :, indices])**2).sum(2)

        return dos_kms / nbzkpts


class KPointData:
    def __init__(self, K, wfs, v_mn):
        self.bz_index = K
        self.wfs = wfs
        self.v_mn = v_mn
        self.f_m = wfs.occ_n
        self.projections = wfs.P_ani

    def wavefunctions(self, calc, periodic=True):
        kd = calc.wfs.kd
        assert kd.nibzkpts == kd.nbzkpts

        # For spinors the number of bands is doubled and a
        # spin dimension is added
        Ns = calc.wfs.nspins
        Nm, Nn = self.v_mn.shape

        if calc.wfs.collinear:
            u_snR = [[calc.wfs.get_wave_function_array(n, self.bz_index, s,
                                                       periodic=periodic)
                      for n in range(Nn // 2)]
                     for s in range(Ns)]
            u_msR = np.empty((Nm, 2) + u_snR[0][0].shape, complex)
            np.einsum('mn, nabc -> mabc', self.v_mn[:, ::2], u_snR[0],
                      out=u_msR[:, 0])
            np.einsum('mn, nabc -> mabc', self.v_mn[:, 1::2], u_snR[-1],
                      out=u_msR[:, 1])
        else:
            u_nsR = np.array(
                [calc.wfs.get_wave_function_array(n, self.bz_index, 0,
                                                  periodic=periodic)
                 for n in range(Nn)])
            u_msR = np.einsum('mn, nsxyz -> msxyz',
                              self.v_mn, u_nsR)

        return u_msR


class WaveFunction:
    def __init__(self,
                 eigenvalues: Array1D,
                 projections: Projections,
                 bz_index: int = None):
        self.eig_m = eigenvalues
        self.projections = projections
        self.spin_projection_mv: Array2D | None = None
        self.v_mn: Array2D | None = None
        self.f_m = np.empty_like(self.eig_m)
        self.f_m[:] = np.nan
        self.bz_index = bz_index

    def transform(self, IBZ2BZMap, K) -> WaveFunction:
        """Transforms PAW projections from IBZ to BZ k-point."""
        projections = IBZ2BZMap.map_projections(self.projections)
        return WaveFunction(self.eig_m.copy(), projections, K)

    def redistribute_atoms(self,
                           atom_partition: AtomPartition
                           ) -> WaveFunction:
        projections = self.projections.redist(atom_partition)
        return WaveFunction(self.eig_m.copy(), projections, self.bz_index)

    def add_soc(self,
                dVL_avii: dict[int, Array3D],
                s_vss: list[Array2D],
                C_ss: Array2D) -> None:
        """Evaluate H in a basis of S_z eigenstates."""
        if self.projections.bcomm.rank > 0:
            return

        M = self.projections.nbands
        H_mm = np.zeros((M, M), complex)
        for a, dVL_vii in dVL_avii.items():
            ni = dVL_vii.shape[1]
            H_ssii = np.zeros((2, 2, ni, ni), complex)
            H_ssii[0, 0] = dVL_vii[2]
            H_ssii[0, 1] = dVL_vii[0] - 1.0j * dVL_vii[1]
            H_ssii[1, 0] = dVL_vii[0] + 1.0j * dVL_vii[1]
            H_ssii[1, 1] = -dVL_vii[2]

            # Tranform to theta, phi basis
            H_ssii = np.tensordot(C_ss, H_ssii, (0, 1))
            H_ssii = np.tensordot(C_ss.T.conj(), H_ssii, (1, 1))
            H_ssii *= Ha

            P_msi = self.projections[a]
            for s1 in range(2):
                for s2 in range(2):
                    H_ii = H_ssii[s1, s2]
                    P1_mi = P_msi[:, s1]
                    P2_mi = P_msi[:, s2]
                    H_mm += np.dot(np.dot(P1_mi.conj(), H_ii), P2_mi.T)

        domain_comm = self.projections.atom_partition.comm
        domain_comm.sum(H_mm, 0)
        if domain_comm.rank == 0:
            H_mm += np.diag(self.eig_m)
            self.eig_m, v_nm = np.linalg.eigh(H_mm)
        else:
            v_nm = np.empty((M, M), complex)

        domain_comm.broadcast(v_nm, 0)

        P_mI = self.projections.matrix.array
        P_mI[:] = v_nm.T.copy().dot(P_mI)

        sx_m = []
        sy_m = []
        sz_m = []

        v_msn = v_nm.copy().reshape((M // 2, 2, M)).T.copy()
        for v_sn in v_msn:
            sx_m.append(np.trace(v_sn.T.conj().dot(s_vss[0]).dot(v_sn)))
            sy_m.append(np.trace(v_sn.T.conj().dot(s_vss[1]).dot(v_sn)))
            sz_m.append(np.trace(v_sn.T.conj().dot(s_vss[2]).dot(v_sn)))

        self.spin_projection_mv = np.array([sx_m, sy_m, sz_m]).real.T.copy()
        self.v_mn = v_nm.T

    def wavefunctions(self, calc, periodic=True):
        kd = calc.wfs.kd
        assert kd.nibzkpts == kd.nbzkpts

        # For spinors the number of bands is doubled and a
        # spin dimension is added
        Ns = calc.wfs.nspins
        Nm, Nn = self.v_mn.shape

        if calc.wfs.collinear:
            u_snR = [[calc.wfs.get_wave_function_array(n, self.bz_index, s,
                                                       periodic=periodic)
                      for n in range(Nn // 2)]
                     for s in range(Ns)]
            u_msR = np.empty((Nm, 2) + u_snR[0][0].shape, complex)
            np.einsum('mn, nabc -> mabc', self.v_mn[:, ::2], u_snR[0],
                      out=u_msR[:, 0])
            np.einsum('mn, nabc -> mabc', self.v_mn[:, 1::2], u_snR[-1],
                      out=u_msR[:, 1])
        else:
            u_nsR = np.array(
                [calc.wfs.get_wave_function_array(n, self.bz_index, 0,
                                                  periodic=periodic)
                 for n in range(Nn)])
            u_msR = np.einsum('mn, nsxyz -> msxyz',
                              self.v_mn, u_nsR)

        return u_msR

    @property
    def P_amj(self):
        M = self.projections.nbands
        return {
            a: P_msi.transpose((0, 2, 1)).copy().reshape((M, -1))
            for a, P_msi in self.projections.items()}

    def pdos_weights(self,
                     a: int,
                     indices: list[int]
                     ) -> Array3D:
        """PDOS weights."""
        dos_ms = np.zeros((self.projections.nbands, 2))

        P_amsi = self.projections

        if a in P_amsi:
            dos_ms[:, :] = (abs(P_amsi[a][:, :, indices])**2).sum(2)

        return dos_ms


class BZWaveFunctions:
    """Container for eigenvalues and PAW projections (all of BZ)."""
    def __init__(self,
                 kd: KPointDescriptor,
                 wfs: dict[int, WaveFunction],
                 occ: OccupationNumberCalculator | None,
                 nelectrons: float,
                 n_aj: list[list[int]],
                 l_aj: list[list[int]]):
        self.wfs = wfs
        self.occ = occ
        self.nelectrons = nelectrons
        self.n_aj = n_aj
        self.l_aj = l_aj

        self.nbzkpts = kd.nbzkpts

        # Initialize ranks:
        self.ranks = np.zeros(kd.nbzkpts, int)
        for k in wfs:
            self.ranks[k] = kd.comm.rank
        kd.comm.sum(self.ranks)

        wf = next(iter(wfs.values()))  # get the first WaveFunction object

        self.shape = (kd.nbzkpts, wf.projections.nbands)
        self.domain_comm = wf.projections.atom_partition.comm
        self.bcomm = wf.projections.bcomm
        self.kpt_comm = kd.comm

        self.fermi_level = self._calculate_occ_numbers_and_fermi_level()

        self.size = kd.N_c
        self.bz2ibz_map = np.arange(self.nbzkpts)

    def weights(self):
        return np.zeros(len(self)) + 1 / self.nbzkpts

    def _calculate_occ_numbers_and_fermi_level(self) -> float:
        if self.occ is not None:
            eig_im = [wf.eig_m for wf in self]
            weight = 1.0 / self.nbzkpts
            weight_i = [weight] * len(eig_im)

            f_im, (fermi_level,), _ = self.occ.calculate(
                self.nelectrons,
                eig_im,
                weight_i)
            for wf, f_m in zip(self, f_im):
                wf.f_m[:] = f_m
        else:
            fermi_level = 0.0

        if self.domain_comm.rank == 0:
            fermi_level = self.bcomm.sum_scalar(fermi_level)
        fermi_level = self.domain_comm.sum_scalar(fermi_level)

        return fermi_level

    def calculate_band_energy(self) -> float:
        """Calculate sum over occupied eigenvalues."""
        if self.domain_comm.rank == 0 and self.bcomm.rank == 0:
            weight = 1.0 / self.nbzkpts
            e_band = sum(wf.eig_m.dot(wf.f_m) for wf in self) * weight
            e_band = self.kpt_comm.sum_scalar(e_band)
        else:
            e_band = 0.0

        if self.domain_comm.rank == 0:
            e_band = self.bcomm.sum_scalar(e_band)
        e_band = self.domain_comm.sum_scalar(e_band)
        return e_band

    def __iter__(self):
        for bz_index in sorted(self.wfs):
            yield self[bz_index]

    def __getitem__(self, bz_index):
        return self.wfs[bz_index]

    def __len__(self):
        return len(self.wfs)

    def eigenvalues(self,
                    broadcast: bool = True
                    ) -> Array2D:
        """Eigenvalues in eV for the whole BZ."""
        return self._collect(attrgetter('eig_m'), broadcast=broadcast)

    def occupation_numbers(self,
                           broadcast: bool = True
                           ) -> Array2D:
        """Occupation numbers for the whole BZ."""
        return self._collect(attrgetter('f_m'), broadcast=broadcast)

    def eigenvectors(self,
                     broadcast: bool = True
                     ) -> Array4D:
        """Eigenvectors for the whole BZ."""
        nbands = self.shape[1]
        assert nbands % 2 == 0
        return self._collect(attrgetter('v_mn'), (nbands,), complex,
                             broadcast=broadcast)

    def spin_projections(self,
                         broadcast: bool = True
                         ) -> Array3D:
        """Spin projections for the whole BZ."""
        return self._collect(attrgetter('spin_projection_mv'), (3,),
                             broadcast=broadcast)

    def get_atomic_density_matrices(self):
        """Returns atomic density matrices for each atom."""
        from gpaw.new.wave_functions import add_to_4component_density_matrix

        assert self.domain_comm.size == 1
        assert self.bcomm.size == 1

        D_asii = {}  # Could also be an AtomArrays object?
        for a, l_j in enumerate(self.l_aj):
            ni = (2 * np.array(l_j) + 1).sum()
            D_sii = np.zeros([4, ni, ni], dtype=complex)
            for wfs, weight in zip(self.wfs.values(), self.weights()):
                add_to_4component_density_matrix(D_sii, wfs.projections[a],
                                                 wfs.f_m * weight, np)
            self.kpt_comm.sum(D_sii)
            D_asii[a] = D_sii

        return D_asii

    def get_orbital_magnetic_moments(self):
        """Returns the orbital magnetic moment vector for each atom a."""
        D_asii = self.get_atomic_density_matrices()
        from gpaw.new.orbmag import calculate_orbmag_from_density
        return calculate_orbmag_from_density(D_asii, self.n_aj, self.l_aj)

    def pdos_weights(self,
                     a: int,
                     indices: list[int],
                     broadcast: bool = True
                     ) -> Array4D:
        """Projections for PDOS.

        Returns (nbzkpts, nbands, 2)-shaped ndarray
        of the square of absolute value of the projections.
        """
        def func(wf):
            return wf.pdos_weights(a, indices)

        return self._collect(func,
                             (2,),
                             broadcast=broadcast,
                             sum_over_domain=True)

    def _collect(self,
                 func: Callable[[WaveFunction], ArrayND],
                 shape: tuple[int, ...] = None,
                 dtype=float,
                 broadcast: bool = True,
                 sum_over_domain: bool = False) -> ArrayND:
        """Helper method for collecting (and broadcasting) ndarrays."""

        total_shape = self.shape + (shape or ())

        if broadcast:
            array_kmx = self._collect(func,
                                      shape,
                                      dtype,
                                      False,
                                      sum_over_domain)
            if array_kmx.ndim == 0:
                array_kmx = np.empty(total_shape, dtype)
            return broadcast_array(array_kmx,
                                   self.kpt_comm, self.bcomm, self.domain_comm)

        if self.bcomm.rank != 0:
            return np.empty(shape=())

        if not sum_over_domain and self.domain_comm.rank != 0:
            return np.empty(shape=())

        comm = self.kpt_comm
        if comm.rank == 0:
            array_kmx = np.empty(total_shape, dtype)
            for k, rank in enumerate(self.ranks):
                if rank == 0:
                    array_kmx[k] = func(self[k])
                else:
                    comm.receive(array_kmx[k], rank)
            if sum_over_domain:
                self.domain_comm.sum(array_kmx)
            if self.domain_comm.rank == 0:
                return array_kmx
            else:
                return np.empty(shape=())

        for k, rank in enumerate(self.ranks):
            if rank == comm.rank:
                comm.send(np.ascontiguousarray(func(self[k])), 0)

        return np.empty(shape=())


def soc_eigenstates_raw(ibzwfs: Iterable[tuple[int, WaveFunction]],
                        dVL_avii: dict[int, Array3D],
                        ibz2bzmaps: IBZ2BZMaps,
                        atom_partition,
                        theta: float = 0.0,
                        phi: float = 0.0) -> dict[int, WaveFunction]:

    theta *= np.pi / 180
    phi *= np.pi / 180

    # Hamiltonian with SO in KS basis
    # The even indices in H_mm are spin up along \hat n defined by \theta, phi
    # Basis change matrix for constructing Pauli matrices in \theta,\phi basis:
    #     \sigma_i^n = C^\dag\sigma_i C
    C_ss = np.array([[np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
                      -np.sin(theta / 2) * np.exp(-1.0j * phi / 2)],
                     [np.sin(theta / 2) * np.exp(1.0j * phi / 2),
                      np.cos(theta / 2) * np.exp(1.0j * phi / 2)]])

    sx_ss = np.array([[0, 1], [1, 0]], complex)
    sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
    sz_ss = np.array([[1, 0], [0, -1]], complex)
    s_vss = [C_ss.T.conj() @ sx_ss @ C_ss,
             C_ss.T.conj() @ sy_ss @ C_ss,
             C_ss.T.conj() @ sz_ss @ C_ss]

    bzwfs = {}
    for ibz_index, ibzwf in ibzwfs:
        for K in np.nonzero(ibz2bzmaps.kd.bz2ibz_k == ibz_index)[0]:
            bzwf = ibzwf.transform(ibz2bzmaps[K], K)

            # Redistribute to match dVL_avii:
            bzwf = bzwf.redistribute_atoms(atom_partition)

            bzwf.add_soc(dVL_avii, s_vss, C_ss)
            bzwfs[int(K)] = bzwf  # MYPY: signedinteger[_32Bit | _64Bit] -> int

    return bzwfs


def extract_ibz_wave_functions(kpt_qs: list[list[KPoint]],
                               bd: BandDescriptor,
                               gd: GridDescriptor,
                               n1: int,
                               n2: int,
                               eigenvalues: Array3D = None
                               ) -> Iterator[tuple[int, WaveFunction]]:
    """Yield tuples of IBZ-index and wave functions.

    All atoms and bands will be on rank == 0 of gd.comm and bd.comm
    respectively.  This makes slicing the bands (from n1 to n2-1)
    and symmetry operations on the projections easier.
    """

    nproj_a = kpt_qs[0][0].projections.nproj_a

    collinear = kpt_qs[0][0].s is not None

    nbands = n2 - n1
    if collinear:
        nbands *= 2

    # All atoms on rank-0:
    atom_partition = AtomPartition(gd.comm, np.zeros_like(nproj_a))

    # All bands on rank-0 (nrow * ncol = 1 * 1):
    bdist = (bd.comm, 1, 1)

    for kpt_s in kpt_qs:
        # Collect bands and atoms to bd.comm.rank == 0 and gd.comm.rank == 0:
        if eigenvalues is None:
            eig_sn = [bd.collect(kpt.eps_n) for kpt in kpt_s]
        P_snI = [kpt.projections.collect() for kpt in kpt_s]

        projections = Projections(
            nbands=nbands,
            nproj_a=nproj_a,
            atom_partition=atom_partition,
            bdist=bdist,
            collinear=False)

        if bd.comm.rank == 0 and gd.comm.rank == 0:
            P1_nI = P_snI[0]
            P2_nI = P_snI[-1]
            assert P1_nI is not None
            assert P2_nI is not None
            if collinear:
                eig_m = np.empty((n2 - n1) * 2)
                if eigenvalues is None:
                    eig_m[::2] = eig_sn[0][n1:n2] * Ha
                    eig_m[1::2] = eig_sn[-1][n1:n2] * Ha
                else:
                    for s in range(2):
                        eig_m[s::2] = eigenvalues[-s, kpt_s[-s].k]
                projections.array[:] = 0.0
                projections.array[::2, 0] = P1_nI[n1:n2]
                projections.array[1::2, 1] = P2_nI[n1:n2]
            else:
                eig_m = eig_sn[0][n1:n2] * Ha
                projections.matrix.array[:] = P1_nI[n1:n2]
        else:
            eig_m = np.empty(0)

        ibz_index = kpt_s[0].k

        yield ibz_index, WaveFunction(eig_m, projections)


def soc_eigenstates(calc: ASECalculator | OldGPAW | str | Path,
                    n1: int = 0,
                    n2: int = None,
                    scale: float = 1.0,
                    theta: float = 0.0,  # degrees
                    phi: float = 0.0,  # degrees
                    eigenvalues: Array3D = None,  # eV
                    occcalc: OccupationNumberCalculator = None,
                    projected: bool = False,
                    ignore_xc_potential: bool = False,
                    new=1,
                    ) -> BZWaveFunctions | SOCResult:
    """Calculate SOC eigenstates.

    Parameters:
        calc: Calculator
            GPAW calculator or path to gpw-file.
        n1, n2: int
            Range of bands to include (n1 <= n < n2).  Default is all
            bands available.
        scale: float
            Scale the spinorbit coupling by this amount.
        theta: float
            Angle in degrees.
        phi: float
            Angle in degrees.
        eigenvalues: ndarray
            Optionally use these eigenvalues instead for those from *calc*.
            The shape must be: (nspins, nibzkpts, n2 - n1).  Units: eV.
        occcalc:
            Occupation-number calculator.  By default, the one from *calc*
            will be used.
        ignore_xc_potential:
            Ignore XC-contribution to dU/dr for the effective potential
            (dU/dr is dominated by the Hartree part).

    Returns a BZWaveFunctions object covering the whole BZ.
    """
    from gpaw import GPAW
    from gpaw.new.ase_interface import ASECalculator

    if isinstance(calc, (str, Path)):
        calc = GPAW(calc)

    if isinstance(calc, ASECalculator):
        assert not projected
        assert occcalc is None
        assert eigenvalues is None
        ibzwfs, v_Kmn = calculate_soc(
            calc.dft,
            n1=n1,
            n2=n2,
            scale=scale,
            theta=theta,
            phi=phi,
            ignore_xc_potential=ignore_xc_potential)
        return SOCResult(ibzwfs, v_Kmn)

    n1 = n1 or 0
    n2 = n2 or 0
    if n2 <= 0:
        if eigenvalues is None:
            nbands = calc.get_number_of_bands()
        else:
            nbands = eigenvalues.shape[2]
        n2 += nbands

    # <phi_i|dV_adr / r * L_v|phi_j>
    dVL_avii = {a: soc(calc.wfs.setups[a],
                       calc.hamiltonian.xc,
                       D_sp,
                       ignore_xc_potential) * scale
                for a, D_sp in calc.density.D_asp.items()}

    if projected:
        dVL_avii = {a: projected_soc(dVL_vii, theta=theta, phi=phi)
                    for a, dVL_vii in dVL_avii.items()}

    kd = calc.wfs.kd
    bd = calc.wfs.bd
    gd = calc.wfs.gd
    atom_partition = calc.density.atom_partition

    if eigenvalues is not None:
        assert eigenvalues.shape == (kd.nspins, kd.nibzkpts, n2 - n1)

    kpt_qs = get_both_spins(calc.wfs)
    ibzwfs = extract_ibz_wave_functions(kpt_qs,  # type: ignore
                                        bd, gd, n1, n2, eigenvalues)
    ibz2bzmaps = IBZ2BZMaps.from_calculator(calc)

    bzwfs = soc_eigenstates_raw(ibzwfs,
                                dVL_avii,
                                ibz2bzmaps,
                                atom_partition,
                                theta, phi)

    if bd.comm.rank == 0 and gd.comm.rank == 0:
        parallel_layout = ParallelLayout(BandDescriptor(1),
                                         kd.comm,
                                         serial_comm)
        occcalc = occcalc or calc.wfs.occupations
        occcalc = occcalc.copy(bz2ibzmap=list(range(kd.nbzkpts)),
                               parallel_layout=parallel_layout)
    else:
        occcalc = None

    n_aj = [setup.n_j for setup in calc.wfs.setups]
    l_aj = [setup.l_j for setup in calc.wfs.setups]

    return BZWaveFunctions(kd, bzwfs, occcalc, calc.wfs.nvalence, n_aj, l_aj)


def soc(a: Setup,
        xc,
        D_sp: Array2D,
        ignore_xc_potential=False) -> Array3D:
    """<phi_i|dU^a/dr / r * L_v|phi_j>"""
    v_g = get_radial_potential_derivative(a, xc, D_sp, ignore_xc_potential)
    Ng = len(v_g)
    phi_jg = a.data.phi_jg

    Lx_lmm, Ly_lmm, Lz_lmm = get_L_vlmm()

    dVL_vii = np.zeros((3, a.ni, a.ni), complex)
    N1 = 0
    for j1, l1 in enumerate(a.l_j):
        Nm = 2 * l1 + 1
        N2 = 0
        for j2, l2 in enumerate(a.l_j):
            if l1 == l2:
                f_g = phi_jg[j1][:Ng] * v_g * phi_jg[j2][:Ng]
                c = a.xc_correction.rgd.integrate(f_g, n=-1) / (4 * np.pi)
                dVL_vii[0, N1:N1 + Nm, N2:N2 + Nm] = c * Lx_lmm[l1]
                dVL_vii[1, N1:N1 + Nm, N2:N2 + Nm] = c * Ly_lmm[l1]
                dVL_vii[2, N1:N1 + Nm, N2:N2 + Nm] = c * Lz_lmm[l1]
            else:
                pass
            N2 += 2 * l2 + 1
        N1 += Nm
    return dVL_vii * alpha**2 / 4.0


def projected_soc(dVL_vii: Array3D,
                  theta: float = 0,
                  phi: float = 0) -> Array3D:
    """
    Optional Args:
        theta (float): The angle from z-axis in degrees
        phi (float): The angle from x-axis in degrees
    """
    theta *= np.pi / 180
    phi *= np.pi / 180
    n_v = np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)])
    dVL_vii = (np.dot(dVL_vii.T, n_v)[:, :, np.newaxis] * n_v).T
    return dVL_vii


def get_radial_potential_derivative(a: Setup,
                                    xc,
                                    D_sp: Array2D,
                                    ignore_xc_potential=False) -> Array1D:
    """Calculates (dU/dr) for the effective potential.
    Below, f_g denotes dU/dr which is also the negative of the radial force"""

    rgd = a.xc_correction.rgd
    r_g = rgd.r_g.copy()
    r_g[0] = 1.0e-12
    dr_g = rgd.dr_g

    B_pq = a.xc_correction.B_pqL[:, :, 0]
    n_qg = a.xc_correction.n_qg
    D_sq = np.dot(D_sp, B_pq)
    n_sg = np.dot(D_sq, n_qg) / (4 * np.pi)**0.5
    Ns = len(D_sp)
    if Ns == 4:
        Ns = 1
    n_sg[:Ns] += a.xc_correction.nc_g / Ns

    # Coulomb force from nucleus
    fc_g = a.Z / r_g**2

    # Hartree force
    rho_g = 4 * np.pi * r_g**2 * dr_g * np.sum(n_sg[:Ns], axis=0)
    fh_g = -np.array([np.sum(rho_g[:ig]) for ig in range(len(r_g))]) / r_g**2

    f_g = fc_g + fh_g

    # xc force
    if not ignore_xc_potential:
        v_sg = np.zeros_like(n_sg)
        xc.calculate_spherical(rgd, n_sg, v_sg)
        fxc_g = np.mean([rgd.derivative(v_g) for v_g in v_sg[:Ns]],
                        axis=0)
        f_g += fxc_g

    return f_g


def get_anisotropy(calc, theta=0.0, phi=0.0, nbands=0, width=None):
    """Calculates the sum of occupied spinorbit eigenvalues.

    Returns the result relative to the sum of eigenvalues without
    spinorbit coupling.
    """
    raise RuntimeError('Please use BZWaveFunctions.calculate_band_energy() '
                       'instead.')


def get_magnetic_moments(calc, theta=0.0, phi=0.0, nbands=None, width=None):
    """Calculates the magnetic moments inside all PAW spheres"""

    raise RuntimeError(
        'This function has no tests.  It is very likely that it no longer '
        'works correctly after merging !677.')

    from gpaw.utilities import unpack_hermitian

    if nbands is None:
        nbands = calc.get_number_of_bands()
    Nk = len(calc.get_ibz_k_points())

    C_ss = np.array([[np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
                      -np.sin(theta / 2) * np.exp(-1.0j * phi / 2)],
                     [np.sin(theta / 2) * np.exp(1.0j * phi / 2),
                      np.cos(theta / 2) * np.exp(1.0j * phi / 2)]])
    sx_ss = np.array([[0, 1], [1, 0]], complex)
    sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
    sz_ss = np.array([[1, 0], [0, -1]], complex)
    sx_ss = C_ss.T.conj().dot(sx_ss).dot(C_ss)
    sy_ss = C_ss.T.conj().dot(sy_ss).dot(C_ss)
    sz_ss = C_ss.T.conj().dot(sz_ss).dot(C_ss)

    states = soc_eigenstates(calc,
                             theta=theta,
                             phi=phi,
                             return_wfs=True,
                             bands=range(nbands))
    e_km = states['eigenvalues']
    v_knm = states['eigenstates']

    from gpaw.occupations import occupation_numbers
    if width is None:
        assert calc.wfs.occupations.name == 'fermi-dirac'
        width = calc.wfs.occupations._width
    if width == 0.0:
        width = 1.e-6
    weight_k = calc.get_k_point_weights() / 2
    ne = calc.wfs.setups.nvalence - calc.density.charge
    f_km = occupation_numbers({'name': 'fermi-dirac', 'width': width},
                              e_km[np.newaxis],
                              weight_k=weight_k,
                              nelectrons=ne)[0][0]

    m_v = np.zeros(3, complex)
    for ik in range(Nk):
        ut0_nG = np.array([calc.wfs.get_wave_function_array(n, ik, 0)
                           for n in range(nbands)])
        ut1_nG = np.array([calc.wfs.get_wave_function_array(n, ik, 1)
                           for n in range(nbands)])
        mocc = np.where(f_km[ik] * Nk - 1.0e-6 < 0.0)[0][0]
        for m in range(mocc + 1):
            f = f_km[ik, m]
            ut0_G = np.dot(v_knm[ik][::2, m], np.swapaxes(ut0_nG, 0, 2))
            ut1_G = np.dot(v_knm[ik][1::2, m], np.swapaxes(ut1_nG, 0, 2))
            ut_sG = np.array([ut0_G, ut1_G])

            mx_G = np.zeros(np.shape(ut0_G), complex)
            my_G = np.zeros(np.shape(ut0_G), complex)
            mz_G = np.zeros(np.shape(ut0_G), complex)
            for s1 in range(2):
                for s2 in range(2):
                    mx_G += ut_sG[s1].conj() * sx_ss[s1, s2] * ut_sG[s2]
                    my_G += ut_sG[s1].conj() * sy_ss[s1, s2] * ut_sG[s2]
                    mz_G += ut_sG[s1].conj() * sz_ss[s1, s2] * ut_sG[s2]
            m_v += calc.wfs.gd.integrate(np.array([mx_G, my_G, mz_G])) * f

    m_av = []
    for a in range(len(calc.atoms)):
        N0_p = calc.density.setups[a].N0_p.copy()
        N0_ij = unpack_hermitian(N0_p)
        Dx_ij = np.zeros_like(N0_ij, complex)
        Dy_ij = np.zeros_like(N0_ij, complex)
        Dz_ij = np.zeros_like(N0_ij, complex)
        Delta_p = calc.density.setups[a].Delta_pL[:, 0].copy()
        Delta_ij = unpack_hermitian(Delta_p)
        for ik in range(Nk):
            P_ami = ...  # get_spinorbit_projections(calc, ik, v_knm[ik])
            P_smi = np.array([P_ami[a][:, ::2], P_ami[a][:, 1::2]])
            P_smi = np.dot(C_ss, np.swapaxes(P_smi, 0, 1))

            P0_mi = P_smi[0]
            P1_mi = P_smi[1]
            f_mm = np.diag(f_km[ik])

            Dx_ij += P0_mi.conj().T.dot(f_mm).dot(P1_mi)
            Dx_ij += P1_mi.conj().T.dot(f_mm).dot(P0_mi)
            Dy_ij -= 1.0j * P0_mi.conj().T.dot(f_mm).dot(P1_mi)
            Dy_ij += 1.0j * P1_mi.conj().T.dot(f_mm).dot(P0_mi)
            Dz_ij += P0_mi.conj().T.dot(f_mm).dot(P0_mi)
            Dz_ij -= P1_mi.conj().T.dot(f_mm).dot(P1_mi)

        mx = np.sum(N0_ij * Dx_ij).real
        my = np.sum(N0_ij * Dy_ij).real
        mz = np.sum(N0_ij * Dz_ij).real

        m_av.append([mx, my, mz])
        m_v[0] += np.sum(Delta_ij * Dx_ij) * (4 * np.pi)**0.5
        m_v[1] += np.sum(Delta_ij * Dy_ij) * (4 * np.pi)**0.5
        m_v[2] += np.sum(Delta_ij * Dz_ij) * (4 * np.pi)**0.5

    return m_v.real, m_av


def get_parity_eigenvalues(calc, ik=0, spin_orbit=False, bands=None, Nv=None,
                           inversion_center=[0, 0, 0], deg_tol=1.0e-6,
                           tol=1.0e-6):
    """Calculates parity eigenvalues at time-reversal invariant k-points.
    Only works in plane wave mode.
    """

    assert len(calc.get_bz_k_points()) == len(calc.get_ibz_k_points())

    kpt_c = calc.get_ibz_k_points()[ik]
    if Nv is None:
        Nv = int(calc.get_number_of_electrons() / 2)

    if bands is None:
        bands = range(calc.get_number_of_bands())

    # Find degenerate subspaces
    eig_n = calc.get_eigenvalues(kpt=ik)[bands]
    e_in = []
    used_n = []
    for n1, e1 in enumerate(eig_n):
        if n1 not in used_n:
            n_n = []
            for n2, e2 in enumerate(eig_n):
                if np.abs(e1 - e2) < deg_tol:
                    n_n.append(n2)
                    used_n.append(n2)
            e_in.append(n_n)

    print()
    print(f' Inversion center at: {inversion_center}')
    print(f' Calculating inversion eigenvalues at k = {kpt_c}')
    print()

    center_v = np.array(inversion_center) / Bohr
    G_Gv = calc.wfs.pd.get_reciprocal_vectors(q=ik, add_q=True)

    psit_nG = np.array([calc.wfs.kpt_u[ik].psit_nG[n]
                        for n in bands])
    if spin_orbit:
        n1 = bands[0]
        n2 = bands[-1] + 1
        assert (bands == np.arange(n1, n2)).all()
        soc = soc_eigenstates(calc, n1=n1, n2=n2)
        v_kmn = soc.eigenvectors()
        psit0_mG = np.dot(v_kmn[ik, :, ::2], psit_nG)
        psit1_mG = np.dot(v_kmn[ik, :, 1::2], psit_nG)
    for n in range(len(bands)):
        psit_nG[n] /= (np.sum(np.abs(psit_nG[n])**2))**0.5
    if spin_orbit:
        for n in range(2 * len(bands)):
            A = np.sum(np.abs(psit0_mG[n])**2) + np.sum(np.abs(psit1_mG[n])**2)
            psit0_mG[n] /= A**0.5
            psit1_mG[n] /= A**0.5

    P_GG = np.ones((len(G_Gv), len(G_Gv)), float)
    for iG, G_v in enumerate(G_Gv):
        P_GG[iG] -= ((G_Gv[:] + G_v).round(6)).any(axis=1)
    assert (P_GG == P_GG.T).all()

    phase_G = np.exp(-2.0j * np.dot(G_Gv, center_v))

    p_n = []
    print('n   P_n')
    for n_n in e_in:
        if spin_orbit:
            # The dimension of parity matrix is doubled with spinorbit
            m_m = [2 * n_n[0] + i for i in range(2 * len(n_n))]
            Ppsit0_mG = np.dot(P_GG, psit0_mG[m_m].T).T
            Ppsit0_mG[:] *= phase_G
            Ppsit1_mG = np.dot(P_GG, psit1_mG[m_m].T).T
            Ppsit1_mG[:] *= phase_G
            P_nn = np.dot(psit0_mG[m_m].conj(), np.array(Ppsit0_mG).T)
            P_nn += np.dot(psit1_mG[m_m].conj(), np.array(Ppsit1_mG).T)
        else:
            Ppsit_nG = np.dot(P_GG, psit_nG[n_n].T).T
            Ppsit_nG[:] *= phase_G
            P_nn = np.dot(psit_nG[n_n].conj(), np.array(Ppsit_nG).T)
        P_eig = np.linalg.eigh(P_nn)[0]
        if np.allclose(np.abs(P_eig), 1, tol):
            P_n = np.sign(P_eig).astype(int).tolist()
            if spin_orbit:
                # Only include one of the degenerate pair of eigenvalues
                Pm = np.sign(P_eig).tolist().count(-1)
                Pp = np.sign(P_eig).tolist().count(1)
                P_n = Pm // 2 * [-1] + Pp // 2 * [1]
            print(f'{str(n_n)[1:-1]}: {str(P_n)[1:-1]}')
            p_n += P_n
        else:
            print(f'  {n_n} are not parity eigenstates')
            print(f'     P_n: {P_eig}')
            print(f'     e_n: {eig_n[n_n]}')
            p_n += [0 for n in n_n]

    return np.ravel(p_n)


def get_L_vlmm():
    if len(_L_vlmm) == 3:
        return _L_vlmm

    s = np.array([[0.0]])
    p = np.zeros((3, 3), complex)  # y, z, x
    p[0, 1] = -1.0j
    p[1, 0] = 1.0j
    d = np.zeros((5, 5), complex)  # xy, yz, z^2, xz, x^2-y^2
    d[0, 3] = -1.0j
    d[3, 0] = 1.0j
    d[1, 2] = -3**0.5 * 1.0j
    d[2, 1] = 3**0.5 * 1.0j
    d[1, 4] = -1.0j
    d[4, 1] = 1.0j
    f = np.zeros((7, 7), complex)
    f[0, 5] = -0.5 * 6**0.5 * 1.0j
    f[5, 0] = 0.5 * 6**0.5 * 1.0j
    f[1, 4] = -0.5 * 10**0.5 * 1.0j
    f[4, 1] = 0.5 * 10**0.5 * 1.0j
    f[1, 6] = -0.5 * 6**0.5 * 1.0j
    f[6, 1] = 0.5 * 6**0.5 * 1.0j
    f[2, 3] = -6**0.5 * 1.0j
    f[3, 2] = 6**0.5 * 1.0j
    f[2, 5] = -0.5 * 10**0.5 * 1.0j
    f[5, 2] = 0.5 * 10**0.5 * 1.0j
    _L_vlmm.append([s, p, d, f])

    p = np.zeros((3, 3), complex)  # y, z, x
    p[1, 2] = -1.0j
    p[2, 1] = 1.0j
    d = np.zeros((5, 5), complex)  # xy, yz, z^2, xz, x^2-y^2
    d[0, 1] = 1.0j
    d[1, 0] = -1.0j
    d[2, 3] = -3**0.5 * 1.0j
    d[3, 2] = 3**0.5 * 1.0j
    d[3, 4] = -1.0j
    d[4, 3] = 1.0j
    f = np.zeros((7, 7), complex)
    f[0, 1] = 0.5 * 6**0.5 * 1.0j
    f[1, 0] = -0.5 * 6**0.5 * 1.0j
    f[1, 2] = 0.5 * 10**0.5 * 1.0j
    f[2, 1] = -0.5 * 10**0.5 * 1.0j
    f[3, 4] = -6**0.5 * 1.0j
    f[4, 3] = 6**0.5 * 1.0j
    f[4, 5] = -0.5 * 10**0.5 * 1.0j
    f[5, 4] = 0.5 * 10**0.5 * 1.0j
    f[5, 6] = -0.5 * 6**0.5 * 1.0j
    f[6, 5] = 0.5 * 6**0.5 * 1.0j
    _L_vlmm.append([s, p, d, f])

    p = np.zeros((3, 3), complex)  # y, z, x
    p[0, 2] = 1.0j
    p[2, 0] = -1.0j
    d = np.zeros((5, 5), complex)  # xy, yz, z^2, xz, x^2-y^2
    d[0, 4] = 2.0j
    d[4, 0] = -2.0j
    d[1, 3] = 1.0j
    d[3, 1] = -1.0j
    f = np.zeros((7, 7), complex)
    f[0, 6] = 3.0j
    f[6, 0] = -3.0j
    f[1, 5] = 2.0j
    f[5, 1] = -2.0j
    f[2, 4] = 1.0j
    f[4, 2] = -1.0j
    _L_vlmm.append([s, p, d, f])
    return _L_vlmm


def get_both_spins(wfs):
    if hasattr(wfs, 'kpt_qs'):
        return wfs.kpt_qs
    if wfs.nspins == 1:
        return [[wfs] for wfs in wfs.kpt_u]
    kpt_u = wfs.kpt_u.copy()
    kpt1 = kpt_u[0]
    kpt2 = kpt_u[-1]
    kpt_comm = wfs.kd.comm
    if kpt1.s == 1:
        reqs = [
            kpt_comm.send(kpt1.f_n.copy(), kpt_comm.rank - 1, block=False),
            kpt_comm.send(kpt1.eps_n, kpt_comm.rank - 1, block=False),
            kpt_comm.send(kpt1.projections.array, kpt_comm.rank - 1,
                          block=False)]
        del kpt_u[0]
    if kpt2.s == 0:
        kpt3 = KPoint(kpt2.weightk,
                      kpt2.weight,
                      1,
                      kpt2.k,
                      kpt2.q,
                      kpt2.phase_cd)
        kpt3.f_n = np.empty_like(kpt2.f_n)
        kpt3.eps_n = np.empty_like(kpt2.eps_n)
        kpt3._projections = kpt2.projections.new()
        kpt_comm.receive(kpt3.f_n, kpt_comm.rank + 1)
        kpt_comm.receive(kpt3.eps_n, kpt_comm.rank + 1)
        kpt_comm.receive(kpt3._projections.array, kpt_comm.rank + 1)
        kpt_u.append(kpt3)
    if kpt1.s == 1:
        kpt_comm.waitall(reqs)
    kpt_qs = [[up, dn] for up, dn in zip(kpt_u[::2], kpt_u[1::2])]
    return kpt_qs
