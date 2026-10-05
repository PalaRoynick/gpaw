from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Bohr, Ha, alpha

from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.mpi import MPIComm, Parallelization, serial_comm
from gpaw.new import zips as zip
from gpaw.new.brillouin import IBZ, MonkhorstPackKPoints, ranks
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.smearing import OccupationNumberCalculator as OccNumCalc
from gpaw.new.symmetry import Symmetries, SymmetrizationPlan
from gpaw.new.wave_functions import (WaveFunctions,
                                     add_to_4component_density_matrix)
from gpaw.occupations import OccupationNumberCalculator
from gpaw.setup import Setup
from gpaw.utilities import pack_density
from gpaw.core.matrix import suggest_blocking, Matrix
from gpaw.new.logger import Logger

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
                  ignore_xc_potential: bool = False,
                  occ_calc_params: dict | None = None,
                  projected=False,
                  zora=False,
                  log: str | Path | Logger | None = '-'
                  ) -> tuple[IBZWaveFunctions,
                             float,
                             dict[int, np.ndarray],
                             dict[int, np.ndarray]]:
    """Calculate SOC.

    Parameters:
        dft: DFT-calculation
            DFT-calculation.
        n1, n2: int
            Range of bands to include (n1 <= n < n2).  Default is all
            bands available.
        scale: float
            Scale the spinorbit coupling by this amount.
        theta: float
            Angle in degrees.
        phi: float
            Angle in degrees.
        occ_calc_params: dict
            By default, the one from *calc* will be used.
        ignore_xc_potential:
            Ignore XC-contribution to dU/dr for the effective potential
            (dU/dr is dominated by the Hartree part).
    """
    if not isinstance(log, Logger):
        log = Logger(log, comm=dft.comm)

    ibzwfs = dft.ibzwfs
    ibz = ibzwfs.ibz

    n2 = n2 or ibzwfs.nbands  # default is all bands
    log(f'Calculating SOC for bands [{n1}, {n2}[')

    P_ksani = {(wfs.k, wfs.spin): wfs.P_ani for wfs in ibzwfs}
    l_aj = [setup.l_j for setup in dft.setups]

    # Calculate full BZ projections from IBZ and distribute
    # projections over kpt_comm and atomdist:
    P_Ksani, kpt_comm, atomdist = distribute_projections(
        P_ksani,
        l_aj,
        ibzwfs.rank_ks,
        ibz,
        ibzwfs.comm,
        n1, n2,
        dft.relpos_ac)
    log(f'Distributing {len(ibz.bz)} k-point x {len(dft.atoms)} atoms '
        f'over {kpt_comm.size} x {atomdist.comm.size} cores')

    C_ss, s_vss = basis_change(theta=theta, phi=phi)

    eig_skn = ibzwfs.get_all_eigs(broadcast=True)

    # <phi_i|dV_adr / r * L_v|phi_j>
    D_asii = dft.density.D_asii
    # Complemetary comm to D_asii:
    comm1 = (serial_comm if D_asii.layout.atomdist.comm.size == ibzwfs.comm
             else ibzwfs.kpt_band_comm)
    D_asii = D_asii.redist(atomdist=atomdist,
                           comm1=comm1,
                           comm2=kpt_comm)
    dVL_avii = {a: soc(dft.setups[a],
                       dft.pot_calc.xc.xc,
                       np.array([pack_density(D_ii) for D_ii in D_sii.real]),
                       ignore_xc_potential,
                       zora=zora) * scale
                for a, D_sii in D_asii.items()}

    if projected:
        for dVL_vii in dVL_avii.values():
            dVL_vii[:] = projected_soc(dVL_vii, theta=theta, phi=phi)

    # Calculate SOC-hamiltonians and diagonalize:
    eig_Km = {}
    P_Kamsi = {}
    v_Kmn = {}
    spin_projections_Kmv: dict[int, np.ndarray] = {}
    for K, P_sani in P_Ksani.items():
        k = ibz.bz2ibz_K[K]
        eig_sn = eig_skn[:, k, n1:n2]
        P_amsi, eig_m = to_spinors(P_sani, eig_sn)
        P_amsi, eig_m, v_mn, spin_projection_mv = diagonalize(
            P_amsi, eig_m, dVL_avii, s_vss, C_ss)
        eig_Km[K] = eig_m
        P_Kamsi[K] = P_amsi
        v_Kmn[K] = v_mn
        spin_projections_Kmv[K] = spin_projection_mv

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

    # Create IBWZWaveFunctions object representing the full BZ:
    nosym = Symmetries(cell=ibz.symmetries.cell_cv)
    bzwfs = IBZWaveFunctions.create(
        ibz=ibz.bz.reduce(nosym, use_time_reversal=False),
        ncomponents=4,
        create_wfs_func=create_wfs_func,
        kpt_comm=kpt_comm,
        kpt_band_comm=kpt_comm,
        comm=dft.comm)

    if isinstance(ibz.bz, MonkhorstPackKPoints):
        # We have a Monkhorst-Pack sampling of the BZ
        # (not a band-path): Calculate occupation numbers:
        nelectrons = (dft.density.nvalence - dft.density.charge +
                      dft.pot_calc.charge)
        log('Calculationg occupations numbers for '
            f'{nelectrons:.3f} electrons')
        occcalc = OccNumCalc(
            occ_calc_params or dft.params.occupations.params,
            dft.atoms.pbc,
            bzwfs.ibz,
            bzwfs.nbands,
            {'k': kpt_comm, 'K': atomdist.comm},
            np.zeros(3) + np.nan,
            4,
            nelectrons,
            None)
        e_band = bzwfs.calculate_occs(occcalc, nelectrons)[0]
    else:
        e_band = np.nan

    return bzwfs, e_band, v_Kmn, spin_projections_Kmv


def distribute_projections(P_ksani: dict[tuple[int, int], AtomArrays],
                           l_aj: list[list[int]],
                           rank_ks: np.ndarray,
                           ibz: IBZ,
                           comm: MPIComm,
                           n1: int,
                           n2: int,
                           relpos_ac: np.ndarray
                           ) -> tuple[dict[int, list[AtomArrays]],
                                      MPIComm,
                                      AtomDistribution]:
    """Distribute projections from IBZ to BZ."""
    # Parallelize full BZ projections over kpt_comm and domain_comm:
    nbzkpts = len(ibz.bz)
    p = Parallelization(comm, nbzkpts)
    p.set(band=1)
    comms = p.build_communicators()
    kpt_comm = comms['k']
    domain_comm = comms['d']

    krank_K = ranks(kpt_comm.size, nbzkpts)
    _P_ani = next(iter(P_ksani.values()))
    input_domain_band_comm_size = (
        _P_ani.comm.size * _P_ani.layout.atomdist.comm.size)
    collinear = len(_P_ani.dims) == 1
    P_layout = _P_ani.layout.new(atomdist=serial_comm)
    atomdist = AtomDistribution.from_number_of_atoms(len(l_aj), domain_comm)

    # Figure out which ranks need which k-points:
    to_r = defaultdict(set)
    for K in range(nbzkpts):
        k = ibz.bz2ibz_K[K]
        for spin, from_rank in enumerate(rank_ks[k]):
            from_rank *= input_domain_band_comm_size
            to_rank = krank_K[K] * domain_comm.size
            to_r[int(from_rank)].add((int(to_rank), int(k), spin))

    # Non-blocking sends:
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
                        # print('s', comm.rank, r, P0_ani.data.shape, k)
                        req = comm.send(P0_ani.data, r,
                                        tag=k * 2 + spin,
                                        block=False)
                        requests.append(req)

    # Receives:
    dims = (n2 - n1,) if collinear else (n2 - n1, 2)
    if domain_comm.rank == 0:
        for from_rank, to in to_r.items():
            for to_rank, k, spin in to:
                if to_rank == comm.rank and from_rank != to_rank:
                    P_ani = P_layout.empty(dims)
                    # print('r', comm.rank, from_rank, P_ani.data.shape, k)
                    comm.receive(P_ani.data, from_rank, tag=k * 2 + spin)
                    P2_ksani[(k, spin)] = P_ani
    comm.waitall(requests)

    # Use symmetries to map projections from IBZ to BZ:
    symplan = SymmetrizationPlan(ibz.symmetries, l_aj)
    nspins = rank_ks.shape[1]
    P_Ksani = {}
    for K, krank in enumerate(krank_K):
        if krank != kpt_comm.rank:
            continue
        k = ibz.bz2ibz_K[K]
        P_sani = []
        for spin in range(nspins):
            if domain_comm.rank == 0:
                P0_ani = symplan.rotate_projections(
                    P2_ksani[(k, spin)],
                    symnum=ibz.s_K[K],
                    time_reversal=ibz.time_reversal_K[K],
                    ignore_phase=True)
            else:
                P0_ani = None
            P_ani = P_layout.new(atomdist=atomdist).empty(dims)
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
                dVL_avii: dict[int, np.ndarray],
                s_vss: list[np.ndarray],
                C_ss: np.ndarray) -> tuple[AtomArrays,
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
        H_mm += np.einsum('msi, szij, wzj -> mw',
                          P_msi.conj(), H_ssii, P_msi)

    domain_comm = P_amsi.layout.atomdist.comm
    domain_comm.sum(H_mm, 0)
    if domain_comm.rank == 0:
        H_mm += np.diag(eig_m)

    if domain_comm.size == 1 or M < 1000:
        if domain_comm.rank == 0:
            eig_m, v_nm = np.linalg.eigh(H_mm)
        else:
            v_nm = np.empty((M, M), complex)
        domain_comm.broadcast(v_nm, 0)
    else:
        rows, columns, blocksize = suggest_blocking(M, domain_comm.size)
        M_mm = Matrix(M, M,
                      dtype=complex,
                      dist=(domain_comm, 1, 1),
                      data=H_mm.copy() if domain_comm.rank == 0 else None)
        eig_m = M_mm.eigh(scalapack=(domain_comm, rows, columns, blocksize))
        if domain_comm.rank == 0:
            v_mn = M_mm.data
        else:
            v_mn = H_mm
        domain_comm.broadcast(v_mn, 0)
        v_nm = v_mn.T

    Pout_amsi = P_amsi.new(dtype=complex)
    Pout_amsi.data[:] = (v_nm.T @
                         P_amsi.data.reshape((M, -1))).reshape((M, 2, -1))

    v_nsm = v_nm.reshape((M // 2, 2, M))
    spin_projection_mv = np.einsum(
        'nsm, vsz, nzm -> mv', v_nsm.conj(), s_vss, v_nsm)

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
    """Container for eigenvalues and PAW projections (all of BZ)."""
    def __init__(self,
                 bzwfs: IBZWaveFunctions,
                 e_band: float,
                 v_Kmn: dict[int, np.ndarray],
                 spin_projections_Kmv: dict[int, np.ndarray]):
        self.bzwfs = bzwfs
        self.e_band = e_band
        self.v_Kmn = v_Kmn
        self.spin_prijections_Kmv = spin_projections_Kmv
        try:
            self.fermi_level = bzwfs.fermi_level * Ha
        except ValueError:
            self.fermi_level = np.nan
        self.bz2ibz_map = bzwfs.ibz.bz2ibz_K

    def weights(self):
        return self.bzwfs.ibz.weight_k

    def eigenvalues(self):
        """Eigenvalues in eV for the whole BZ."""
        return self.bzwfs.get_all_eigs(broadcast=True)[0] * Ha

    def eigenvectors(self):
        """Eigenvectors for the whole BZ."""
        nbands = self.bzwfs.nbands
        v_Kmn = np.zeros((len(self.bzwfs.ibz), nbands, nbands), complex)
        for K, v_mn in self.v_Kmn.items():
            v_Kmn[K] = v_mn
        self.bzwfs.kpt_comm.sum(v_Kmn)
        return v_Kmn

    def calculate_band_energy(self):
        """Sum over occupied eigenvalues."""
        return self.e_band * Ha

    @property
    def size(self):
        bz = self.bzwfs.ibz.bz
        assert isinstance(bz, MonkhorstPackKPoints)
        return bz.size_c

    def spin_projections(self):
        """Spin projections for the whole BZ."""
        nbands = self.bzwfs.nbands
        sp_Kmn = np.zeros((len(self.bzwfs.ibz), nbands, 3))
        for K, sp_mn in self.spin_prijections_Kmv.items():
            sp_Kmn[K] = sp_mn
        self.bzwfs.kpt_comm.sum(sp_Kmn)
        return sp_Kmn

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
                     ) -> np.ndarray:
        """Projections for PDOS.

        Returns (nbzkpts, nbands, 2)-shaped ndarray
        of the square of absolute value of the projections.
        """
        nbands = self.bzwfs.nbands
        nbzkpts = len(self.bzwfs.ibz)
        dos_kms = np.zeros((nbzkpts, nbands, 2))

        for wfs in self.bzwfs:
            if a in wfs.P_ani:
                dos_kms[wfs.k] = (abs(wfs.P_ani[a][:, :, indices])**2).sum(2)

        self.bzwfs.comm.sum(dos_kms)

        return dos_kms


class KPointData:
    def __init__(self, K, wfs, v_mn):
        self.bz_index = K
        self.wfs = wfs
        self.v_mn = v_mn
        self.f_m = wfs.occ_n
        self.eig_m = wfs.eig_n * Ha
        self.P_amsi = wfs.P_ani
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

    @property
    def P_amj(self):
        M = self.P_amsi.dims[0]
        return {
            a: P_msi.transpose((0, 2, 1)).copy().reshape((M, -1))
            for a, P_msi in self.P_amsi.items()}


def soc_eigenstates(calc: ASECalculator | OldGPAW | str | Path,
                    n1: int = 0,
                    n2: int = None,
                    scale: float = 1.0,
                    theta: float = 0.0,  # degrees
                    phi: float = 0.0,  # degrees
                    eigenvalues: np.ndarray = None,  # eV
                    occcalc: OccupationNumberCalculator = None,
                    projected: bool = False,
                    ignore_xc_potential: bool = False
                    ) -> SOCResult:
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

    if isinstance(calc, (str, Path)):
        calc = GPAW(calc)

    assert eigenvalues is None
    ibzwfs, e_band, v_Kmn, spin_projections_Kmv = calculate_soc(
        calc.dft,
        n1=n1,
        n2=n2,
        scale=scale,
        theta=theta,
        phi=phi,
        occ_calc_params=(occcalc.todict()
                         if occcalc is not None else None),
        ignore_xc_potential=ignore_xc_potential,
        projected=projected)
    return SOCResult(ibzwfs, e_band, v_Kmn, spin_projections_Kmv)


def soc(a: Setup,
        xc,
        D_sp: np.ndarray,
        ignore_xc_potential=False,
        zora=False) -> np.ndarray:
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


def projected_soc(dVL_vii: np.ndarray,
                  theta: float = 0,
                  phi: float = 0) -> np.ndarray:
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
                                    D_sp: np.ndarray,
                                    ignore_xc_potential=False,
                                    zora: bool = False) -> np.ndarray:
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
    Ns = len(D_sp) % 3

    n_sg[:Ns] += a.xc_correction.nc_g / Ns
    n_g = n_sg[:Ns].sum(axis=1)

    # Coulomb force from nucleus
    f_g = a.Z

    # Hartree force
    x_g = 4 * np.pi * r_g**2 * dr_g * n_g
    f_g[1:] -= np.add.accumulate(x_g[:-1])
    f_g *= r_g**-2

    if zora:
        v_g = (rgd.poisson(n_g) - a.Z) / r_g

    # xc force
    if not ignore_xc_potential:
        v_sg = np.zeros_like(n_sg)
        xc.calculate_spherical(rgd, n_sg, v_sg)
        if zora:
            v_g += v_sg[:Ns].mean(axis=1)
        fxc_g = np.mean([rgd.derivative(v_g) for v_g in v_sg[:Ns]],
                        axis=0)
        f_g += fxc_g

    if zora:
        f_g *= (1.0 - v_g * alpha**2 / 2)**-2

    return f_g


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
