from __future__ import annotations

from collections import defaultdict

import numpy as np
from ase.units import Ha

from gpaw.dft import DFT
from gpaw.mpi import MPIComm, Parallelization, receive, send, serial_comm
from gpaw.spinorbit import projected_soc, soc


def calculate_soc(mykpts,
                  kpt_comm,
                  band_comm,
                  theta: float = 0.0,  # degrees
                  phi: float = 0.0,  # degrees
                  ):
    # <phi_i|dV_adr / r * L_v|phi_j>
    setups = dft.setups
    dVL_avii = {a: soc(setups[a],
                       dft.pot_calc.xc,
                       D_sii,
                       ignore_xc_potential) * scale
                for a, D_sii in dft.density.D_asii.items()}

    if projected:
        dVL_avii = {a: projected_soc(dVL_vii, theta=theta, phi=phi)
                    for a, dVL_vii in dVL_avii.items()}


def calculate_soc(dft: DFT,
                  *,
                  n1: int | None = None,
                  n2: int | None = None,
                  eigenvalues: np.ndarray | None = None,  # eV
                  ):
    """..."""
    if eigenvalues is None:
        eig_skn, occ_skn = dft.ibzwfs.get_all_eigs_and_occs()
    else:
        eig_skn = eigenvalues

    n1 = n1 or 0
    n2 = n2 or 0
    if n2 <= 0:
        n2 += eig_skn.shape[2]

    ibz = dft.ibzwfs.ibz
    nbzkpts = len(ibz.bz)
    nspins = dft.ibzwfs.nspins
    comm = dft.comm
    p = Parallelization(comm.size, nbzkpts)
    p.set(band=1)
    comms = p.build_communicators()
    kpt_comm = comms['k']
    domain_comm = comms['b']

    mynk = (nbzkpts + kpt_comm.size - 1) // kpt_comm.size
    K1 = min(kpt_comm.rank * mynk, nbzkpts)
    K2 = min((kpt_comm.rank + 1) * mynk, nbzkpts)

    who = defaultdict(set)
    for K in range(nbzkpts):
        k = ibz.bz2ibz_K[K]
        for spin in range(nspins):
            from_rank = dft.ibzwfs.rank_ks[k, spin]
            to_rank = K // mynk * domain_comm.size
            who[from_rank].add((to_rank, k, spin))

    P_ksani = {}
    requests = []
    for wfs in dft.ibzwfs:
        P_ani = wfs.P_ani.gathergather()
        if P_ani is not None:
            for r, k, spin in who[comm.rank]:
                if wfs.k == k and wfs.spin == spin:
                    if r == comm.rank:
                        P_ksani[(k, spin)] = P_ani
                    else:
                        req = comm.send(P_ani.data, r, block=False)
                        requests.append(req)

    for rank, what in who.items():
        for r, k, spin in what:
            if r == comm.rank:
                P_ani = ...
                comm.receive(P_ani.data, r)
                P_ksani[(k, spin)] = P_ani
    comm.waitall(requests)

    for K in range(K1, K2):
        k = ibz.bz2ibz_K[K]
        for s in range(nspins):
            rank = dft.ibzwfs.rank_ks[k, s]
            if rank != comm.rank:
                kpt = KPointData.receive(rank)
            mykpts.append(kpt)
    for K in range(nbzkpts):
        k = ibz.bz2ibz_K[K]
        for s in range(nspins):
            rank = dft.ibzwfs.rank_ks[k, s]
            if rank != comm.rank:
                KPointData.send(dft.ibzwfs, k, s, rank)
    for kpt in mykpts:
        kpt.wait()

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


def add_soc(dVL_avii: dict[int, Array3D],
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


def hmm(theta: float = 0.0,
        phi: float = 0.0):

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
    return s_vss


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
