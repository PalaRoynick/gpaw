from __future__ import annotations

from collections import defaultdict

import numpy as np
from gpaw.mpi import MPIComm, Parallelization
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new.brillouin import IBZ


def distribute_projections(P_ksani: dict[tuple[int, int], AtomArrays],
                           l_aj: list[list[int]],
                           rank_ks: np.ndarray,
                           ibz: IBZ,
                           comm: MPIComm,
                           n1: int,
                           n2: int) -> dict[int, list[AtomArrays]]:
    """dfgklæj
    """
    nbzkpts = len(ibz.bz)
    p = Parallelization(comm, nbzkpts)
    p.set(band=1)
    comms = p.build_communicators()
    kpt_comm = comms['k']
    domain_comm = comms['b']
    mynk = (nbzkpts + kpt_comm.size - 1) // kpt_comm.size

    P_layout = next(iter(P_ksani.values())).layout

    to_r = defaultdict(set)
    for K in range(nbzkpts):
        k = ibz.bz2ibz_K[K]
        for spin, from_rank in enumerate(rank_ks[k]):
            to_rank = K // mynk * domain_comm.size
            to_r[from_rank].add((to_rank, k, spin))

    P2_ksani = {}
    requests = []
    for (k1, spin1), P_ani in P_ksani.items():
        P_ani = P_ani.gathergather()
        if P_ani is not None:
            print(P_ani.data.shape)
            P_ani = P_ani[:, n1:n2]
            print(P_ani.data.shape)
            for r, k, spin in to_r[comm.rank]:
                if k == k1 and spin == spin1:
                    if r == comm.rank:
                        P2_ksani[(k, spin)] = P_ani
                    else:
                        req = comm.send(P_ani.data, r, block=False)
                        requests.append(req)

    print(P2_ksani)
    print(to_r)
    for from_rank, to in to_r.items():
        for to_rank, k, spin in to:
            if to_rank == comm.rank and from_rank != to_rank:
                P_ani = P_layout.empty(n2 - n1)
                comm.receive(P_ani.data, from_rank)
                P2_ksani[(k, spin)] = P_ani
    comm.waitall(requests)

    from gpaw.new.symmetry import SymmetrizationPlan
    symplan = SymmetrizationPlan(ibz.symmetries, l_aj)
    K1 = min(kpt_comm.rank * mynk, nbzkpts)
    K2 = min((kpt_comm.rank + 1) * mynk, nbzkpts)
    nspins = rank_ks.shape[1]
    P_Ksani = []
    if domain_comm.rank == 0:
        for K in range(K1, K2):
            k = ibz.bz2ibz_K[K]
            P_sani = []
            for spin in range(nspins):
                P_ani = symplan.rotate_projections(P2_ksani[(k, spin)],
                                                   ibz.s_K[K],
                                                   ibz.time_reversal_K[K])
                P_sani.append(P_ani)
            P_Ksani.append(P_sani)
    return P_Ksani
