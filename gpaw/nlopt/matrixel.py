from __future__ import annotations
from typing import TYPE_CHECKING
import warnings

from ase.parallel import parprint
from ase.units import alpha
from ase.utils.timing import Timer
from pathlib import Path
import numpy as np

from gpaw.gaunt import gaunt
from gpaw.new.ase_interface import ASECalculator
from gpaw.nlopt.adapters import CollinearGSInfo, NoncollinearGSInfo
from gpaw.nlopt.basic import NLOData
from gpaw.utilities.progressbar import ProgressBar

if TYPE_CHECKING:
    from gpaw.typing import ArrayND


def get_mml(gs: CollinearGSInfo | NoncollinearGSInfo,
            bands: slice,
            spin: int,
            timer: Timer | None = None) -> ArrayND:
    """
    Compute momentum matrix elements.

    Parameters
    ----------
    gs
        Ground state adapter.
    bands
        Range of band indices.
    spin
        Spin channel index (for spin-polarized systems 0 or 1).
    timer
        Timer for monitoring code performance.

    Returns
    -------
    p_qvnn
        Momentum matrix elements for each local q-point.
    """

    # Start the timer
    if timer is None:
        timer = Timer()
    parprint(f'Calculating momentum matrix elements for spin channel {spin}.')

    # Spin input
    assert spin < gs.ns, 'Wrong spin input'

    # Allocate the matrix elements
    ibzwfs = gs.ibzwfs
    master = (ibzwfs.kpt_comm.rank == 0)
    nb = bands.stop - bands.start
    nq = len(ibzwfs.q_k.keys())  # Number of k-points (q-indices) for each core
    p_qvnn = np.empty([nq, 3, nb, nb], dtype=complex)

    # Initial call to print 0 % progress
    if master:
        pb = ProgressBar()

    # Calculate matrix elements in loop over k-points
    for wfs_s in ibzwfs.wfs_qs:
        wfs = gs.get_wfs(wfs_s, spin)

        with timer('Contribution from pseudo wave functions'):
            G_plus_k_Gv, u_nG = gs.get_plane_wave_coefs(wfs, bands, spin)
            p_vnn = np.einsum('Gv,mG,nG->vmn',
                              G_plus_k_Gv, u_nG.conj(), u_nG) * gs.ucvol

        with timer('Contribution from PAW corrections'):
            P_ani = gs.get_paw_projection_coefs(wfs, bands, spin)
            for P_ni, nabla_iiv in zip(P_ani.values(), gs.nabla_aiiv):
                p_vnn -= 1j * np.einsum('mi,nj,ijv->vmn',
                                        P_ni.conj(), P_ni, nabla_iiv)

        p_qvnn[wfs.q] = p_vnn

        if master:
            pb.update(wfs.q / nq)

    if master:
        pb.finish()

    return p_qvnn


def get_vmlsoc(gs: NoncollinearGSInfo,
               bands: slice) -> ArrayND:
    """
    Compute the spin–orbit correction to the velocity matrix elements.

    Parameters
    ----------
    gs
        Ground state adapter.
    bands
        Range of bands.

    Returns
    -------
    vsoc_kvnn
        SOC correction elements to the velocity matrix elements
    """

    parprint('Calculating spin–orbit correction to velocity matrix elements.')

    # Allocate the matrix elements
    ibzwfs = gs.ibzwfs
    master = (ibzwfs.kpt_comm.rank == 0)
    nb = bands.stop - bands.start
    nq = len(ibzwfs.q_k.keys())
    vsoc_qvnn = np.empty([nq, 3, nb, nb], dtype=complex)

    G_vLL = get_gaunt_coefs()
    if master:
        pb = ProgressBar()

    for wfs_s in ibzwfs.wfs_qs:
        vsoc_vnn = np.empty([3, nb, nb], dtype=complex)

        wfs = gs.get_wfs(wfs_s, 0)
        P_up_ani = gs.get_paw_projection_coefs(wfs, bands, 0)
        P_dn_ani = gs.get_paw_projection_coefs(wfs, bands, 1)
        for (setup, dUdr_q), (P_up_ni, P_dn_ni) in zip(zip(gs.setups,
                                                           gs.dUdr_aq),
                                                       zip(P_up_ani.values(),
                                                           P_dn_ani.values())):
            vsoc_vnn += vsoc_1k_1atom(setup, dUdr_q, P_up_ni, P_dn_ni, G_vLL)

        vsoc_qvnn[wfs.q] = vsoc_vnn

        if master:
            pb.update(wfs.q / nq)

    if master:
        pb.finish()

    return vsoc_qvnn


def vsoc_1k_1atom(setup, dUdr_q, P_up_ni, P_dn_ni, G_vLL):

    vsoc_vnn = np.zeros([3, P_up_ni.shape[0], P_up_ni.shape[0]], dtype=complex)

    l_j = setup.l_j
    nj = len(l_j)

    islice1 = slice(0, 0)
    for j1, l1 in enumerate(l_j):
        islice1 = slice(islice1.stop, islice1.stop + 2 * l1 + 1)
        Lslice1 = slice(l1**2, (l1 + 1)**2)
        P1_up_ni = P_up_ni[:, islice1].conj()
        P1_dn_ni = P_dn_ni[:, islice1].conj()

        islice2 = slice(0, 0)
        for j2, l2 in enumerate(l_j):
            islice2 = slice(islice2.stop, islice2.stop + 2 * l2 + 1)
            if l1 == l2:
                continue  # No s-s, p-p or d-d coupling.
            if (l1 == 0 and l2 == 2) or (l1 == 2 and l2 == 0):
                continue  # No s-d or d-s coupling.
            Lslice2 = slice(l2**2, (l2 + 1)**2)
            P2_up_ni = P_up_ni[:, islice2]
            P2_dn_ni = P_dn_ni[:, islice2]

            # Radial part
            dUdr = dUdr_q[j2 + j1 * nj - j1 * (j1 + 1) // 2 if j1 < j2
                          else j1 + j2 * nj - j2 * (j2 + 1) // 2]

            # Angular part
            Gx_ii, Gy_ii, Gz_ii = G_vLL[:, Lslice1, Lslice2]

            # Spin part
            P_sigmax_nnii = (np.einsum('mi,nj->mnij', P1_up_ni, P2_dn_ni)
                             + np.einsum('mi,nj->mnij', P1_dn_ni, P2_up_ni))
            P_sigmay_nnii = - 1j * (
                np.einsum('mi,nj->mnij', P1_up_ni, P2_dn_ni)
                - np.einsum('mi,nj->mnij', P1_dn_ni, P2_up_ni))
            P_sigmaz_nnii = (np.einsum('mi,nj->mnij', P1_up_ni, P2_up_ni)
                             - np.einsum('mi,nj->mnij', P1_dn_ni, P2_dn_ni))

            vsoc_vnn[0] += dUdr * (
                np.einsum('mnij,ij->mn', P_sigmaz_nnii, Gy_ii)
                - np.einsum('mnij,ij->mn', P_sigmay_nnii, Gz_ii))

            vsoc_vnn[1] += dUdr * (
                np.einsum('mnij,ij->mn', P_sigmax_nnii, Gz_ii)
                - np.einsum('mnij,ij->mn', P_sigmaz_nnii, Gx_ii))

            vsoc_vnn[2] += dUdr * (
                np.einsum('mnij,ij->mn', P_sigmay_nnii, Gx_ii)
                - np.einsum('mnij,ij->mn', P_sigmax_nnii, Gy_ii))

    return vsoc_vnn


def get_gaunt_coefs():
    G_LLL = gaunt(1)

    G_vLL = np.zeros([3, 9, 9])
    G_vLL[0] = G_LLL[3]  # < Y_lm | px | Y_lm >
    G_vLL[1] = G_LLL[1]  # < Y_lm | py | Y_lm >
    G_vLL[2] = G_LLL[2]  # < Y_lm | pz | Y_lm >

    return G_vLL


def gather_to_master(p_qvnn, ibzwfs):
    kpt_comm = ibzwfs.kpt_comm
    master = (kpt_comm.rank == 0)

    if not master:
        kpt_comm.send(p_qvnn, 0)
    else:
        rank_k = ibzwfs.rank_k
        nk = len(rank_k)
        nb = p_qvnn.shape[-1]

        p_kvnn = np.empty([nk, 3, nb, nb], complex)

        k_q = np.where(rank_k == 0)[0]
        p_kvnn[k_q] = p_qvnn
        for gather_rank in range(1, kpt_comm.size):
            k_q = np.where(rank_k == gather_rank)[0]
            nq = len(k_q)

            p_qvnn = np.empty([nq, 3, nb, nb], complex)
            kpt_comm.receive(p_qvnn, gather_rank)
            p_kvnn[k_q] = p_qvnn

    if master:
        return p_kvnn
    else:
        return np.array([], dtype=complex)


def make_nlodata(calc: ASECalculator | str | Path,
                 spin_string: str = 'all',
                 add_soc_correction: bool = False,
                 ni: int | None = None,
                 nf: int | None = None) -> NLOData:
    """
    This function calculates and returns all required
    NLO data: w_sk, f_skn, E_skn, p_skvnn.

    Parameters
    ----------
    calc
        Calculator or string/path pointing to a .gpw file.
    spin_string
        String denoting which spin channels to include ('all', 's0' , 's1').
    ni
        First band to compute the mml.
    nf
        Last band to compute the mml (relative to number of bands for nf <= 0).

    Returns
    -------
    NLOData
        Data object carrying required matrix elements for NLO calculations.

    """

    if not isinstance(calc, ASECalculator):
        if not (isinstance(calc, str) or isinstance(calc, Path)):
            raise TypeError('Input must be a calculator or a string / path'
                            'pointing to a calculator.')
        from gpaw.new.ase_interface import GPAW
        calc = GPAW(calc, txt=None, parallel={'domain': 1, 'band': 1})
    assert not calc.symmetry.point_group, \
        'Point group symmetry should be off.'

    gs: CollinearGSInfo | NoncollinearGSInfo
    if calc.dft.state.density.collinear:
        gs = CollinearGSInfo(calc)
    else:
        gs = NoncollinearGSInfo(calc, add_soc_correction)

    # Start the timer
    timer = Timer()

    # Parse spin string
    ns = gs.ns
    if spin_string == 'all':
        spins = list(range(ns))
    elif spin_string == 's0':
        spins = [0]
    elif spin_string == 's1':
        spins = [1]
        assert spins[0] < ns, 'Wrong spin input'
    else:
        raise NotImplementedError

    # Parse band input
    ibzwfs = gs.ibzwfs
    nb_full = ibzwfs.nbands
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb_full
    nf = nb_full + nf if (nf <= 0) else nf
    bands = slice(ni, nf)

    # Memory estimate
    nk = len(ibzwfs.rank_k)  # Total number of k-points
    est_mem = 2 * 3 * nk * (nf - ni)**2 * 16 / 2**30
    parprint(f'At least {est_mem:.2f} GB of memory is required on master.')

    # Get the energy and Fermi-Dirac occupations (data is only in master)
    with timer('Get energies wrt. Fermi level and occupations'):
        E_skn, f_skn = ibzwfs.get_all_eigs_and_occs()

        E_F = ibzwfs.fermi_levels
        assert E_F is not None and len(E_F) == 1
        E_skn -= E_F[0]

        w_sk = np.array([ibzwfs.ibz.weight_k for _ in range(gs.ndensities)])
        w_sk *= gs.bzvol * ibzwfs.spin_degeneracy

    # Compute the momentum matrix elements
    with timer('Compute the momentum matrix elements'):
        p_sqvnn = []
        for spin in spins:
            p_qvnn = get_mml(gs, bands, spin, timer)
            p_sqvnn.append(p_qvnn)
        if not gs.collinear:
            p_sqvnn = [p_sqvnn[0] + p_sqvnn[1]]
    if add_soc_correction:
        assert isinstance(gs, NoncollinearGSInfo)
        if not gs.soc:
            warnings.warn('Ground state was determined without the spin–'
                          'orbit Hamiltonian. Adding the SOC to the velocity '
                          'matrix elements may give unpredictable results.')
        with timer('Compute and add the spin–orbit correction '
                   'to velocity matrix elements'):
            v_qvnn = get_vmlsoc(gs, bands)
            prefactor = alpha**2 * np.sqrt(np.pi / 12)
            p_sqvnn = [p_sqvnn[0] + prefactor * v_qvnn]
    with timer('Gather the data to master'):
        p_skvnn = []
        for p_qvnn in p_sqvnn:
            p_kvnn = gather_to_master(p_qvnn, ibzwfs)
            p_skvnn.append(p_kvnn)

    if ibzwfs.kpt_comm.rank == 0:
        timer.write()

    # Save the output to the file
    return NLOData(w_sk=w_sk,
                   f_skn=f_skn[:, :, bands],
                   E_skn=E_skn[:, :, bands],
                   p_skvnn=np.array(p_skvnn, complex),
                   comm=ibzwfs.kpt_comm)


def get_rml(E_n, p_vnn, pol_v, Etol=1e-6):
    """
    Compute the position matrix elements

    Parameters
    ----------
    E_n
        Band energies wrt. Fermi level.
    p_vnn
        Momentum matrix elements.
    pol_v
        Tensor element.
    Etol
        Tolerance in energy to consider degeneracy.

    Returns
    -------
    r_vnn
        Position matrix elements.
    D_vnn
        Velocity difference matrix elements.
    """

    # Useful variables
    nb = len(E_n)
    r_vnn = np.zeros((3, nb, nb), complex)
    D_vnn = np.zeros((3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    # Loop over components
    for v1 in set(pol_v):
        r_vnn[v1] = p_vnn[v1] / (1j * E_nn)
        r_vnn[v1, zeroind] = 0
        p_n = np.diag(p_vnn[v1])
        D_vnn[v1] = np.tile(p_n[:, None], (1, nb)) - \
            np.tile(p_n[None, :], (nb, 1))

    return r_vnn, D_vnn


def get_derivative(E_n, r_vnn, D_vnn, pol_v, Etol=1e-6):
    """
    Compute the generalised derivative of position matrix elements

    Parameters
    ----------
    E_n
        Band energies wrt. Fermi level.
    r_vnn
        Momentum matrix elements.
    D_vnn
        Velocity difference matrix elements.
    pol_v
        Tensor element.
    Etol
        Tolerance in energy to consider degeneracy.

    Returns
    -------
    rd_vvnn
        Generalised derivative of position matrix elements.

    """

    # Useful variables
    nb = len(E_n)
    rd_vvnn = np.zeros((3, 3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    for v1 in set(pol_v):
        for v2 in set(pol_v):
            tmp = (r_vnn[v1] * np.transpose(D_vnn[v2])
                   + r_vnn[v2] * np.transpose(D_vnn[v1])
                   + 1j * np.dot(r_vnn[v1], r_vnn[v2] * E_nn)
                   - 1j * np.dot(r_vnn[v2] * E_nn, r_vnn[v1])) / E_nn
            tmp[zeroind] = 0
            rd_vvnn[v1, v2] = tmp

    return rd_vvnn
