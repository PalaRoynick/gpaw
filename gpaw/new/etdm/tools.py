import numpy as np
import scipy.linalg as lalg

# put mom excite here

def get_ndim(ibzwfs):

    return ibzwfs.nbands


def get_n_occ(f_n):
    occupied = f_n > 1.0e-10
    n_occ = len(f_n[occupied])

    return n_occ


def sort_orbitals_according_to_occ(ibzwfs, update_mom=False):

    """
    Sort orbitals according to the occupation
    numbers so that there are no holes in the
    distribution of occupation numbers
    :return:
    """

    # for LCAO
    for wfs in ibzwfs:
        changedocc, ind = \
            sort_orbitals_according_to_occ_wfs(ibzwfs, wfs,
                                               update_mom=update_mom)


def sort_orbitals_according_to_occ_wfs(ibzwfs, wfs, update_mom=False, 
                                       update_eps=True):

    changedocc = False
    ind = np.array([])

    occupied = wfs.occ_n > 1.0e-10
    n_occ = sum(occupied)
    if n_occ != 0.0 and np.min(wfs.occ_n[:n_occ]) == 0:
        ind_occ = np.argwhere(occupied)
        ind_unocc = np.argwhere(~occupied)
        ind = np.vstack((ind_occ, ind_unocc))
        ind = np.squeeze(ind)

        wfs.C_nM.data[np.arange(len(ind)), :] = wfs.C_nM.data[ind, :]
        wfs._P_ani = None

        wfs._occ_n = wfs._occ_n[ind]

        if update_eps:
            wfs._eig_n[:] = wfs._eig_n[ind]

        #if update_mom:
            # OccupationsMOM.numbers needs
            # to be updated after sorting
        # update_mom_numbers(ibzwfs, wfs)

        changedocc = True

    return changedocc, ind


def update_mom_numbers(ibzwfs, wfs):
    # check if it works properly
    if ibzwfs.collinear and ibzwfs.nspins == 1:
        degeneracy = 2
    else:
        degeneracy = 1
    # is it translated like this?
    wfs._occ_n = \
        (wfs._occ_n / wfs.weight * degeneracy)


def sort_orbitals_according_to_energies(hamiltonian, ibzwfs):
    """
    Sort orbitals according to the eigenvalues or
    the diagonal elements of the Hamiltonian matrix
    """

    for wfs in ibzwfs:
        orb_energies = wfs._eig_n

        ind = np.argsort(orb_energies)
        wfs._eig_n[np.arange(len(ind))] = orb_energies[ind]

        wfs.C_nM.data[np.arange(len(ind)), :] = wfs.C_nM.data[ind, :]
        wfs._P_ani = None

        assert len(ind) == len(wfs.occ_n)
        wfs._occ_n = wfs._occ_n[ind]

        #occ_name = getattr(wfs.occupations, "name", None)
        #if occ_name == 'mom':
            # OccupationsMOM.numbers needs to be updated
            # after sorting
        #    update_mom_numbers(wfs, kpt)


def excite(calc, i, a, spin=(0, 0), sort=False):
    """Helper function to initialize a variational excited state calculation.

    Promote an electron from homo + i of k-point spin[0] to lumo + a of
    k-point spin[1].

    Parameters
    ----------
    calc: GPAW instance
        GPAW calculator object.
    i: int
        Subtract 1 from the occupation number of the homo + i orbital of
        k-point spin[0]. E.g. if i=-1, an electron is removed from the
        homo - 1 orbital.
    a: int
        Add 1 to the occupation number of the lumo + a orbital of k-point
        spin[1]. E.g. if a=1, an electron is added to the lumo + 1 orbital.
    spin: tuple of two int
        spin[0] is the k-point from which an electron is removed and spin[1]
        is the k-point where an electron is added.
    sort: bool
        If True, sort the orbitals in the wfs object according to the new
        occupation numbers, and modify the f_n attribute of the kpt objects.
        Default is False.

    Returns
    -------
    list of numpy.ndarray
        List of new occupation numbers. Can be supplied to
        mom.prepare_mom_calculation to initialize an excited state calculation
        with MOM.
    """
    f_sn = [calc.get_occupation_numbers(spin=s).copy()
            for s in range(calc.wfs.nspins)]

    f_n0 = np.asarray(f_sn[spin[0]])
    lumo = len(f_n0[f_n0 > 0])
    homo = lumo - 1

    f_sn[spin[0]][homo + i] -= 1.0
    f_sn[spin[1]][lumo + a] += 1.0

    if sort:
        for s in spin:
            for wfs in calc.ibzwfs_wfs_u:
                if wfs.s == s:
                    wfs.myocc_n = f_sn[s]
                    changedocc = sort_orbitals_according_to_occ_wfs(
                        calc.ibzwfs, wfs, update_mom=False)[0]
                    if changedocc:
                        f_sn[s] = wfs.myocc_n

    return f_sn


def loewdin_lcao(C_nM, S_MM):
    """
    Loewdin based orthonormalization
    for LCAO mode

    C_nM <- sum_m C_nM[m] [1/sqrt(S)]_mn
    S_mn = (C_nM[m].conj(), S_MM C_nM[n])

    :param C_nM: LCAO coefficients
    :param S_MM: Overlap matrix between basis functions
    :return: Orthonormalized coefficients so that new S_mn = delta_mn
    """

    ev, S_overlapp = np.linalg.eigh(C_nM.conj() @ S_MM @ C_nM.T)
    ev_sqrt = np.diag(1.0 / np.sqrt(ev))

    S = S_overlapp @ ev_sqrt @ S_overlapp.T.conj()

    return S.T @ C_nM


def gramschmidt_lcao(C_nM, S_MM):
    """
    Gram-Schmidt orthonormalization using Cholesky decomposition
    for LCAO mode

    :param C_nM: LCAO coefficients
    :param S_MM: Overlap matrix between basis functions
    :return: Orthonormalized coefficients so that new S_mn = delta_mn
    """

    S_nn = C_nM @ S_MM.conj() @ C_nM.T.conj()
    L_nn = lalg.cholesky(S_nn, lower=True,
                         overwrite_a=True, check_finite=False)
    return lalg.solve(L_nn, C_nM)



def vec2skewmat(a_vec, dim, ind_up, dtype):
    """
    Convert parameter vector to full skew-Hermitian matrix.

    Parameters
    ----------
    a_vec : ndarray
        1D vector of independent upper-triangular entries.
    dim : int
        Dimension of the full matrix.
    ind_up : tuple
        Indices of the upper-triangular elements.
    dtype : type
        float or complex.

    Returns
    -------
    a_mat : ndarray
        Full skew-Hermitian matrix with A^\\dagger = -A.
    """
    a_mat = np.zeros((dim, dim), dtype=dtype)
    a_mat[ind_up] = a_vec
    a_mat -= a_mat.T.conj()  # enforce skew-Hermitian property
    np.fill_diagonal(a_mat, a_mat.diagonal() * 0.5)  # fix diagonal factor
    return a_mat


def random_a(shape, dtype, seed=None):
    """
    Generate random parameter vector for skew-Hermitian matrices.

    Parameters
    ----------
    shape : tuple
        Shape of the vector.
    dtype : type
        float or complex.

    Returns
    -------
    a : ndarray
        Random vector of given dtype.
    """
    if seed:
        np.random.seed(seed)
    a = np.random.random_sample(shape)
    if dtype == complex:
        a = a.astype(complex)
        a += 1.0j * np.random.random_sample(shape)
    return a
