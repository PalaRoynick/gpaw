import numpy as np
from ase.units import Ha

from gpaw.new.extensions import Extension
from gpaw.typing import Array1D, Array2D, Array3D


#: Supported corrective functionals.
#:
#: 'dudarev'
#:     Simplified rotationally invariant DFT+U, Dudarev et al., Phys. Rev. B
#:     57, 1505 (1998).  A single parameter enters, and it is the *effective*
#:     one: U here means U_eff = U - J.
#: 'himmetoglu'
#:     DFT+U+J, Eq. (47) of Himmetoglu et al., Int. J. Quantum Chem. 114, 14
#:     (2014), introduced in Phys. Rev. B 84, 115108 (2011).  U and J0 enter
#:     independently; J0 is the on-site exchange that stabilises spin
#:     polarisation.  Equivalent to Hubbard_J0 in Quantum ESPRESSO.
SCHEMES = ('dudarev', 'himmetoglu')


class HubbardUJ(Extension):
    """Hubbard correction applied to the l-manifold of selected atoms.

    Parameters
    ----------
    indices:
        Atom indices the correction acts on.
    l:
        Angular momentum of the corrected manifold (2 for d, 3 for f).
    U:
        Hubbard parameter in eV.  For scheme='dudarev' this is the
        *effective* U_eff = U - J; for scheme='himmetoglu' it is the bare U.
    scheme:
        Which corrective functional to use; see SCHEMES.  Required, because
        the meaning of U differs between them and a silent default would
        make published U values ambiguous.
    J0:
        On-site exchange in eV.  Required for scheme='himmetoglu', and must
        not be given for scheme='dudarev'.

        Named J0, following Quantum ESPRESSO, because it is *not* the
        Slater-integral J of the Liechtenstein scheme: it is defined by its
        own atomic average (Eq. 47 of the review) rather than by Eq. (12),
        which uses a different prefactor and excludes the m = m' diagonal.
        A literature J computed for a Liechtenstein calculation must not be
        passed here without checking the convention.
    scale:
        Scale the inner products so that the bounded partial wave has unit
        norm.
    """

    def __init__(self,
                 indices: list,
                 *,
                 l: int,
                 U: float,
                 scheme: str,
                 J0: float | None = None,
                 scale: bool = True):
        if scheme not in SCHEMES:
            raise ValueError(
                f'Unknown Hubbard scheme {scheme!r}. '
                f'Choose one of {SCHEMES}.')

        if scheme == 'dudarev':
            if J0 is not None:
                raise ValueError(
                    "scheme='dudarev' takes no J0: its single parameter is "
                    'already the effective U_eff = U - J.  Either pass '
                    'U=U_eff and no J0, or use scheme=\'himmetoglu\' to '
                    'treat U and J0 independently.')
            J0 = 0.0
        else:
            if J0 is None:
                raise ValueError(
                    "scheme='himmetoglu' requires J0.  Pass J0=0.0 "
                    'explicitly if you want no exchange term, or use '
                    "scheme='dudarev'.")

        self.indices = indices
        self.l = l
        self.scheme = scheme
        self.U = U / Ha
        self.J0 = J0 / Ha
        self.scale = scale

    def descriptions(self):
        """Lines recorded in the output file, so that a completed
        calculation states unambiguously which functional was used."""
        yield f'Hubbard: {{scheme: {self.scheme},\n'
        yield f'          l: {self.l},\n'
        if self.scheme == 'dudarev':
            yield f'          U_eff: {self.U * Ha},  # eV\n'
        else:
            yield f'          U: {self.U * Ha},  # eV\n'
            yield f'          J0: {self.J0 * Ha},  # eV\n'
        yield f'          scale: {bool(self.scale)}}}'

    def update_non_local_hamiltonian(self,
                                     D_sii,
                                     setup,
                                     atom_index,
                                     dH_sii) -> float:
        if atom_index not in self.indices:
            return 0.0

        eU, dHU_sii = self.calculate(D_sii, setup.l_j, setup.n_j, setup.N0_q)
        dH_sii += dHU_sii
        return eU.real

    def calculate(self,
                  D_sii: Array3D,
                  l_j: list[int],
                  n_j: list[int],
                  N0_q: Array1D) -> tuple[float, Array3D]:
        l = self.l
        scale = self.scale

        nl = np.argwhere(np.equal(l_j, l))
        if not (len(nl) == 1 or len(nl) == 2):
            raise RuntimeError(f'Setup has {len(nl)} radial partial waves '
                               f'with angular momentum quantum number {l}.'
                               ' Must be 1 or 2 for Hubbard-corrected DFT.')
        if scale and len(nl) == len(np.argwhere(np.array(n_j)[nl] == -1)):
            raise RuntimeError('Hubbard correction cannot be scaled if '
                               'there is no bounded partial waves.')

        return hubbard(D_sii, U=self.U, J0=self.J0,
                       l=l, l_j=l_j, n_j=n_j, N0_q=N0_q, scale=scale)


def _scatter(dHU_ii, dHU_mm, nl, nm_j):
    """Multiply the aoom projection structure in-place by the on-site
    potential matrix dHU_mm, for 1 or 2 radial partial waves."""
    nm = nm_j[nl[0]]
    i1 = slice(nm_j[:nl[0]].sum(), nm_j[:nl[0]].sum() + nm)
    if len(nl) == 1:
        dHU_ii[i1, i1] *= dHU_mm
    elif len(nl) == 2:
        i2 = slice(nm_j[:nl[1]].sum(), nm_j[:nl[1]].sum() + nm)
        dHU_ii[i1, i1] *= dHU_mm
        dHU_ii[i1, i2] *= dHU_mm
        dHU_ii[i2, i1] *= dHU_mm
        dHU_ii[i2, i2] *= dHU_mm
    else:
        raise NotImplementedError


def hubbard(D_sii: Array3D,
            U: float,
            J0: float,
            l: int,
            l_j: list[int],
            n_j: list[int],
            N0_q: Array1D,
            scale: bool) -> tuple[float, Array3D]:
    nspins = len(D_sii)

    if nspins == 1 and J0 != 0.0:
        raise ValueError(
            'A non-zero J0 requires a spin-polarised calculation: the J0 '
            'term stabilises spin polarisation, which a spin-restricted '
            'calculation cannot represent.  Use spinpol=True, or '
            "scheme='dudarev' with U=U_eff.")

    # j-indices which have the correct angular momentum quantum number
    nl = np.where(np.equal(l_j, l))[0]

    nm_j = 2 * np.array(l_j) + 1
    nm = nm_j[nl[0]]

    eU = 0.
    dHU_sii = np.zeros_like(D_sii)

    if nspins == 4:
        for s, D_ii in enumerate(D_sii):
            N_mm, dHU_ii = aoom(D_ii, l, l_j, n_j, N0_q, scale)
            N_mm = N_mm / 2 * nspins
            N_mm = N_mm / 2.0
            # The J0 term must be a genuine matrix trace, not a sum of
            # squared elements.  D_sii is complex on this path, and
            # sum_ij D_ij**2 = Tr[D D*] is basis dependent for complex D,
            # whereas Tr[D D] = sum_ij |D_ij|**2 is not.  The two agree
            # only when D is real.
            if s == 0:
                eU1 = ((U - J0) / 2. * (N_mm - 0.5 * N_mm @ N_mm).trace()
                       + J0 / 4. * (N_mm @ N_mm).trace())
                dHU_mm = ((U - J0) / 2. * (np.eye(nm) - N_mm.T)
                          + J0 / 2. * N_mm.T)
            else:
                eU1 = (-(U - J0) / 2. * (0.5 * N_mm @ N_mm).trace()
                       - J0 / 4. * (N_mm @ N_mm).trace())
                dHU_mm = -(U - J0) / 2. * N_mm.T - J0 / 2. * N_mm.T
            eU += eU1
            _scatter(dHU_ii, dHU_mm, nl, nm_j)
            dHU_sii[s] = dHU_ii
        return eU, dHU_sii

    # Collinear (nspins == 1 or 2): Eq. (47) of Himmetoglu et al. (2014),
    # E = sum_s (U-J0)/2 Tr[n^s(1-n^s)] + sum_s J0/2 Tr[n^s n^-s]
    N_smm = []
    struct_sii = []
    for D_ii in D_sii:
        N_mm, dHU_ii = aoom(D_ii, l, l_j, n_j, N0_q, scale)
        N_mm = N_mm / 2 * nspins
        N_smm.append(N_mm)
        struct_sii.append(dHU_ii)

    for s in range(nspins):
        N_mm = N_smm[s]
        N_opp_mm = N_smm[1 - s] if nspins == 2 else N_mm

        # D_sii is guaranteed real on the collinear code path, so
        # n^s is real symmetric and no transposes/conjugates are needed
        # here.  (If that guarantee is ever relaxed, dE/dn_{mm'} picks up
        # n_{m'm}: use N_mm.T and N_opp_mm.T below, and .real on eU1.)
        eU1 = ((U - J0) / 2. * (N_mm - N_mm @ N_mm).trace()
               + J0 / 2. * (N_mm @ N_opp_mm).trace())

        # Note the coefficient of the cross term is J0, not J0/2.
        dHU_mm = (U - J0) / 2. * (np.eye(nm) - 2 * N_mm) + J0 * N_opp_mm

        eU += eU1
        if nspins == 1:
            eU += eU1

        _scatter(struct_sii[s], dHU_mm, nl, nm_j)
        dHU_sii[s] = struct_sii[s]

    return eU, dHU_sii


def aoom(D_ii: Array2D,
         l: int,
         l_j: list[int],
         n_j: list[int],
         N0_q: Array1D,
         scale: bool = True) -> tuple[Array2D, Array2D]:
    """
    This function returns the atomic orbital occupation matrix (aoom) for a
    given l quantum number.

    The submatrix / submatrices of the density matrix (D_ii) for the
    selected l quantum number are determined and summed together which
    represents the orbital occupation matrix. For l=2 this is a 5x5 matrix.

    If scale = True, the inner products are scaled such that the inner product
    of the bounded partial waves is 1.
    """

    # j-indices which have the correct angular momentum quantum number
    nl = np.where(np.equal(l_j, l))[0]

    nm_j = 2 * np.array(l_j) + 1
    nm = nm_j[nl[0]]

    # Get relevant entries of the density matrix
    i1 = slice(nm_j[:nl[0]].sum(), nm_j[:nl[0]].sum() + nm)

    dHU_ii = np.zeros_like(D_ii)
    if len(nl) == 2:
        # First get q-indices for the radial inner products
        q1 = nl[0] * len(l_j) - (nl[0] - 1) * nl[0] // 2  # Bounded-bounded
        q2 = nl[1] * len(l_j) - (nl[1] - 1) * nl[1] // 2  # Unbounded-unbounded
        q12 = q1 + nl[1] - nl[0]  # Bounded-unbounded

        # If the Hubbard correction should be scaled, the three inner products
        # will be divided by the inner product of the bounded partial wave,
        # increasing the value of these inner products since 0 < N0_q[q1] < 1
        if scale:
            if n_j[nl[1]] == -1:
                N1 = 1
                N2 = N0_q[q2] / N0_q[q1]
                N12 = N0_q[q12] / N0_q[q1]
            else:
                N1 = 1
                N2 = 1
                N12 = N0_q[q12] / np.sqrt(N0_q[q1] * N0_q[q2])
        else:
            N1 = N0_q[q1]
            N2 = N0_q[q2]
            N12 = N0_q[q12]

        # Get the entries in the density matrix of the unbounded partial waves
        i2 = slice(nm_j[:nl[1]].sum(), nm_j[:nl[1]].sum() + nm)

        # Scale and add the four submatrices to get the occupation matrix
        N_mm = D_ii[i1, i1] * N1 + D_ii[i2, i2] * N2 + (D_ii[i1, i2]
                                                        + D_ii[i2, i1]) * N12

        dHU_ii[i1, i1] = N1
        dHU_ii[i1, i2] = N12
        dHU_ii[i2, i1] = N12
        dHU_ii[i2, i2] = N2

        return N_mm, dHU_ii
    elif len(nl) == 1:
        q1 = nl[0] * len(l_j) - (nl[0] - 1) * nl[0] // 2
        if scale:
            N1 = 1
        else:
            N1 = N0_q[q1]

        N_mm = D_ii[i1, i1] * N1
        dHU_ii[i1, i1] = N1
        return N_mm, dHU_ii
    else:
        raise NotImplementedError(f'Setup has {len(nl)} partial waves with '
                                  f'angular momentum quantum number {l}. '
                                  'Must be 1 or 2 for DFT+U correction.')
