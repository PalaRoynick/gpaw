import numpy as np
from ase.units import Ha

from gpaw.extensions import Extension
from gpaw.typing import Array1D, Array2D, Array3D


U_mM = np.zeros((5, 5), dtype=complex)
U_mM[0, 0] = 1j
U_mM[0, 4] = -1j
U_mM[1, 1] = 1j
U_mM[1, 3] = 1j
U_mM[2, 2] = np.sqrt(2)
U_mM[3, 1] = 1
U_mM[3, 3] = -1
U_mM[4, 0] = 1
U_mM[4, 4] = 1
U_mM /= np.sqrt(2)


class HubbardUJ(Extension):
    def __init__(self,
                 atom_indices: list,
                 *,
                 l: int,
                 U: float,
                 J: float,
                 scale: bool = True):
        self.atom_indices = atom_indices
        assert l == 2
        self.l = l
        nm = 2 * l + 1
        U = self.U = U / Ha  # Hartree
        J = self.J = J / Ha  # Hartree
        self.scale = scale

        # Arrays in basis of complex spherical harmonics!
        a0 = np.load('/home/niflheim/martov/LiFePO4_attempts/a0_float.npy')
        a2 = np.load('/home/niflheim/martov/LiFePO4_attempts/a2_float.npy')
        a4 = np.load('/home/niflheim/martov/LiFePO4_attempts/a4_float.npy')
        Vee_MMMM = a0 * U + (a2 * (112 / 13) + a4 * (70 / 13)) * J
        kronecker_MMMM = np.zeros((nm,) * 4)
        I_MM = np.identity(nm)
        for M_index in range(nm):
            kronecker_MMMM[M_index, M_index] += I_MM
        self.density_MMMM = (2 * Vee_MMMM - np.swapaxes(Vee_MMMM, 1, 3)
                             - (2 * U - J) * kronecker_MMMM)
        self.spin_MMMM = -np.swapaxes(Vee_MMMM, 1, 3) + J * kronecker_MMMM

    def update_non_local_hamiltonian(self,
                                     D_sii,
                                     setup,
                                     atom_index,
                                     dH_sii) -> float:
        if atom_index not in self.atom_indices:
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

        nl = np.argwhere(np.equal(l_j, l))
        if not (len(nl) == 1 or len(nl) == 2):
            raise RuntimeError(f'Setup has {len(nl)} radial partial waves '
                               f'with angular momentum quantum number {l}.'
                               ' Must be 1 or 2 for Hubbard-corrected DFT.')
        scale = self.scale
        if scale and len(nl) == len(np.argwhere(np.array(n_j)[nl] == -1)):
            raise RuntimeError('Hubbard correction cannot be scaled if '
                               'there is no bounded partial waves.')

        nspins = len(D_sii)
        assert nspins == 4

        # j-indices which have the correct angular momentum quantum number
        nl = np.where(np.equal(l_j, l))[0]

        nm_j = 2 * np.array(l_j) + 1
        nm = nm_j[nl[0]]

        # Get relevant entries of the density matrix
        i1 = slice(nm_j[:nl[0]].sum(), nm_j[:nl[0]].sum() + nm)

        U = self.U
        J = self.J
        density_MMMM = self.density_MMMM
        spin_MMMM = self.spin_MMMM

        eU = 0.
        dHU_sii = np.zeros_like(D_sii)

        for s, D_ii in enumerate(D_sii):
            N_mm, dHU_ii = aoom(D_ii, l, l_j, n_j, N0_q, scale)

            # To complex harmonics basis
            N_MM = (U_mM.conj().T @ N_mm) @ U_mM

            if s == 0:
                eU1 = ((U - J) / 2. * N_MM.trace()
                       + np.einsum('mnop,mn,op', density_MMMM, N_MM, N_MM) / 4)

                dHU_MM = ((U - J) / 2. * np.identity(nm)
                          + np.einsum('mnop,op->nm', density_MMMM, N_MM) / 2.)

            else:
                eU1 = np.einsum('mnop,mn,op', spin_MMMM, N_MM, N_MM) / 4.

                dHU_MM = np.einsum('mnop,op->nm', spin_MMMM, N_MM) / 2.

            eU += eU1
            dHU_mm = (U_mM @ dHU_MM) @ U_mM.conj().T

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

            dHU_sii[s] = dHU_ii

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
