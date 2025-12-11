import numpy as np
from gpaw.rotation import rotation
from gpaw.utilities import unpack_hermitian, unpack_density


class PolynomialProjectable:
    def __init__(self, cell_cv, weights_c, normalize=True):
        self.cell_cv = cell_cv
        if normalize:
            vector_v = weights_c @ cell_cv
            self.weights_c = weights_c / np.linalg.norm(vector_v)
        else:
            self.weights_c = weights_c

    def dot(self, other):
        return np.dot(self.weights_c @ self.cell_cv,
                      other.weights_c @ other.cell_cv)

    def operation(self, op_cc, w_c):
        # op_cc is in fractional coordinates, use directly
        op_cc_int = np.asarray(np.round(op_cc), dtype=np.int64)
        assert np.allclose(op_cc, op_cc_int)
        return PolynomialProjectable(
            self.cell_cv, op_cc_int @ self.weights_c, normalize=False
        )


class LinearCombinationProjectable:
    def __init__(self, w_x, projectables_x):
        self.w_x = w_x
        self.projectables_x = projectables_x

    def dot(self, other):
        s = 0.0
        for w, projectable in zip(self.w_x, self.projectables_x):
            for w2, projectable2 in zip(other.w_x, other.projectables_x):
                s += w * np.conjugate(w2) * projectable.dot(projectable2)
        return s

    def operation(self, op_cc, w_c):
        return LinearCombinationProjectable(
            self.w_x,
            [projectable.operation(op_cc, w_c=w_c)
             for projectable in self.projectables_x],
        )


class Projectable:
    def __init__(self, calc, cell_cv, wf, pseudo_wf=True):
        self.calc = calc
        self.cell_cv = cell_cv
        self.wf = wf
        self.pseudo_wf = pseudo_wf

    @classmethod
    def from_calc(cls, calc, n, pseudo_wf=True):
        if pseudo_wf:
            gamma = next(iter(calc.dft.ibzwfs))
            wf = gamma.psit_nX[n]  # get_pseudo_wave_function(n, grid_spacing=0.05)
        else:
            raise NotImplementedError()
            wf = calc.dft.ibzwfs.get_all_electron_wave_function(n, grid_spacing=0.05)
        return Projectable(calc, calc.atoms.cell, wf, pseudo_wf=pseudo_wf)

    def dot(self, projectable):
        result = self.wf.integrate(projectable.wf)
        return result

    def operation(self, op_cc, w_c):
        # For reciprocal space (plane waves), we need the transpose of the
        # real-space rotation because reciprocal vectors transform with (R^-1)^T = (R^T)^-1
        # Since op_cc is already integer in fractional coords,
        # just transpose it.
        op_cc_reciprocal = op_cc.T.copy()
        op_cc_int = np.asarray(np.round(op_cc_reciprocal), dtype=np.int64)
        assert np.allclose(op_cc_reciprocal, op_cc_int), (
            f"Operation matrix is not integer: {op_cc_reciprocal}"
        )

        if self.pseudo_wf:
            wf2 = self.wf.transform(op_cc_int)
            wf2.data *= np.exp(2j * np.pi * w_c @ self.wf.desc.indices_cG)
            return Projectable(self.calc, self.cell_cv, wf2)
        else:
            # This one does not in fact work
            raise NotImplementedError()
            wf = self.wf.copy()
            wf.symmetrize([op_cc_int], np.array([[0, 0, 0]], dtype=np.int64))
            return Projectable(self.calc, self.cell_cv, wf)


class PaniProjectable:
    """Projectable for point group analysis using PAW projections.
    Parameters
    ----------
    P_ai : dict
        Dictionary mapping atom index to projection coefficients P_i
        for this band.
        P_i is a 1D array of length ni (number of projectors for atom a).
    atoms : ase.Atoms
        The atoms object defining positions.
    setups : gpaw.setup.Setups
        GPAW setups containing l_j, n_j, N0_p for each atom type.
    """
    def __init__(self, P_ai, atoms, setups):
        self.P_ai = P_ai
        self.atoms = atoms
        self.setups = setups
        self.norm = self.dot(self) ** 0.5
        self.P_ai = {a: P_i/ self.norm for a, P_i in P_ai.items()}

    @staticmethod
    def compute_atom_mapping(atoms, op_scc, w_c, tol=1e-1):
        """Compute atom mapping under symmetry operations.

        For each symmetry operation s and atom a, finds which atom b
        satisfies: op_scc[s] @ spos_ac[a] + w_c ≈ spos_ac[b]
        (modulo lattice).

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object.
        op_scc : ndarray
            Symmetry operations in fractional coordinates. Shape (nsym, 3, 3).
        w_c : ndarray
            Translation vector in fractional coordinates. Shape (3,).
        tol : float
            Tolerance for atom position matching.

        Returns
        -------
        a_sa : ndarray
            Atom mapping array. Shape (nsym, natoms).
            a_sa[s, a] = b means operation s maps atom a to atom b.
        """
        spos_ac = atoms.get_scaled_positions()
        # Wrap to [0,1)
        spos_ac = spos_ac % 1.0 % 1.0
        natoms = len(atoms)
        nsym = len(op_scc)
        a_sa = np.zeros((nsym, natoms), dtype=int)

        for s, op_cc in enumerate(op_scc):
            for a in range(natoms):
                # Apply symmetry operation: r' = W_cc @ r + w_c
                spos_c = op_cc @ spos_ac[a] + w_c
                spos_c = spos_c % 1.0 % 1.0

                # Find which atom this maps to (handle PBC wrapping)
                diff_ac = spos_ac - spos_c
                diff_ac -= np.round(diff_ac)  # Wrap to [-0.5, 0.5]
                dist_a = np.linalg.norm(diff_ac, axis=1)
                b = np.argmin(dist_a)

                if dist_a[b] > tol:
                    raise ValueError(
                        f"Symmetry operation {s} does not map atom {a} "
                        f"to any atom. Min distance: {dist_a[b]:.6f}")
                a_sa[s, a] = b
        return a_sa

    def dot(self, other):
        """Calculate overlap <self|other> using PAW metric.

        The overlap includes the atomic PAW correction N0_p as the metric:
        <psi|psi'> = sum_a P_i^* N0_p P'_i
        N0_p is overlap of the all-electron partial waves φ_i and φ_j
        inside the augmentation sphere.
        """
        s = 0.0
        for a, P_i in self.P_ai.items():
            if a in other.P_ai:
                # unpack or unpack2 N0_p.
                N0_ii = unpack_hermitian(self.setups[a].N0_p)
                s += np.vdot(P_i, N0_ii @ other.P_ai[a])
        return s

    def operation(self, op_cc, w_c):
        """Apply symmetry operation to create R|psi>.

        Parameters
        ----------
        op_cc : ndarray
            Rotation matrix R in fractional coordinates (3x3).
        w_c : ndarray
            Translation vector in fractional coordinates (3,).

        Returns
        -------
        PaniProjectable
            New state representing R|psi>.
        """
        a_sa = self.compute_atom_mapping(self.atoms,
                                         op_cc[np.newaxis, :, :],
                                         w_c=w_c)
        map_a = {a: a_sa[0, a] for a in range(len(self.atoms))}

        # Rotate coefficients
        # Convert to Cartesian for Wigner rotation
        cell_cv = np.array(self.atoms.cell)
        op_vv = (cell_cv.T @ op_cc @ np.linalg.inv(cell_cv.T))

        # Cache Wigner D-matrices for each l
        D_l = {}
        new_P_ai = {}

        for a in self.P_ai.keys():
            b = map_a[a]
            setup = self.setups[a]
            P_b = self.P_ai[b]
            P_a_new = np.zeros_like(P_b)

            # Iterate over projector blocks (each radial function l_j)
            ni = 0
            for l, n in zip(setup.l_j, setup.n_j):
                nm = 2 * l + 1

                # Get or compute Wigner D-matrix for this l
                if l not in D_l:
                    D_l[l] = rotation(l, op_vv.T)
                D = D_l[l]
                # Apply rotation to the spherical harmonic coefficients
                P_a_new[ni:ni+nm] = D @ P_b[ni:ni+nm]
                ni += nm
            new_P_ai[a] = P_a_new

        return PaniProjectable(new_P_ai, self.atoms, self.setups)
