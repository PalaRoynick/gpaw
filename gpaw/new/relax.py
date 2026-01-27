from gpaw.new.ase_interface import GPAW
from ase import Atoms
from gpaw.new.symmetry import Symmetries, create_symmetries_object
import numpy as np
from gpaw.new.ase_interface import ASECalculator


def pretty(C_cv):
    for i in range(3):
        for j in range(3):
            print(f"{C_cv[i, j]:7.2f} ", end="")
        print()


def chol_derivative(A, dA):
    L = np.linalg.cholesky(A)
    Lp = np.linalg.cholesky(A + dA)
    return Lp - L

    # L lower-triangular Cholesky of A
    Linv = np.linalg.inv(L)
    S = Linv @ dA @ Linv.T

    X = np.tril(S)
    X[np.diag_indices_from(X)] *= 0.5

    dL = L @ X
    return dL


def unit_cell_symmetry(C_cv, U_scc):
    print("Symmetries", len(U_scc))
    # Calculate the cell metric
    M_cc = C_cv @ C_cv.T

    # Symmetrize the cell metric
    M_cc = np.einsum("scd,de,sfe->cf", U_scc, M_cc, U_scc, optimize=True) / len(U_scc)

    print("Old cell", C_cv)
    symC_cv = np.linalg.cholesky(M_cc)

    print("New cell", symC_cv)

    # Deformation gradient
    F_vv = np.linalg.inv(C_cv) @ symC_cv

    # Sanity check
    print("Rotated", C_cv @ F_vv)
    print("Symmetric", symC_cv)
    assert np.allclose(C_cv @ F_vv, symC_cv)

    # Do a polar decomposition to rotate the symmetrized cell back
    import scipy

    rot_vv, P_vv = scipy.linalg.polar(F_vv)
    osymC_cv = symC_cv @ rot_vv.T
    print("Old cell like, but symmetrized", osymC_cv)

    # Now we can construct exact Cartesian rotation matrices
    iosymC_cv = np.linalg.inv(osymC_cv).T
    U_svv = np.array([iosymC_cv @ U_cc @ osymC_cv for U_cc in U_scc])

    # Build unit vector in symmetric matrix space
    def e(i, j):
        eps_ij = np.zeros((3, 3))
        eps_ij[i, j] = 1.0
        return eps_ij
        return (eps_ij + eps_ij.T) / 2

    A_blocks = []
    for U_vv in U_svv:
        rows = []
        for i in range(3):
            for j in range(3):
                rows.append((U_vv @ e(i, j) @ U_vv.T - e(i, j)).reshape((9,)))
        A_blocks.append(np.vstack(rows))
    A = np.vstack(A_blocks)
    print(A)
    # Compute null space via SVD
    U, S, Vh = np.linalg.svd(A)
    print(S)
    tol = 1e-6
    null_mask = S < tol
    nullspace = Vh[null_mask]
    print(nullspace)
    dM_zcc = []
    for B in nullspace:
        dof = osymC_cv.T @ B.reshape((3, 3)) @ osymC_cv
        dM_zcc.append(dof)
    dM_zcc = np.array(dM_zcc)

    # Do a QR decomposition, try to get more zeros to coordinates
    basis = np.array(dM_zcc).reshape((-1, 9))
    Q, R = np.linalg.qr(basis)
    dM_zcc = (Q.T @ basis).reshape((-1, 3, 3))

    symC_cv = np.linalg.cholesky(M_cc)

    for z, dM_cc in enumerate(dM_zcc):
        print(f"Tangent {z} of cell")
        print("In metric space")
        pretty(dM_cc)
        print("C_cv:")
        pretty(chol_derivative(M_cc, 1e-5 * dM_cc) @ rot_vv.T)

    return osymC_cv, U_svv, np.array(dM_zcc)


class Relax:
    def __init__(self, *, atoms: Atoms, calc: GPAW, optimizer_factory, symprec):
        if atoms.calc is not None:
            raise ValueError("Do not attach a calculator to Atoms yet.")

        if not isinstance(calc, ASECalculator):
            raise ValueError("Calculator must be new GPAW.")

        self.atoms = atoms
        self.calc = calc
        self.optimizer_factory = optimizer_factory
        self.symprec = symprec

        self.symmetries = create_symmetries_object(self.atoms, tolerance=self.symprec)

        self.C_cv, U_svv, self.dM_zcc = unit_cell_symmetry(
            self.atoms.cell, self.symmetries.rotation_scc
        )

        self.atoms.set_cell(C_cv, scale_atoms=True)
        # Now, with cell (and later atoms) symmetrized, it is safe to assign the calculator
        self.atoms.calc = calc

        self._ndofs = len(dM_zcc)

        self.optimizer = optimizer_factory(self)

    def run(self, *, fmax, smax):
        self.optimizer.run(fmax=fmax)

    def __ase_optimizable__(self):
        return self

    def get_gradient(self):
        grad_z = np.zeros((self.ndofs(),))
        S_vv = self.atoms.get_stress(voight=False)
        iC_cv = np.linalg.inv(self.C_cv)
        for z in range(self.ndofs()):
            dC_vv = 0.5 * iC_cv.T @ self.dM_zcc[z] @ iC_cv
            grad_z[z] = np.sum(np.sum(dC_vv * S_vv))
        return grad_z

    def ndofs(self):
        return self._ndofs


if __name__ == "__main__":
    from ase.build import bulk

    atoms = bulk("Au")
    eps_cc = np.random.rand(3, 3) * 0.000
    atoms.set_cell(atoms.cell @ (np.eye(3) + eps_cc), scale_atoms=True)
    calc = GPAW(mode="pw")
    print(type(calc))
    from ase.optimize.bfgs import BFGS

    relax = Relax(atoms=atoms, calc=calc, optimizer_factory=BFGS, symprec=1e-1)
    relax.run(fmax=0.005, smax=0.001)
