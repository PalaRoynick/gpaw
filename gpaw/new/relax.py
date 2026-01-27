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
    eps = 1e-5
    L = np.linalg.cholesky(A)
    Lp = np.linalg.cholesky(A + eps * dA)
    return (Lp - L) / eps

    # L lower-triangular Cholesky of A
    Linv = np.linalg.inv(L)
    S = Linv @ dA @ Linv.T

    X = np.tril(S)
    X[np.diag_indices_from(X)] *= 0.5

    dL = L @ X
    return dL


def unit_cell_symmetry(C_cv, U_scc):
    print('Original cell', Atoms(cell=C_cv).cell.lengths(), Atoms(cell=C_cv).cell.angles())
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
    print('rot_vv', rot_vv)
    print("Old cell like, but symmetrized", osymC_cv)
    print(Atoms(cell=osymC_cv).cell.angles())
    print(Atoms(cell=osymC_cv).cell.lengths())
    # Now we can construct exact Cartesian rotation matrices
    iosymC_cv = np.linalg.inv(osymC_cv)
    U_svv = np.array([osymC_cv.T @ U_cc.T @ iosymC_cv.T for U_cc in U_scc])

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
    # Compute null space via SVD
    U, S, Vh = np.linalg.svd(A)
    tol = 1e-6
    null_mask = S < tol
    nullspace = Vh[null_mask]
    dM_zcc = []
    dM_zvv = []
    for B in nullspace:
        dM_vv = B.reshape((3,3))
        dof = osymC_cv @ dM_vv @ osymC_cv.T
        dM_zcc.append(dof)
        dM_zvv.append(rot_vv @ dM_vv @ rot_vv.T)
    dM_zcc = np.array(dM_zcc).reshape((-1, 3, 3))
    dM_zvv = np.array(dM_zvv).reshape((-1, 3, 3))

    # Do a QR decomposition, try to get more zeros to coordinates
    # basis = np.array(dM_zcc).reshape((-1, 9))
    # Q, R = np.linalg.qr(basis)
    # dM_zcc = (Q.T @ basis).reshape((-1, 3, 3))

    symC_cv = np.linalg.cholesky(M_cc)

    for z, dM_cc in enumerate(dM_zcc):
        print(f"Tangent {z} of cell")
        print("In metric space")
        pretty(dM_cc)
        print("C_cv:")
        pretty(chol_derivative(M_cc, dM_cc) @ rot_vv.T)

    return M_cc, osymC_cv, U_svv, dM_zcc, dM_zvv, rot_vv


class Relax:
    def __init__(self, *, atoms: Atoms, calc: GPAW, optimizer_factory, symprec):
        if atoms.calc is not None:
            raise ValueError("Do not attach a calculator to Atoms yet.")

        # if not isinstance(calc, ASECalculator):
        #    raise ValueError("Calculator must be new GPAW.")

        self.atoms = atoms
        self.calc = calc
        self.optimizer_factory = optimizer_factory
        self.symprec = symprec

        self.symmetries = create_symmetries_object(self.atoms, tolerance=self.symprec)

        self.M_cc, self.C_cv, U_svv, self.dM_zcc, self.dM_zvv, self.rot_vv = unit_cell_symmetry(
            self.atoms.cell, self.symmetries.rotation_scc
        )

        self.atoms.set_cell(self.C_cv, scale_atoms=True)
        # Now, with cell (and later atoms) symmetrized, it is safe to assign the calculator
        self.atoms.calc = calc

        self._ndofs = len(self.dM_zcc)

        self.optimizer = optimizer_factory(self)

        self.value_z = np.zeros((self._ndofs))

    def run(self, *, fmax, smax):
        self.optimizer.run(fmax=fmax)

    def __ase_optimizable__(self):
        return self

    def get_value(self):
        return self.atoms.get_potential_energy()

    def get_x(self):
        return self.value_z.copy()

    def _get_cell(self):
        M_cc = self.M_cc + np.einsum("z,zcd->cd", self.value_z, self.dM_zcc)
        C_cv = np.linalg.cholesky(M_cc) @ self.rot_vv.T
        return Atoms(cell=C_cv).cell

    def set_x(self, x):
        self.value_z[:] = x
        # print('z values', self.value_z)
        M_cc = self.M_cc + np.einsum("z,zcd->cd", self.value_z, self.dM_zcc)
        C_cv = np.linalg.cholesky(M_cc) @ self.rot_vv.T
        # print('Current cell', C_cv)
        # print('Volume', np.linalg.det(C_cv))
        self.atoms.set_cell(C_cv, scale_atoms=True)

    def get_gradient(self):
        if 0:
            eps = 1e-5
            xref = np.array(self.get_x())
            grad_z = np.zeros((self.ndofs(),))
            for z in range(self.ndofs()):
                x = xref.copy()
                x[z] += eps
                self.set_x(x)
                E1 = self.get_value()
                x[z] -= 2 * eps
                self.set_x(x)
                E0 = self.get_value()

                grad_z[z] = (E1 - E0) / (2 * eps)
            self.set_x(xref)
            grad2_z = grad_z.copy()
        
        grad_z = np.zeros(self.ndofs())
        S_vv = self.atoms.get_stress(voigt=False)
        C_cv = np.array(self._get_cell())
        V = np.linalg.det(C_cv)
        Cinv = np.linalg.inv(C_cv)
        
        M_cc = self.M_cc + np.einsum("z,zcd->cd", self.value_z, self.dM_zcc)

        # dE/deps_vv deps_vv/dC_cv dC_cv/dz
        for z in range(self.ndofs()):
            dC_cv = chol_derivative(M_cc, self.dM_zcc[z]) @ self.rot_vv.T
            grad_z[z] = V * np.sum(S_vv * (Cinv @ dC_cv + dC_cv.T @ Cinv.T)/2)
        #print('grad', grad_z)
        #print('fd grad', grad2_z)
        return grad_z

    def converged(self, gradient, fmax):
        cell = self._get_cell()
        print(cell.lengths(), cell.angles())
        return self.gradient_norm(gradient) < fmax

    def gradient_norm(self, grad_z):
        # Go actually to cell metric
        return np.max(np.abs(grad_z))

    def ndofs(self):
        return self._ndofs


if __name__ == "__main__":
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from ase.io.jsonio import read_json
    atoms = read_json('output.json')

    #atoms = Atoms('NaCl',
    #        cell=[4, 4, 4],
    #        positions=[[0, 0, 0],
    #                   [2, 2, 2]],
    #        pbc=True)

    
    #angle = 62
    #c = np.cos(angle / 180 * np.pi)
    #a = atoms.cell.lengths()[0]
    #M_cc = a**2 * np.array([[1, c, c], [c, 1, c], [c, c, 1]])
    #cell_cv = np.linalg.cholesky(M_cc)
    #atoms.set_cell(cell_cv, scale_atoms=True)

    #eps_cc = np.random.rand(3, 3) * 0.0001
    #atoms.set_cell(atoms.cell @ (np.eye(3) + eps_cc), scale_atoms=True)
    calc = GPAW(mode={'name': "pw", 'ecut': 800}, kpts={'size': (4,4,4), 'gamma': True},
                txt='out.txt', xc='PBE',
                convergence={'density':1e-7})

    #calc = EMT()
    from ase.optimize.bfgs import BFGS

    relax = Relax(atoms=atoms, calc=calc, optimizer_factory=BFGS, symprec=1e-1)
    if 0:
        for z in range(2):
            vec = np.zeros((2,))
            relax.set_x(vec)
            E0 = relax.get_value()
            grad = relax.get_gradient()
            vec[z] = 1e-3
            relax.set_x(vec)
            E1 = relax.get_value()
            print("Finite difference grad", (E1 - E0) / 1e-3)
            print("Gotten grad", grad[z])
            print("div", grad[z] / ((E1 - E0) / 1e-3))

    relax.run(fmax=0.005, smax=0.001)

"""
    [ cos(alpha) |v1|^2   
"""
