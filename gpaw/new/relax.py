# TODO:
# [X] Test cholesky derivative
# [X] Add analytical cholesky derivative
# [ ] Precalculate Cholesky derivative
# [ ] Prettier print of atomic degrees of freedom
from sys import argv
from gpaw.new.logger import GREEN, RESET
from gpaw.new.ase_interface import GPAW
from ase import Atoms
from gpaw.new.symmetry import create_symmetries_object
import numpy as np
from dataclasses import dataclass
from gpaw.new.relax_print import (
    pretty,
    pprint_atoms,
    pretty_dofs,
    pretty_header,
    pretty_subheader,
    pretty_atomic_dofs,
)

def minimize_l1(dof_zac):
    nz = len(dof_zac)
    if nz < 2:
        # Nothing to minimize with just 1 dof
        return
    from scipy.linalg import expm
    from scipy.optimize import fmin    

    basis = []
    for i in range(nz):
        for j in range(i + 1, nz):
            b = np.zeros((nz, nz))
            b[i,j] = 1
            b[j,i] = -1
            basis.append(b)

    nb = len(basis)

    def U_zz(x):
        x = np.array(x).reshape((-1, ))
        M = np.zeros((nz, nz))
        for i, e in enumerate(x):
            M += e * basis[i]
        return expm(M)

    def function(x):
        return np.sum(np.abs(np.einsum('zw,wac->zac', U_zz(x), dof_zac)))

    xopt, *args = fmin(function, np.zeros((nb, )), xtol=1e-7, ftol=1e-7)
    print(args)
    dof_zac[:] = np.einsum('zw,wac->zac', U_zz(np.array(xopt)), dof_zac)
    

def chol_derivative(A, dA, L=None):
    """
    Compute the derivative of the Cholesky factorization.

        A = L L^T

    where L is lower-triangular. For a small symmetric perturbation `dA`,
    this function returns an approximation of the differential `dL` of
    the Cholesky factor:

        L + dL ≈ chol(A + dA).

    Either L or A must be given as an input.
    """
    A = (A + A.T) / 2
    dA = (dA + dA.T) / 2

    if A is not None:
        L = np.linalg.cholesky(A)
    Linv = np.linalg.inv(L)
    S = Linv @ dA @ Linv.T
    X = np.tril(S)
    X[np.diag_indices_from(X)] *= 0.5
    return L @ X


def symmetrize_atoms(
    S_ac: np.ndarray, U_scc: np.ndarray,
    f_sc: np.ndarray, atommap_sa, tol=1e-12
):
    """
    Symmetrize fractional atomic coordinates under a space-group.

    Given atomic scaled positions `S_ac` and a set of space-group operations
    (U_scc, f_sc), this function projects the positions onto the symmetry-
    invariant subspace by averaging over all symmetry-related images.

    Parameters
    ----------
    S_ac : ndarray, shape (na, 3)
        Scaled atomic coordinates.

    U_scc : ndarray, shape (ns, 3, 3)
        Rotation matrices.

    f_sc : ndarray, shape (ns, 3)
        Translation vectors.

    atommap_sa : ndarray, shape (ns, na)
        Mapping such that atommap_sa[s, a] gives the index of the atom
        to which atom `a` is mapped by symmetry operation `s`.

    tol : float, optional
        Tolerance for snapping values close to 0 or 1 back to 0.

    Returns
    -------
    Ssym_ac : ndarray, shape (na, 3)
        Symmetrized fractional atomic coordinates in [0, 1).

    Notes
    -----
    The symmetrization is done in the complex phase representation
    exp(2πi x) to correctly average periodic fractional coordinates.
    """
    ns, na = atommap_sa.shape
    Ssym_ac = np.zeros_like(S_ac, dtype=np.complex128)
    for a in range(na):
        for s in range(ns):
            new = U_scc[s].T @ S_ac[a] - f_sc[s]
            Ssym_ac[atommap_sa[s, a]] += np.exp(2j * np.pi * new)
    Ssym_ac = (np.angle(Ssym_ac) / (2 * np.pi)) % 1.0 % 1.0
    Ssym_ac[np.abs(Ssym_ac) < tol] = 0.0
    Ssym_ac[np.abs(Ssym_ac - 1.0) < tol] = 0.0
    return Ssym_ac


@dataclass
class AtomsSymmetries:
    """
    Dataclass to contain symmetry information from atoms.

    This is to set up an interface for spglib/GPAW or whatever
    source of symmetry operations.
    """

    rotation_scc: np.ndarray
    atommap_sa: np.ndarray
    translation_sc: np.ndarray
    symmorphic: bool
    symprec: float

    @classmethod
    def from_GPAW(cls, atoms, log=None, *, tolerance, symmorphic):
        gpaw_symmetries = create_symmetries_object(
            atoms, tolerance=tolerance, symmorphic=symmorphic
        )
        log(gpaw_symmetries)
        sym = AtomsSymmetries(
            gpaw_symmetries.rotation_scc,
            gpaw_symmetries.atommap_sa,
            gpaw_symmetries.translation_sc,
            symmorphic,
            tolerance,
        )
        return sym


@dataclass
class SymmeryAdaptedCellCoordinates:
    """Class for defining symmetry adapted cell coordinates

    Note: This is not symmetry adapted cell, it just provides the set of
    generalized coordinates for the symmetry adapted cell.
    To get the cell, call C_cv = get_cell(cell_z).

    sacc = SymmeryAdaptedCellCoordinates(...)
    sacc.get_cell(cell_z), where cell_z is 1D array of the cell coordinates.

    Thus, the M_cc and C_cv here is just the origin of the coordinate system.
    """

    # Symmetrized cell
    M_cc: np.ndarray  # Rename to M0_cc
    C_cv: np.ndarray  # Rename to C0_cv
    dM_zcc: np.ndarray
    dM_zvv: np.ndarray
    rot_vv: np.ndarray

    def get_cell(self, cell_z):
        """
        Construct the real-space unit cell from symmetry-adapted coordinates.

        Given generalized cell coordinates `cell_z`, this method reconstructs
        the metric tensor (see get_M_cc) and then computes a corresponding
        cell matrix C_cv via a Cholesky factorization of M_cc,
        followed by a fixed rotation `rot_vv`:

            C_cv = chol(M_cc) @ rot_vv.T

        Parameters
        ----------
        cell_z : ndarray, shape (nz,)
            Symmetry-adapted cell coordinates.

        Returns
        -------
        cell : ase.geometry.Cell
            The reconstructed unit cell.
        """

        M_cc = self.get_M_cc(cell_z)
        try:
            C_cv = np.linalg.cholesky(M_cc) @ self.rot_vv.T
        except np.linalg.LinAlgError:
            print("Failed to create cell from metric", M_cc)
            raise
        return Atoms(cell=C_cv).cell

    def get_M_cc(self, cell_z):
        """
        Reconstruct the metric tensor from symmetry-adapted coordinates.

        Computes the metric tensor as a linear expansion around a reference
        metric M0_cc in the symmetry-allowed tangent directions:

            M_cc = M0_cc + sum_z cell_z[z] * dM_zcc[z]

        Parameters
        ----------
        cell_z : ndarray, shape (nz,)
            Symmetry-adapted cell coordinates.

        Returns
        -------
        M_cc : ndarray, shape (3, 3)
            Symmetrized metric tensor corresponding to `cell_z`.
        """
        return self.M_cc + np.einsum("z,zcd->cd", cell_z, self.dM_zcc)

    @classmethod
    def build(cls, cell, pbc_c, rotation_scc: np.ndarray, *, log):
        return cls(*cls.unit_cell_symmetry(cell, rotation_scc, pbc_c, log=log))

    @classmethod
    def symmetrize_cell(cls, C_cv, rotation_scc):
        """Symmetrize the cell

        Calculates the cell metric, and applies the rotation operations to it.
        New cell lower diagonal cell is calculated via Cholesky decomposition.
        By doing polar decomposition to the deformation gradient, the rotation
        back to the original cell rotation is obtained.

        Returns osymC_cV, symC_cv, M_cc, rot_vv

        Where osymC_cv is the symmetrized original like cell
        symC_cv is the lower diagonal symmetrized cell
        M_cc is the symmetrized cell metric
        rot_vv is the rotation matrix between osymC_Cv and symC_cv
            such that osymC_cv = symC_cv @ rot_vv.T

        """
        # Calculate the cell metric
        M_cc = C_cv @ C_cv.T
        # Symmetrize the cell metric
        M_cc = np.einsum(
            "scd,de,sfe->cf", rotation_scc, M_cc, rotation_scc, optimize=True
        ) / len(rotation_scc)

        symC_cv = np.linalg.cholesky(M_cc)

        # Deformation gradient
        F_vv = np.linalg.inv(C_cv) @ symC_cv

        # Sanity check
        assert np.allclose(C_cv @ F_vv, symC_cv)

        # Do a polar decomposition to rotate the symmetrized cell back
        import scipy

        rot_vv, P_vv = scipy.linalg.polar(F_vv)
        osymC_cv = symC_cv @ rot_vv.T

        return osymC_cv, symC_cv, M_cc, rot_vv

    @classmethod
    def unit_cell_symmetry(cls, C_cv, rotation_scc,
                           pbc_c, units="Å^2", log=None):
        pretty(C_cv @ C_cv.T,
               "Cell metric (M_cc' = C_cv C_c'v)",
               units,
               log=log)
        osymC_cv, symC_cv, M_cc, rot_vv = cls.symmetrize_cell(C_cv,
                                                              rotation_scc)
        pretty(M_cc,
               "Symmetrized cell metric (M_cc' = C_cv C_c'v)",
               units,
               log=log)

        # Now we can construct exact Cartesian rotation matrices
        iosymC_cv = np.linalg.inv(osymC_cv)
        U_svv = [osymC_cv.T @ U_cc.T @ iosymC_cv.T for U_cc in rotation_scc]
        U_svv = np.array(U_svv)

        # Build unit vector in symmetric matrix space
        def e(i, j):
            eps_ij = np.zeros((3, 3))
            eps_ij[i, j] = 1.0
            return eps_ij

        A_blocks = []
        for U_vv in U_svv:
            rows = []
            for i in range(3):
                for j in range(3):
                    row = U_vv @ e(i, j) @ U_vv.T - e(i, j)
                    rows.append(row.reshape((9,)))
            A_blocks.append(np.vstack(rows))
        for c in range(3):
            if not pbc_c[c]:
                A_blocks.append(e(c, c).reshape((9,)))
        A = np.vstack(A_blocks)
        # Compute null space via SVD
        U, S, Vh = np.linalg.svd(A)
        tol = 1e-6
        null_mask = S < tol
        nullspace = Vh[null_mask]
        dM_zcc = []
        dM_zvv = []
        for B in nullspace:
            dM_vv = B.reshape((3, 3))
            dof = osymC_cv @ dM_vv @ osymC_cv.T
            dM_zcc.append(dof)
            dM_zvv.append(rot_vv @ dM_vv @ rot_vv.T)
        dM_zcc = np.array(dM_zcc).reshape((-1, 3, 3))
        dM_zvv = np.array(dM_zvv).reshape((-1, 3, 3))

        # Do a QR decomposition, try to get more zeros to coordinates
        basis = np.array(dM_zcc).reshape((-1, 9))
        Q, R = np.linalg.qr(basis)
        dM_zcc = (Q.T @ basis).reshape((-1, 3, 3))

        # Normalize tangent space vectors
        Cinv = np.linalg.inv(C_cv)
        for z in range(len(dM_zcc)):
            dC = chol_derivative(M_cc, dM_zcc[z]) @ rot_vv.T
            eps = 0.5 * (Cinv @ dC + dC.T @ Cinv.T)
            dM_zcc[z] /= np.sum(np.abs(eps)) * np.linalg.det(C_cv)
            dM_zcc[z] *= 40
            # Define the direction of the tangent vector such that
            # it increases the volume. Sign cannot be used because of shear
            if np.trace(np.linalg.inv(C_cv) @ dC) < 0:
                dM_zcc[z] *= -1

        pretty_dofs(dM_zcc, M_cc, rot_vv, osymC_cv, log=log)

        # TODO: Move U_svv
        return M_cc, osymC_cv, dM_zcc, dM_zvv, rot_vv


@dataclass
class SymmetryAdaptedScaledCoordinates:
    dof_zac: np.ndarray
    s0_ac: np.ndarray
    precondition_z: np.ndarray

    def get_scaled_coordinates(self, atoms_z: np.ndarray):
        return self.s0_ac + np.einsum("zac,z->ac", self.dof_zac, atoms_z)

    @classmethod
    def build(cls, s_ac, rotation_scc, atommap_sa, symprec, C_cv, *, log):
        ns, na = atommap_sa.shape
        B_ascac = np.zeros((na, ns, 3, na, 3), int)
        for s, U_cc in enumerate(rotation_scc):
            for a in range(na):
                a2 = atommap_sa[s, a]
                B_ascac[a, s, :, a] = U_cc.T
                B_ascac[a, s, :, a2] -= np.eye(3, dtype=int)
        B_EA = B_ascac.reshape((na * ns * 3, na * 3))
        # Extra translational gauge degrees of freedom
        B_A = np.zeros((na * 3, 3))
        for a in range(na):
            B_A[(a * 3):(a * 3 + 3), :] = np.eye(3)
        B_EA = np.vstack([B_EA, B_A.T])
        import sympy as sp
        nullspace = np.array(sp.Matrix(np.array(B_EA, dtype=int)).nullspace(), dtype=float)[:, :, 0]

        # Make sure the old svd code reproduces the same result      
        U, S, Vh = np.linalg.svd(B_EA, False)
        tol = 1e-6
        null_mask = S < tol
        nullspace2 = Vh[null_mask]
       
        def same_rowspace(N, M, tol=1e-10):
            A = np.vstack([N, M])
            rA = np.linalg.matrix_rank(A, tol)
            rN = np.linalg.matrix_rank(N, tol)
            rM = np.linalg.matrix_rank(M, tol)
            return rA == rN == rM
        assert same_rowspace(nullspace, nullspace2)

        # Just make the printing prettyer for now
        nullspace = np.where(np.abs(nullspace) < 1e-10, 0, nullspace)

        if len(nullspace) == 0:
            log("No atomic degrees of freedom")
            return SymmetryAdaptedScaledCoordinates(
                np.empty((0, na, 3)), s_ac, np.empty((0,))
            )

        dof_zac = nullspace.reshape((-1, na, 3))

        if len(dof_zac):
            dof_zav = np.einsum("zac,cv->zav", dof_zac, C_cv)
            # Normalize such that the distance in Cartesian real space
            # is reflected on the generalized coordinate
            dof_zac /= np.max(np.linalg.norm(dof_zav, axis=2), axis=1)[:, None, None]
            #minimize_l1(dof_zac)

        precondition_z = np.max(np.max(np.abs(dof_zac), axis=2), axis=1)
        precondition_z /= np.sum(np.sum(np.abs(dof_zac), axis=2), axis=1)
        sasc = SymmetryAdaptedScaledCoordinates(dof_zac, s_ac, precondition_z)
        return sasc


class SymmetryAdaptedAtoms:
    """Implementation of symmetry adapted atoms

    Symmetry adapted atoms WILL symmetrize the actual_atoms given to init.

    SymmetryAdaptedAtoms does not behave like Atoms object, but will expose the
    __ase_optimizable__ protocol, so it can be optimized with ASE.
    """

    def __init__(self, actual_atoms: Atoms, symmetries: AtomsSymmetries,
                 log=print):
        self.actual_atoms = actual_atoms
        self.symmetries = symmetries
        self.symmetry_force_violation = np.inf

        pretty_subheader("Symmetry adapted cell coordinates", log)
        self.cell_coordinates = SymmeryAdaptedCellCoordinates.build(
            self.actual_atoms.cell,
            self.actual_atoms.pbc,
            self.symmetries.rotation_scc,
            log=log,
        )

        pretty_subheader("Symmetry adapted atomic coordinates", log)
        self.atom_coordinates = SymmetryAdaptedScaledCoordinates.build(
            self.actual_atoms.get_scaled_positions(),
            self.symmetries.rotation_scc,
            self.symmetries.atommap_sa,
            self.symmetries.symprec,
            self.cell_coordinates.C_cv,
            log=log,
        )
        pretty_atomic_dofs(actual_atoms, self.atom_coordinates.dof_zac, log=log)

        assert isinstance(self.atom_coordinates,
                          SymmetryAdaptedScaledCoordinates)
        # s_ac = dof_zac s_z -> ds_ac/d_sz = dof_zac
        # dR_av / dsz = dR_av / d_sac ds_ac / ds_z
        # R_av = s_ac C_cv
        #

        self.actual_atoms.set_cell(self.cell_coordinates.C_cv,
                                   scale_atoms=True)

        self.actual_atoms.wrap()
        self.S_ac = symmetrize_atoms(
            self.actual_atoms.get_scaled_positions(),
            self.symmetries.rotation_scc,
            self.symmetries.translation_sc,
            self.symmetries.atommap_sa,
        )
        self.actual_atoms.set_scaled_positions(self.S_ac)
        self.actual_atoms.wrap()
        if 1:
            log("Skipping sanity checks for now")
        else:
            pass
            # new_positions = atoms.get_positions()
            # dR_av = new_positions - old_positions
            # s_ac = np.linalg.solve(self.C_cv, dR_av.T)
            # assert (
            #     np.max(np.abs(new_positions.flatten() -
            #                   old_positions.flatten()))
            #     < symprec
            # )

        self._ndofs_cell = len(self.cell_coordinates.dM_zcc)
        self._ndofs_atoms = len(self.atom_coordinates.dof_zac)
        self._ndofs = self._ndofs_cell + self._ndofs_atoms

        self.value_z = np.zeros((self._ndofs))

    @classmethod
    def from_atoms(cls, atoms, log=print, *, symprec, symmorphic):
        symmetries = AtomsSymmetries.from_GPAW(
            atoms, tolerance=symprec, symmorphic=symmorphic, log=log,
        )
        return cls(atoms, symmetries, log=log)

    def __ase_optimizable__(self):
        return self

    # Properties for internal degrees of freedom
    @property
    def cell_z(self):
        return self.get_x()[: self._ndofs_cell]

    # From here on out, these are the __ase_optimizable__ interface
    def ndofs(self):
        return self._ndofs

    def get_x(self):
        return self.value_z.copy()
    
    def set_x(self, x):
        self.value_z[:] = x
        self.actual_atoms.set_cell(self.cell_coordinates.get_cell(self.cell_z))
        self.actual_atoms.set_scaled_positions(
            self.atom_coordinates.get_scaled_coordinates(self.atoms_z)
        )

    def get_gradient(self):
        grad_z = np.zeros(self._ndofs_cell)
        S_vv = self.actual_atoms.get_stress(voigt=False)
        C_cv = self.cell_coordinates.get_cell(self.cell_z)
        V = np.linalg.det(C_cv)
        Cinv = np.linalg.inv(C_cv)

        M_cc = self.cell_coordinates.get_M_cc(self.cell_z)

        # TODO: Move to SymmetryAdaptedCellCoordinates
        # dE/deps_vv deps_vv/dC_cv dC_cv/dz

        ncellz = len(self.cell_coordinates.dM_zcc)
        for z in range(ncellz):
            dC_cv = (
                chol_derivative(M_cc, self.cell_coordinates.dM_zcc[z])
                @ self.cell_coordinates.rot_vv.T
            )
            grad_z[z] = V * np.sum(S_vv * (Cinv @ dC_cv +
                                           dC_cv.T @ Cinv.T) / 2)

        F_av = self.actual_atoms.get_forces()
        # dE/ds_z = dE/dR_av dR_av/ds_ac ds_ac/ds_z
        # R_av = ds_ac C_cv
        # ds_ac = self.dof_zac S_z
        atoms_grad_z = -np.einsum(
            "av,cv,zac->z", F_av, C_cv, self.atom_coordinates.dof_zac
        )

        natomz = len(self.atom_coordinates.dof_zac)
        # For sanity check, we want to project the atomic gradient back
        # minimizing the Cartesian metrix.
        dof_zX = np.einsum('cv,zac->zav', C_cv, self.atom_coordinates.dof_zac).reshape((natomz, -1))
        back_Fav = -(dof_zX.T @ np.linalg.inv(dof_zX @ dof_zX.T) @ atoms_grad_z).reshape(F_av.shape)

        dF_av = F_av - back_Fav
        dF = np.max(np.linalg.norm(dF_av, axis=1))
        self.symmetry_force_violation = dF
        self.back_Fav = back_Fav
        if dF > self.fmax:
            print('Warning!!! Back projection of symmetry adapted'
                  f' forces to Cartesian space failed by {dF:7.13f}')
            print('atom Obtained force           Back projected force')
            for a, (F_v, F2_v) in enumerate(zip(F_av, back_Fav)):
                print(f'{a:5d} {F_v[0]:7.4f} {F_v[1]:7.4f} {F_v[2]:7.4f} {F2_v[0]:7.4f} {F2_v[1]:7.4f} {F2_v[2]:7.4f}')

        gradient = np.hstack([grad_z, atoms_grad_z])
        return gradient * self.get_preconditioner()

    def get_preconditioner(self):
        return 1
        return np.hstack(
            [np.ones((self._ndofs_cell,)) * 10, self.atom_coordinates.precondition_z]
        )

    def gradient_norm(self, grad_z):
        # Go actually to cell metric
        return np.max(
            np.abs(grad_z)
        )

    def get_value(self):
        return self.actual_atoms.get_potential_energy()

    def iterimages(self):
        return [self.actual_atoms]

    def converged(self, gradient, fmax):
        # Convergence needs to be from the back projected forces.
        # The symmetry violating forces will never converge.
        Fconv = np.max(np.linalg.norm(self.back_Fav, axis=1)) 
        Sconv = np.max(np.max(np.linalg.norm(self.actual_atoms.get_stress())))
        return Fconv < self.fmax and Sconv < self.smax


    @property
    def atoms_z(self):
        return self.get_x()[self._ndofs_cell:]


class Relax:
    """General utility class to log and perform symmetry adapted relax"""

    def __init__(
        self, symmorphic=False, logfile=None, teelog=True, *, atoms: Atoms, calc: GPAW, optimizer_factory,
        symprec, comm
    ):
        self.comm = comm
        self.logfile = logfile
        self.logf = None
        if self.logfile:
            if self.comm.rank == 0:
                self.logf = open(self.logfile, 'w')
        self.teelog = teelog

        if atoms.calc is not None:
            raise ValueError("Do not attach a calculator to Atoms yet.")

        self.symprec = symprec

        pretty_header("Symmetry adapted Cell and Atomic Relaxation", self.log)
        pretty_subheader("Original atoms", self.log)
        self.original_atoms = atoms.copy()
        pprint_atoms(self.original_atoms, self.log)

        self.atoms = atoms
        self.symmetry_adapted_atoms = SymmetryAdaptedAtoms.from_atoms(
            self.atoms, log=self.log, symmorphic=False, symprec=symprec
        )
        
        saa2 = SymmetryAdaptedAtoms.from_atoms(
            self.symmetry_adapted_atoms.actual_atoms, log=self.log, symmorphic=False, symprec=1e-5
        )

        # Now, with cell and atoms symmetrized,
        # it is safe to assign the calculator
        # TODO: Implement Setter or something
        self.calc = calc
        self.symmetry_adapted_atoms.actual_atoms.calc = calc()

        pretty_subheader("Symmetrized atoms", self.log)
        pprint_atoms(self.symmetry_adapted_atoms.actual_atoms, self.log)

        # if not isinstance(calc, ASECalculator):
        #    raise ValueError("Calculator must be new GPAW.")

        self.optimizer_factory = optimizer_factory
        self.optimizer = self.optimizer_factory(self.symmetry_adapted_atoms)
        # atoms.wrap()

    def log(self, *args, **kwargs):
        if self.comm.rank == 0:
            if self.logf:
                print(*args, **kwargs, flush=True, file=self.logf)
            if self.teelog:
                print(*args, **kwargs)

    def run(self, *, fmax, smax, maxiter):
        # Why would symmetry adapted atoms care about smax and fmax
        # But it needs to be ase optimizable, so it needs to do that
        self.symmetry_adapted_atoms.smax = smax
        self.symmetry_adapted_atoms.fmax = fmax
        
        self.smax = smax
        self.fmax = fmax
        self.maxiter = maxiter
        i = 0
        dtitles = '    '.join([f'q{i:02d}' for i in range(len(self.symmetry_adapted_atoms.atoms_z))])
        self.log(f'iter  time     E            maxF   maxS    maxG    a1    a2    a3     L1      L2      L3     {dtitles}   log sym. violation')
        for _ in self.optimizer.irun(fmax=fmax):
            import time
            T = time.localtime()
            tstr = '%02d:%02d:%02d' % (T[3], T[4], T[5])
            E = self.symmetry_adapted_atoms.actual_atoms.get_potential_energy()
            F = self.symmetry_adapted_atoms.back_Fav #get_forces()
            S = self.symmetry_adapted_atoms.actual_atoms.get_stress()
            g = self.symmetry_adapted_atoms.get_gradient()
            Fmax = np.max(np.linalg.norm(F, axis=1))
            sFmax = f'{Fmax:7.3f}'
            if Fmax < self.fmax:
                sFmax = GREEN + sFmax + RESET
            
            Smax = np.max(np.abs(S))
            sSmax = f'{Smax:7.4f}'
            if Smax < self.smax:
                sSmax = GREEN + sSmax + RESET

            gmax = np.max(np.abs(g))

            cell = self.symmetry_adapted_atoms.actual_atoms.cell
            a = cell.angles()
            l = cell.lengths()
            cell = f'{a[0]:5.1f} {a[1]:5.1f} {a[2]:5.1f} '
            cell += f'{l[0]:7.3f} {l[1]:7.3f} {l[2]:7.3f}'

            dofs = ''
            for Z in self.symmetry_adapted_atoms.atoms_z:
                dofs += f' {Z:6.3f}'
            syviol = np.log10(self.symmetry_adapted_atoms.symmetry_force_violation)
            symviol = f'{syviol:4.1f}'
            if syviol < self.fmax:
                symviol = GREEN + symviol + RESET
            self.log(f'{i:5d} {tstr} {E:9.5f} {sFmax} {sSmax} {gmax:7.3f} {cell}{dofs} {symviol}')
            i += 1
            if i > maxiter:
                self.log(f'Not converged in {maxiter}.')
                return False

        return True

    def __ase_optimizable__(self):
        return self

    def visualize_modes(self):
        from ase.io.trajectory import Trajectory

        traj = Trajectory("modes.traj", "w")
        for z in range(self.ndofs()):
            x = np.zeros((self.ndofs(),))
            for i in np.arange(0, 6 * np.pi, 0.1):
                x[z] = np.sin(i) * 0.004
                self.set_x(x)
                traj.write(self.atoms.copy())

    def calc_hessian(self):
        from gpaw.new.symmetry import SymmetryBrokenError
        saa = self.symmetry_adapted_atoms
        x = np.zeros((saa._ndofs))
        H = np.zeros((saa._ndofs, saa._ndofs))
        def grad():
            try:
                G = saa.get_gradient()
            except SymmetryBrokenError:
                saa.actual_atoms.calc = self.calc()
                G = saa.get_gradient()
            return G
        for i in range(saa._ndofs):
            x[:] = 0.0
            x[i] = 1e-3
            saa.set_x(x)
            G = grad()
            x[i] = -1e-3
            saa.set_x(x)
            G0 = grad()
            H[i] = (G - G0) / (2e-3)
            print(i)
        pretty(H, "Hessian", log=self.log)
        eps, vec = np.linalg.eigh(H)
        self.optimizer.H0 = H

        # Reset to full symmetry
        x[:] = 0.0
        saa.set_x(x)
        saa.actual_atoms.calc = self.calc()

# Tests:
# Wurtzite, distorted structure, nice logging, quick convergence
if __name__ == "__main__":
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.2)
    atoms = bulk("ZnO", crystalstructure="wurtzite", a=3.24, c=5.20)
    atoms = bulk("ZnO", crystalstructure="wurtzite", a=3.14, c=5.30)
    # Avoid rotating the cell (making it symmetric)
    #eps = np.array([[0,1,0], [1,0,0], [0,0, 0]]) * 0.02
    #atoms.set_cell(atoms.get_cell() @ (np.eye(3) + eps + eps.T))
    #atoms.rattle(0.1)
    from ase.io import read
    #atoms = read('2AlCl3-1.xyz').copy()
    atoms = read(argv[1]).copy()

    def calc():
        return GPAW(
            mode={"name": "pw", "ecut": 400},
            kpts={"density": 1, "gamma": True},
            symmetry={"symmorphic": False},
            txt='ZnO.txt', 
            xc="PBE",
            convergence={"density": 1e-7},
        )
    from ase.optimize.bfgs import BFGS
    from gpaw.mpi import world
    relax = Relax(
        atoms=atoms,
        calc=calc,
        optimizer_factory=lambda atoms: BFGS(atoms,
                                             maxstep=0.1,logfile='bfgs.log',
                                             trajectory="a.traj"),
        symprec=0.1,
        logfile='relax.log',
        teelog=True,
        comm=world,
    )
    #relax.calc_hessian()
    relax.run(fmax=0.005, smax=0.0002, maxiter=40)
