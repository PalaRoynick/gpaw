from gpaw.new.ase_interface import GPAW
from ase import Atoms
from gpaw.new.symmetry import Symmetries, create_symmetries_object
import numpy as np
from gpaw.new.ase_interface import ASECalculator
from dataclasses import dataclass
from gpaw.new.relax_print import pretty, pprint_atoms, pretty_dofs

def chol_derivative(A, dA):
    eps = 1e-8
    L = np.linalg.cholesky(A)
    Lp = np.linalg.cholesky(A + eps * dA)
    return (Lp - L) / eps


def symmetrize_atoms(S_ac, U_scc, f_sc, atommap_sa, tol=1e-12):
    ns, na = atommap_sa.shape
    Ssym_ac = np.zeros_like(S_ac, dtype=np.complex128)
    for a in range(na):
        for s in range(ns):
            new = U_scc[s].T @ S_ac[a] - f_sc[s]
            Ssym_ac[atommap_sa[s, a]] += np.exp(2j*np.pi*new)
    Ssym_ac = (np.angle(Ssym_ac) / (2 * np.pi)) % 1.0 % 1.0
    Ssym_ac[np.abs(Ssym_ac) < tol] = 0.0
    Ssym_ac[np.abs(Ssym_ac - 1.0) < tol] = 0.0
    return Ssym_ac

@dataclass
class AtomsSymmetries:
    rotation_scc: np.ndarray
    atommap_sa: np.ndarray
    translation_sc: np.ndarray
    symmorphic: bool
    symprec: float

    @classmethod
    def from_GPAW(cls, atoms, log=print, *, tolerance, symmorphic):
        gpaw_symmetries = create_symmetries_object(atoms, tolerance=tolerance, symmorphic=symmorphic)
        log(gpaw_symmetries)
        sym = AtomsSymmetries(gpaw_symmetries.rotation_scc,
                              gpaw_symmetries.atommap_sa,
                              gpaw_symmetries.translation_sc,
                              symmorphic,
                              tolerance)
        return sym

@dataclass
class SymmeryAdaptedCellCoordinates:
    """Class for defining symmetry adapted cell coordinates

    Note: This is not symmetry adapted cell, it just provides the set of generalized coordinates
    for the symmetry adapted cell. To get the cell, call

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
        M_cc = self.get_M_cc(cell_z)
        try:
            C_cv = np.linalg.cholesky(M_cc) @ self.rot_vv.T
        except np.linalg.LinAlgError:
            print('Failed to create cell from metric', M_cc)
            raise
        return Atoms(cell=C_cv).cell

    def get_M_cc(self, cell_z):
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
        M_cc = np.einsum("scd,de,sfe->cf",
                         rotation_scc,
                         M_cc,
                         rotation_scc,
                         optimize=True) / len(rotation_scc)

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
    def unit_cell_symmetry(cls, C_cv, rotation_scc, pbc_c, units='Å^2', log=None):
        pretty(C_cv @ C_cv.T, "Cell metric (M_cc' = C_cv C_c'v)", units, log=log)
        osymC_cv, symC_cv, M_cc, rot_vv = cls.symmetrize_cell(C_cv, rotation_scc)
        pretty(M_cc, "Symmetrized cell metric (M_cc' = C_cv C_c'v)", units, log=log)

        # Now we can construct exact Cartesian rotation matrices
        iosymC_cv = np.linalg.inv(osymC_cv)
        U_svv = np.array([osymC_cv.T @ U_cc.T @ iosymC_cv.T for U_cc in rotation_scc])

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
                    rows.append((U_vv @ e(i, j) @ U_vv.T - e(i, j)).reshape((9,)))
            A_blocks.append(np.vstack(rows))
        for c in range(3):
            if not pbc_c[c]:
                A_blocks.append(e(c,c).reshape((9,)))
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
        basis = np.array(dM_zcc).reshape((-1, 9))
        Q, R = np.linalg.qr(basis)
        dM_zcc = (Q.T @ basis).reshape((-1, 3, 3))

        symC_cv = np.linalg.cholesky(M_cc)

        if 1:
            Cinv = np.linalg.inv(C_cv)
            for z in range(len(dM_zcc)):
                dC = chol_derivative(M_cc, dM_zcc[z]) @ rot_vv.T
                eps = 0.5 * (Cinv @ dC + dC.T @ Cinv.T)
                dM_zcc[z] /= np.sum(np.abs(eps)) * np.linalg.det(C_cv)

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

    def get_scaled_coordinates(self, atoms_z:np.ndarray):
        return self.s0_ac + np.einsum('zac,z->ac', self.dof_zac, atoms_z)

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
            B_A[(a*3):(a*3+3), :] = np.eye(3)
        B_EA = np.vstack([B_EA, B_A.T])

        U, S, Vh = np.linalg.svd(B_EA, False)
        tol = 1e-6
        null_mask = S < tol
        nullspace = Vh[null_mask]

        # Maybe DOFs will be more human understandable after this rotation?
        Q, R = np.linalg.qr(nullspace)
        nullspace = Q.T @ nullspace

        # Just make the printing prettyer for now
        nullspace = np.where(np.abs(nullspace)<1e-10, 0, nullspace)
        
        if len(nullspace) == 0:
            log('No atomic degrees of freedom')
            return SymmetryAdaptedScaledCoordinates(
                    np.empty((0, na, 3)), s_ac, np.empty((0,)))

        dof_zac = nullspace.reshape((-1, na, 3))
        
        if len(dof_zac):
            dof_zav = np.einsum('zac,cv->zav', dof_zac, C_cv)
            # Normalize such that the distance in Cartesian real space is reflected on the generalized coordinate
            dof_zac /= np.max(np.abs(dof_zav)) ##np.sum(np.sum(dof_zav**2, axis=2), axis=1) ** 0.5
        
        log('Atomic degrees of freedom')
        for z, dof_ac in enumerate(dof_zac):
            log(f'DOF {z}')
            for dof_c in dof_ac:
                log(dof_c, end=' ')
            log()

        precondition_z = np.max(np.max(np.abs(dof_zac), axis=2), axis=1) / np.sum(np.sum(np.abs(dof_zac), axis=2), axis=1)
        sasc = SymmetryAdaptedScaledCoordinates(dof_zac, s_ac, precondition_z)
        return sasc


class SymmetryAdaptedAtoms:
    """Implementation of symmetry adapted atoms

    Symmetry adapted atoms WILL symmetrize the actual_atoms given to init.

    SymmetryAdaptedAtoms does not behave like Atoms object, but will expose the
    __ase_optimizable__ protocol, so it can be optimized with ASE.
    """
    def __init__(self,
                 actual_atoms: Atoms,
                 symmetries: AtomsSymmetries,
                 log=print):
        self.actual_atoms = actual_atoms
        self.symmetries = symmetries

        log('Building symmetry adapted cell coordinates')
        self.cell_coordinates = SymmeryAdaptedCellCoordinates.build(self.actual_atoms.cell, self.actual_atoms.pbc, self.symmetries.rotation_scc, log=log)
        
        log('Building symmetry adapted atomic coordinates')
        self.atom_coordinates = SymmetryAdaptedScaledCoordinates.build(self.actual_atoms.get_scaled_positions(), self.symmetries.rotation_scc, self.symmetries.atommap_sa, self.symmetries.symprec, self.cell_coordinates.C_cv, log=log)
        assert isinstance(self.atom_coordinates, SymmetryAdaptedScaledCoordinates)
        # s_ac = dof_zac s_z -> ds_ac/d_sz = dof_zac
        # dR_av / dsz = dR_av / d_sac ds_ac / ds_z
        # R_av = s_ac C_cv 
        # 

        self.actual_atoms.set_cell(self.cell_coordinates.C_cv, scale_atoms=True)
        self.actual_atoms.wrap()
        self.S_ac = symmetrize_atoms(self.actual_atoms.get_scaled_positions(), self.symmetries.rotation_scc, self.symmetries.translation_sc, self.symmetries.atommap_sa)
        self.actual_atoms.set_scaled_positions(self.S_ac)
        self.actual_atoms.wrap()
        if 1:
            log('Skipping sanity checks for now')
        else:
            new_positions = atoms.get_positions()
            dR_av = new_positions - old_positions
            s_ac = np.linalg.solve(self.C_cv, dR_av.T)
            assert np.max(np.abs(new_positions.flatten() - old_positions.flatten())) < symprec 

        log('Symmetrized atoms')
        pprint_atoms(self.actual_atoms, log)

        self._ndofs_cell = len(self.cell_coordinates.dM_zcc)
        self._ndofs_atoms = len(self.atom_coordinates.dof_zac)
        self._ndofs = self._ndofs_cell + self._ndofs_atoms

        self.value_z = np.zeros((self._ndofs))

    @classmethod
    def from_atoms(cls, atoms, log=print, *, symprec, symmorphic):
        symmetries = AtomsSymmetries.from_GPAW(atoms, tolerance=symprec, symmorphic=symmorphic)
        return cls(atoms, symmetries, log=log)

    def __ase_optimizable__(self):
        return self
    
    # Properties for internal degrees of freedom
    @property
    def cell_z(self):
        return self.get_x()[:self._ndofs_cell]
   
    # From here on out, these are the __ase_optimizable__ interface
    def ndofs(self):
        return self._ndofs
    
    def get_x(self):
        return self.value_z.copy()

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
        
        grad_z = np.zeros(self._ndofs_cell)
        S_vv = self.actual_atoms.get_stress(voigt=False)
        C_cv = self.cell_coordinates.get_cell(self.cell_z)
        V = np.linalg.det(C_cv)
        Cinv = np.linalg.inv(C_cv)
        
        M_cc = self.cell_coordinates.get_M_cc(self.cell_z)

        # TODO: Move to SymmetryAdaptedCellCoordinates
        # dE/deps_vv deps_vv/dC_cv dC_cv/dz
        for z in range(len(self.cell_coordinates.dM_zcc)):
            dC_cv = chol_derivative(M_cc, self.cell_coordinates.dM_zcc[z]) @ self.cell_coordinates.rot_vv.T
            grad_z[z] = V * np.sum(S_vv * (Cinv @ dC_cv + dC_cv.T @ Cinv.T)/2)

        F_av = self.actual_atoms.get_forces()
        # dE/ds_z = dE/dR_av dR_av/ds_ac ds_ac/ds_z
        # R_av = ds_ac C_cv
        # ds_ac = self.dof_zac S_z
        atoms_grad_z = -np.einsum('av,cv,zac->z', F_av, C_cv, self.atom_coordinates.dof_zac) 

        gradient = np.hstack([grad_z, atoms_grad_z])
        return gradient * self.get_preconditioner()

    def get_preconditioner(self):
        return np.hstack([np.ones_like(self.cell_z), self.atom_coordinates.precondition_z])
    
    def gradient_norm(self, grad_z):
        # Go actually to cell metric
        return np.max(np.abs(grad_z)) # (np.max(self.atoms.get_forces()), np.max(self.atoms.get_stress()), *self.atoms.cell.lengths())

    def get_value(self):
        return self.actual_atoms.get_potential_energy()

    def iterimages(self):
        return [self.actual_atoms]

    def converged(self, gradient, fmax):
        return self.gradient_norm(gradient) < fmax
        #return F < fmax and S < self.smax

    def set_x(self, x):
        self.value_z[:] = x
        self.actual_atoms.set_cell(self.cell_coordinates.get_cell(self.cell_z))
        self.actual_atoms.set_scaled_positions(self.atom_coordinates.get_scaled_coordinates(self.atoms_z))
    
    @property
    def atoms_z(self):
        return self.get_x()[self._ndofs_cell:]


class Relax:
    """General utility class to log and perform symmetry adapted optimizations
    """

    def __init__(self, symmorphic=False, *, atoms: Atoms, calc: GPAW, optimizer_factory, symprec):
        if atoms.calc is not None:
            raise ValueError("Do not attach a calculator to Atoms yet.")

        self.symprec = symprec

        self.log('Symmetry adapted relaxation')
        self.log('Original atoms')
        self.original_atoms = atoms.copy()
        pprint_atoms(self.original_atoms, self.log)

        self.atoms = atoms
        self.symmetry_adapted_atoms = SymmetryAdaptedAtoms.from_atoms(self.atoms, log=self.log, symmorphic=False, symprec=symprec)

        # Now, with cell and atoms symmetrized, it is safe to assign the calculator
        # TODO: Implement Setter or something
        self.symmetry_adapted_atoms.actual_atoms.calc = calc

        self.log('Symmetrized atoms')
        pprint_atoms(self.symmetry_adapted_atoms.actual_atoms, self.log)

        # if not isinstance(calc, ASECalculator):
        #    raise ValueError("Calculator must be new GPAW.")

        self.optimizer_factory = optimizer_factory
        self.optimizer = self.optimizer_factory(self.symmetry_adapted_atoms)
        #atoms.wrap()

        

    def log(self, *args, **kwargs):
        print(*args, **kwargs)

    def run(self, *, fmax, smax):
        self.smax = smax
        return self.optimizer.run(fmax=fmax)
        
        for _ in self.optimizer.irun(fmax=fmax):
            pass
            #print('INTERNAL', self.get_x()) 
            #print('DIST', self.atoms.cell.lengths(), self.atoms.cell.angles())

    def __ase_optimizable__(self):
        return self

    def visualize_modes(self):
        from ase.io.trajectory import Trajectory
        traj = Trajectory('modes.traj', 'w')
        for z in range(self.ndofs()):
            x = np.zeros((self.ndofs(),))
            for i in np.arange(0, 6 * np.pi, 0.1):
                x[z] = np.sin(i) * 0.004 
                self.set_x(x)
                traj.write(self.atoms.copy())

    def calc_hessian(self):
        saa = self.symmetry_adapted_atoms
        x = np.zeros((saa._ndofs))
        H = np.zeros((saa._ndofs, saa._ndofs))
        for i in range(saa._ndofs):
            x[:] = 0.0
            x[i] = 1e-3
            saa.set_x(x)
            G = saa.get_gradient()
            x[i] = -1e-3
            saa.set_x(x)
            G0 = saa.get_gradient()
            H[i] = (G - G0) / (2e-3)
        pretty(H, 'Hessian', log=self.log)
        eps, vec = np.linalg.eigh(H)
        self.optimizer.H0 = H

# Tests:
# Wurtzite, distorted structure, nice logging, quick convergence
if __name__ == "__main__":
    from ase.build import bulk
    atoms = bulk('NaCl', 'rocksalt', a=5.2)
    atoms = bulk('NiAg', crystalstructure='wurtzite', a=3.24*1.2, c=5.20*1.2)
    # Avoid rotating the cell (making it symmetric)
    eps = np.random.rand(3,3)*0.001
    atoms.set_cell(atoms.get_cell() @ (np.eye(3) + eps + eps.T))

    atoms.rattle(0.001)
    calc = GPAW(mode={'name': "pw", 'ecut': 700}, kpts={'size': (2,2,2),
                'gamma': True}, txt='NaCl.txt',
                symmetry={'symmorphic': False},
                xc='LDA',
                convergence={'density':1e-7})
    from ase.optimize.bfgs import BFGS
    from ase.calculators.emt import EMT
    calc = EMT()
    relax = Relax(atoms=atoms, calc=calc, optimizer_factory=lambda atoms: BFGS(atoms, maxstep=1, trajectory='a.traj'), symprec=0.1)
    relax.calc_hessian()
    relax.run(fmax=0.05, smax=0.003)

if 0: #__name__ == "__main__":
    from ase.build import bulk

    #from ase.io.jsonio import read_json
    #atoms = read_json('output.json')

    atoms = Atoms('AuAg',
            cell=[4, 4, 4],
            positions=[[0, 0, 0],
                       [2, 2, 2]],
            pbc=True)
    atoms.positions[1, 1] += 0.01 
    #angle = 62
    #c = np.cos(angle / 180 * np.pi)
    #a = atoms.cell.lengths()[0]
    #M_cc = a**2 * np.array([[1, c, c], [c, 1, c], [c, c, 1]])
    #cell_cv = np.linalg.cholesky(M_cc)
    #atoms.set_cell(cell_cv, scale_atoms=True)

    #eps_cc = np.random.rand(3, 3) * 0.0001
    #atoms.set_cell(atoms.cell @ (np.eye(3) + eps_cc), scale_atoms=True)

    if 1:
        atoms = bulk('NaCl', crystalstructure='rocksalt', a=5.2)
        #from ase.io import read
        #atoms = read('/home/kuisma/Downloads/2AlCl3-1.xyz').copy()
        print(atoms.cell.angles(), atoms.cell.lengths())
        calc = GPAW(mode={'name': "pw", 'ecut': 700}, kpts={'size': (2,2,1),
                    'gamma': True}, txt='asd.txt',
                    eigensolver='dav',
                    symmetry={'symmorphic': False},
                    xc='LDA',
                    convergence={'density':1e-7})

    from ase.optimize.bfgs import BFGS
    from ase.optimize.mdmin import MDMin
    from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG 
    relax = Relax(atoms=atoms, calc=calc, optimizer_factory=lambda atoms: BFGS(atoms, trajectory='a.traj'), symprec=0.1)
    #relax.calc_hessian()
    relax.visualize_modes()
    if 0:
        for z in range(3):
            grad = relax.get_gradient()
            vec = np.zeros((3,))
            vec[z] = -1e-3
            relax.set_x(vec)
            E0 = relax.get_value()
            vec[z] = 1e-3
            relax.set_x(vec)
            E1 = relax.get_value()
            print("Finite difference grad", (E1 - E0) / (2e-3))
            print("Gotten grad", grad[z])
            print("div", grad[z] / ((E1 - E0) / 2e-3))

    relax.run(fmax=0.001, smax=0.001)
