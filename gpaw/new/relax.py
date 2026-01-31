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
    eps = 1e-8
    L = np.linalg.cholesky(A)
    Lp = np.linalg.cholesky(A + eps * dA)
    return (Lp - L) / eps


def unit_cell_symmetry(C_cv, U_scc, pbc_c):
    print('Original cell', Atoms(cell=C_cv).cell.lengths(), Atoms(cell=C_cv).cell.angles())
    print('Original cell', Atoms(cell=C_cv).cell)
    print("Symmetries", len(U_scc))

    # Calculate the cell metric
    M_cc = C_cv @ C_cv.T
    # Symmetrize the cell metric
    M_cc = np.einsum("scd,de,sfe->cf", U_scc, M_cc, U_scc, optimize=True) / len(U_scc)

    symC_cv = np.linalg.cholesky(M_cc)


    # Deformation gradient
    F_vv = np.linalg.inv(C_cv) @ symC_cv

    # Sanity check
    assert np.allclose(C_cv @ F_vv, symC_cv)

    # Do a polar decomposition to rotate the symmetrized cell back
    import scipy
    rot_vv, P_vv = scipy.linalg.polar(F_vv)
    osymC_cv = symC_cv @ rot_vv.T
    print('Symmetrized cell', Atoms(cell=osymC_cv).cell.lengths(),
                              Atoms(cell=osymC_cv).cell.angles())
    print('Symmetrized cell', osymC_cv)
    # Now we can construct exact Cartesian rotation matrices
    iosymC_cv = np.linalg.inv(osymC_cv)
    U_svv = np.array([osymC_cv.T @ U_cc.T @ iosymC_cv.T for U_cc in U_scc])

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
    # basis = np.array(dM_zcc).reshape((-1, 9))
    # Q, R = np.linalg.qr(basis)
    # dM_zcc = (Q.T @ basis).reshape((-1, 3, 3))

    symC_cv = np.linalg.cholesky(M_cc)

    if 1:
        Cinv = np.linalg.inv(C_cv)
        for z in range(len(dM_zcc)):
            dC = chol_derivative(M_cc, dM_zcc[z]) @ rot_vv.T
            eps = 0.5 * (Cinv @ dC + dC.T @ Cinv.T)
            dM_zcc[z] /= 0.01 * np.max(np.abs(eps)) * np.linalg.det(C_cv) # np.sqrt(np.sum(eps * eps))

    for z, dM_cc in enumerate(dM_zcc):
        print(f"Tangent {z} of cell")
        print("In metric space")
        pretty(dM_cc)
        print("C_cv:")
        pretty(chol_derivative(M_cc, dM_cc) @ rot_vv.T)

    return M_cc, osymC_cv, U_svv, dM_zcc, dM_zvv, rot_vv

def atom_position_symmetry(U_scc, atommap_sa, atoms, tol):
    ns = len(U_scc)
    na = len(atoms)
    S_ac = atoms.get_scaled_positions()
    B_ascac = np.zeros((na, ns, 3, na, 3), int)
    for s, U_cc in enumerate(U_scc):
        for a, S_c in enumerate(S_ac):
            a2 = atommap_sa[s, a]
            #print(S_c)
            #n_c = U_cc.T @ S_c - S_ac[a2] This is missing non-symmorphic shift
            #assert abs(n_c - n_c.round()).sum() < tol, n_c
            #N_asc[a, s] = n_c.astype(int)
            B_ascac[a, s, :, a] = U_cc.T
            B_ascac[a, s, :, a2] -= np.eye(3, dtype=int)
    B_EA = B_ascac.reshape((na * ns * 3, na * 3))
    # Extra translational gauge degrees of freedom
    if 1:
        B_A = np.zeros((na * 3, 3))
        for a in range(na):
            B_A[(a*3):(a*3+3), :] = np.eye(3)
        B_EA = np.vstack([B_EA, B_A.T])

    U, S, Vh = np.linalg.svd(B_EA, False)
    tol = 1e-6
    null_mask = S < tol
    nullspace = Vh[null_mask]

    print('Eigenvalues', S)
    # Maybe DOFs will be more human understandable after this rotation?
    Q, R = np.linalg.qr(nullspace)
    nullspace = Q.T @ nullspace

    
    # Just make the printing prettyer for now
    nullspace = np.where(np.abs(nullspace)<1e-10, 0, nullspace)
    
    if len(nullspace) == 0:
        print('No atomic degrees of freedom')
        return np.empty((0, na, 3))
    dof_zac = nullspace.reshape((-1, na, 3))
    #print(f'{dof_zac=}')
    ## Solve the underdetermined problem, i.e. project symmetric dof
    #dof_zx = dof_zac.reshape((-1, na * 3))
    #print(dof_zx.shape)
    #print(S_ac.reshape((-1,)).shape)
    # TODO: Add checks that the residuals are within the tolerances assumed
    #S_z = np.linalg.lstsq(dof_zx.T, S_ac.reshape((-1,)))[0]
    #print(S_z.shape)
    #print(dof_zac.shape)
    #print('Old scaled positions', S_ac)
    #S_ac = np.einsum('z,zac->ac', S_z, dof_zac)
    #print('New scaled positions', S_ac)
    #  
    # Set symmetrized positions to atoms
    #atoms.set_scaled_positions(S_ac)
    print('Atomic degrees of freedom')
    for z, dof_ac in enumerate(dof_zac):
        print(f'DOF {z}')
        for dof_c in dof_ac:
            print(dof_c, end=' ')
        print()

    return dof_zac #, S_z


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

class Relax:
    def __init__(self, symmorphic=False, *, atoms: Atoms, calc: GPAW, optimizer_factory, symprec):
        if atoms.calc is not None:
            raise ValueError("Do not attach a calculator to Atoms yet.")

        # if not isinstance(calc, ASECalculator):
        #    raise ValueError("Calculator must be new GPAW.")

        self.atoms = atoms
        self.calc = calc
        self.optimizer_factory = optimizer_factory
        self.symprec = symprec
        atoms.wrap()
        old_positions = atoms.get_positions().copy()
        from ase.io import write
        write('old.xyz', atoms)
        self.symmetries = create_symmetries_object(self.atoms, tolerance=self.symprec, symmorphic=symmorphic)
        print('pbc', self.atoms.pbc)
        self.M_cc, self.C_cv, U_svv, self.dM_zcc, self.dM_zvv, self.rot_vv = unit_cell_symmetry(
            self.atoms.cell, self.symmetries.rotation_scc, self.atoms.pbc
        )
        
        self.dof_zac = atom_position_symmetry(self.symmetries.rotation_scc, self.symmetries.atommap_sa, atoms, symprec)

        # s_ac = dof_zac s_z -> ds_ac/d_sz = dof_zac
        # dR_av / dsz = dR_av / d_sac ds_ac / ds_z
        # R_av = s_ac C_cv 
        # 
        if 1:
            dof_zav = np.einsum('zac,cv->zav', self.dof_zac, self.C_cv)
            # Normalize such that the distance in Cartesian real space is reflected on the generalized coordinate
            self.dof_zac /= np.max(np.abs(dof_zav)) ##np.sum(np.sum(dof_zav**2, axis=2), axis=1) ** 0.5

        self.atoms.set_cell(self.C_cv, scale_atoms=True)

        atoms.wrap()
        print('Old positions', atoms.get_scaled_positions())
        print('Old positions', atoms.get_positions())
        self.S_ac = symmetrize_atoms(atoms.get_scaled_positions(), self.symmetries.rotation_scc, self.symmetries.translation_sc, self.symmetries.atommap_sa)
        self.atoms.set_scaled_positions(self.S_ac)
        atoms.wrap()
        new_positions = atoms.get_positions()
        print('New positions', atoms.get_scaled_positions())
        print('New positions', new_positions)
        dR_av = new_positions - old_positions
        s_ac = np.linalg.solve(self.C_cv, dR_av.T)
        print('Scaled diff', s_ac)
        print('Maximum shift after atomic symmetrization', np.max(np.abs(new_positions.flatten() - old_positions.flatten())))
        #assert np.max(np.abs(new_positions.flatten() - old_positions.flatten())) < symprec 
        write('new.xyz', atoms)
        # Now, with cell and atoms symmetrized, it is safe to assign the calculator
        self.atoms.calc = calc

        self._ndofs_cell = len(self.dM_zcc)
        self._ndofs_atoms = len(self.dof_zac)
        self._ndofs = self._ndofs_cell + self._ndofs_atoms

        self.optimizer = optimizer_factory(self)

        self.value_z = np.zeros((self._ndofs))

    def iterimages(self):
        return [self.atoms]

    def run(self, *, fmax, smax):
        self.smax = smax
        return self.optimizer.run(fmax=fmax)
        
        for _ in self.optimizer.irun(fmax=fmax):
            print('INTERNAL', self.get_x()) 
            print('DIST', self.atoms.cell.lengths(), self.atoms.cell.angles())

    def __ase_optimizable__(self):
        return self

    def get_value(self):
        return self.atoms.get_potential_energy()

    @property
    def cell_z(self):
        return self.get_x()[:self._ndofs_cell]
    
    @property
    def atoms_z(self):
        return self.get_x()[self._ndofs_cell:]

    def get_x(self):
        return self.value_z.copy()

    def _get_cell(self):
        M_cc = self.M_cc + np.einsum("z,zcd->cd", self.cell_z, self.dM_zcc)
        C_cv = np.linalg.cholesky(M_cc) @ self.rot_vv.T
        return Atoms(cell=C_cv).cell

    def set_x(self, x):
        self.value_z[:] = x

        M_cc = self.M_cc + np.einsum("z,zcd->cd", self.cell_z, self.dM_zcc)
        C_cv = np.linalg.cholesky(M_cc) @ self.rot_vv.T
        self.atoms.set_cell(C_cv)

        S_ac = self.S_ac + np.einsum('z,zac->ac', self.atoms_z, self.dof_zac)
        self.atoms.set_scaled_positions(S_ac)

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
        S_vv = self.atoms.get_stress(voigt=False)
        C_cv = np.array(self._get_cell())
        V = np.linalg.det(C_cv)
        Cinv = np.linalg.inv(C_cv)
        
        M_cc = self.M_cc + np.einsum("z,zcd->cd", self.cell_z, self.dM_zcc)

        # dE/deps_vv deps_vv/dC_cv dC_cv/dz
        for z in range(len(self.dM_zcc)):
            dC_cv = chol_derivative(M_cc, self.dM_zcc[z]) @ self.rot_vv.T
            grad_z[z] = V * np.sum(S_vv * (Cinv @ dC_cv + dC_cv.T @ Cinv.T)/2)

        F_av = self.atoms.get_forces()
        # dE/ds_z = dE/dR_av dR_av/ds_ac ds_ac/ds_z
        # R_av = ds_ac C_cv
        # ds_ac = self.dof_zac S_z
        atoms_grad_z = -np.einsum('av,cv,zac->z', F_av, C_cv, self.dof_zac)

        gradient = np.hstack([grad_z, atoms_grad_z])
        return gradient

    def converged(self, gradient, fmax):
        return self.gradient_norm(gradient) < fmax
        #return F < fmax and S < self.smax

    def gradient_norm(self, grad_z):
        # Go actually to cell metric
        return np.max(np.abs(grad_z)) # (np.max(self.atoms.get_forces()), np.max(self.atoms.get_stress()), *self.atoms.cell.lengths())

    def ndofs(self):
        return self._ndofs

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
        x = np.zeros((self._ndofs))
        H = np.zeros((self._ndofs, self._ndofs))
        for i in range(self._ndofs):
            x[:] = 0.0
            x[i] = 1e-3
            self.set_x(x)
            G = self.get_gradient()
            x[i] = -1e-3
            self.set_x(x)
            G0 = self.get_gradient()
            H[i] = (G - G0) / (2e-3)
            print(i)
        print('Hessian', H)
        print(np.linalg.eigh(H))
        self.optimizer.H0 = H

if __name__ == "__main__":
    from ase.build import bulk
    from ase.calculators.emt import EMT

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
        atoms = bulk('ZnO', crystalstructure='wurtzite', a=3.24, c=5.20)
        atoms = bulk('NaCl', crystalstructure='rocksalt', a=5.2)
        from ase.io import read
        atoms = read('/home/kuisma/Downloads/2AlCl3-1.xyz').copy()
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
    relax = Relax(atoms=atoms, calc=calc, optimizer_factory=lambda atoms: BFGS(atoms, alpha=[7, 1, 56], maxstep=1, trajectory='a.traj'), symprec=0.1)
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
