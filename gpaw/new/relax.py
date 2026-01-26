from gpaw.new.ase_interface import GPAW
from ase import Atoms
from gpaw.new.symmetry import Symmetries, create_symmetries_object
import numpy as np
from gpaw.new.ase_interface import ASECalculator

def unit_cell_symmetry(C_cv, U_scc):
        print('Symmetries', len(U_scc))
        # Calculate the cell metric
        M_cc = C_cv @ C_cv.T

        # Symmetrize the cell metric
        M_cc = np.einsum('scd,de,sfe->cf',
                         U_scc,
                         M_cc,
                         U_scc,
                         optimize=True) / len(U_scc)

        print('Old cell', C_cv)
        symC_cv = np.linalg.cholesky(M_cc)
        
        print('New cell', symC_cv)

        # Deformation gradient
        F_vv = np.linalg.inv(C_cv) @ symC_cv

        # Sanity check
        print('Rotated', C_cv @ F_vv)
        print('Symmetric', symC_cv)
        assert np.allclose(C_cv @ F_vv, symC_cv)

        import scipy
        U_vv, P_vv = scipy.linalg.polar(F_vv)
        osymC_cv = symC_cv @ U_vv.T
        print('Old cell like, but symmetrized', osymC_cv)
        osym2C_cv = C_cv @ P_vv
        print('Old cell like, but symmetrized2', osym2C_cv)
        print(Atoms(cell=osymC_cv).cell.angles())
        print(Atoms(cell=osym2C_cv).cell.angles())
        return osymC_cv

class Relax:
    def __init__(self, *, atoms: Atoms, calc: GPAW, optimizer, symprec):
        if atoms.calc is not None:
            raise ValueError("Do not attach a calculator to Atoms yet.")

        if not isinstance(calc, ASECalculator):
            raise ValueError('Calculator must be new GPAW.')

        self.atoms = atoms
        self.calc = calc
        self.optimizer = optimizer
        self.symprec = symprec

        self.symmetries = create_symmetries_object(self.atoms, tolerance=self.symprec) 
        
        unit_cell_symmetry(self.atoms.cell, self.symmetries.rotation_scc)


        # Analyse symmetry of the atoms

if __name__ == "__main__":
    from ase.build import bulk
    atoms = bulk('Au')
    eps_cc = np.random.rand(3,3) * 0.001
    atoms.set_cell(atoms.cell @ (np.eye(3) + eps_cc), scale_atoms=True)
    calc = GPAW(mode='pw')
    print(type(calc))
    relax = Relax(atoms=atoms, calc=calc, optimizer=None, symprec=1e-1)
    
