from ase.build import bulk
from ase.calculators.emt import EMT
from gpaw.new.relax import Relax
import numpy as np
from ase.optimize.bfgs import BFGS
from gpaw.mpi import world

def test_generalized_coordinate_units():
    """Test that displacements are in Å and stresses in eV/Å^3.

    Even the relaxation uses generalized coordinates, they are scaled in such way,
    that at least in the initial configuration, moving generalized coordinate
    an amount x, reflects to the actual coordinate to move an amount of x.

    |dR| = |dz(atom)|
    |deps| = |dz(cell)|  (Cell strain)
    """
    atoms = bulk('AuAg', crystalstructure='wurtzite', a=3.24, c=5.20)
    print(atoms.cell.volume, 'SCALE')
    relax = Relax(atoms=atoms, calc=EMT, optimizer_factory=lambda atoms: BFGS(atoms, trajectory='a.traj'), symprec=0.01, comm=world)
    optimizable = relax.symmetry_adapted_atoms.__ase_optimizable__()
    for z in range(3):
        vec = np.zeros((3,))
        optimizable.set_x(vec)
        atoms0 = optimizable.actual_atoms.copy()
        grad = optimizable.get_gradient()
        F, S = optimizable.actual_atoms.get_forces(), optimizable.actual_atoms.get_stress()

        print('generalized grad', grad[2], 'vs.', F)
        scale = np.max(np.abs(grad[2])) / np.max(np.abs(F))
        print('SCALE F', scale)
        assert 3.9 < scale < 4.1, scale
        
        print('generalized gradS S', grad[:2], 'vs', S)
        scale = np.max(np.abs(grad[:2])) / np.max(np.abs(S))
        print('SCALE S', scale)
        assert 39.9 < scale < 40.1, scale

        vec[z] = 1e-6
        optimizable.set_x(vec)
        atoms1 = optimizable.actual_atoms.copy()
        if z < 2:
            deps = np.sum((atoms1.cell - atoms0.cell)**2)**0.5
            print('deps', deps)
        else:
            dR = np.max(np.linalg.norm(atoms0.get_positions() - atoms1.get_positions(), axis=1))
            print('dR', dR)

def test_fd_gradients():
    atoms = bulk('AuAg', crystalstructure='wurtzite', a=3.24, c=5.20)
    relax = Relax(atoms=atoms, calc=EMT, optimizer_factory=lambda atoms: BFGS(atoms, trajectory='a.traj'), symprec=0.01, comm=world)
    optimizable = relax.symmetry_adapted_atoms.__ase_optimizable__()
    for z in range(3):
        vec = np.zeros((3,))
        optimizable.set_x(vec)
        E0 = optimizable.get_value()
        grad = optimizable.get_gradient() / optimizable.get_preconditioner()
        vec[z] = 1e-6
        optimizable.set_x(vec)
        E1 = optimizable.get_value()
        print("Finite difference grad", (E1 - E0) / 1e-6)
        print("Gotten grad", grad[z])
        div = grad[z] / ((E1 - E0) / 1e-6)
        assert 0.99 < div < 1.01 



