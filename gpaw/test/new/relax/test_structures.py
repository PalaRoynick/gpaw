import pytest
from ase.build import bulk
from ase.data import chemical_symbols, atomic_numbers
from ase.calculators.emt import EMT
import numpy as np


def generate_all_bulk_structures():
    cases = []

    for Z in range(1, len(chemical_symbols)):
        symbol = chemical_symbols[Z]
        if symbol in {'Np', 'U', 'Sb', 'As', 'Bi'}:
            continue
        try:
            atoms = bulk(symbol)
            from ase.build import niggli_reduce
            niggli_reduce(atoms)
            print(atoms.cell.angles())
        except Exception:
            continue  # lattice not valid for this element
        cases.append((symbol, atoms))
    return cases


@pytest.fixture(scope="session", params=generate_all_bulk_structures())
def system(request):
    symbol, atoms = request.param
    atoms = atoms.copy()
    atoms.set_chemical_symbols(['Au'] * len(atoms))
    return symbol, atoms


def test_symmetry_constrained_relaxation_emt(system):
    symbol, atoms_ref = system
    atoms_ref = atoms_ref.copy()
    atoms = atoms_ref.copy()

    # Symmetry adapted relaxation
    from gpaw.new.relax import Relax
    from ase.optimize.bfgs import BFGS
    relax = Relax(atoms=atoms, calc=EMT(), optimizer_factory=lambda atoms: BFGS(atoms, alpha=100.0, trajectory='new.traj'), symprec=0.01)
    relax.run(fmax=0.0001, smax=0.00001)

    print('Relax complete', atoms.get_stress())
    from ase.filters import FrechetCellFilter 
    from ase.optimize.cellawarebfgs import CellAwareBFGS
    atoms_ref.calc = EMT()
    relax = CellAwareBFGS(FrechetCellFilter(atoms_ref, exp_cell_factor=1.0), trajectory='old.traj')
    relax.run(fmax=0.0001, smax=0.00001)
    print('Ref relax complete', atoms_ref.get_stress())
    
    print(atoms.cell.lengths(), atoms_ref.cell.lengths())
    print(atoms.cell.angles(), atoms_ref.cell.angles())

    assert np.allclose(atoms.cell.lengths(), atoms_ref.cell.lengths(), atol=0.01)
    assert np.allclose(atoms.cell.angles(), atoms_ref.cell.angles(), atol=0.01)

