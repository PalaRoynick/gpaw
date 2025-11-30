from ase.build import molecule
from gpaw.utilities.pointgroup import PointGroup, SPGOperations
from ase import Atoms
from ase.io import read


def symmetry_from(mol):
    if isinstance(mol, str):
        atoms = molecule(mol)
    else:
        atoms = mol
    atoms.center(vacuum=5)
    atoms.set_pbc((True, True, True))
    spg_ops = SPGOperations.from_atoms(atoms, False, layergroup=False)
    return PointGroup(spg_ops, None)


def test_C1():
    pg = symmetry_from("H2COH")
    assert set(pg.names_g) == {"E"}
    assert pg.spg_ops.pointgroup == "C1"


def test_Ci():
    atoms = Atoms("H3", positions=[[1, 2, 3], [4, 5, 6], [2, 1, 3]])
    atoms.center(vacuum=5)
    atoms2 = atoms.copy()
    atoms2.set_scaled_positions(-atoms2.get_scaled_positions())
    atoms.extend(atoms2)
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "Ci"
    assert set(pg.names_g) == {"E", "i"}


def test_Cs():
    pg = symmetry_from("C3H6_Cs")
    assert pg.spg_ops.pointgroup == "Cs"
    assert set(pg.names_g) == {"E", "1sh"}


def test_C2():
    pg = symmetry_from("H2O2")
    assert pg.spg_ops.pointgroup == "C2"
    assert set(pg.names_g) == {"E", "1C2"}


def test_C2h():
    pg = symmetry_from("OCHCHO")
    assert pg.spg_ops.pointgroup == "C2h"
    assert set(pg.names_g) == {"E", "1C2", "i", "1sh"}


def test_C2v():
    pg = symmetry_from("H2O")
    assert pg.spg_ops.pointgroup == "C2v"
    assert set(pg.names_g) == {"E", "1C2", "1sv_xz", "1sv_yz"}


def test_D2():
    pg = symmetry_from(read("twistane.xyz"))
    assert pg.spg_ops.pointgroup == "D2"
    assert set(pg.names_g) == {"E", "1C2_z", "1C2_y", "1C2_x"}


def test_D2h():
    pg = symmetry_from("C2H4")
    assert pg.spg_ops.pointgroup == "D2h"
    assert set(pg.textbook_names_g) == {"E", "1C2_z", "1C2_y", "1C2_x",
                "i", "1s_xy", "1s_xz", "1s_yz"}

