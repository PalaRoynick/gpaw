from ase.build import molecule
from gpaw.utilities.pointgroup import PointGroup, SPGOperations
from ase import Atoms
from ase.io import read
import numpy as np


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
    assert set(pg.textbook_names_g) == {"E", "1C2_z", "1C2_y", "1C2_x"}


def test_D2h():
    pg = symmetry_from("C2H4")
    assert pg.spg_ops.pointgroup == "D2h"
    assert set(pg.textbook_names_g) == {
        "E",
        "1C2_z",
        "1C2_y",
        "1C2_x",
        "i",
        "1s_xy",
        "1s_xz",
        "1s_yz",
    }


def test_D2d():
    pg = symmetry_from("cyclobutane")
    assert pg.spg_ops.pointgroup == "D2d"
    assert set(pg.textbook_names_g) == {"E", "2S4", "1C2", "2C2'", "2sd"}


def test_C3():
    atoms = Atoms("H2", positions=[[1, 2, 3], [4, 5, 6]])
    atoms2 = atoms.copy()
    atoms3 = atoms.copy()
    atoms2.rotate(120, "z")
    atoms3.rotate(-120, "z")
    atoms.extend(atoms2)
    atoms.extend(atoms3)
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C3"
    assert set(pg.textbook_names_g) == {"E", "1C3", "1C3^2"}


def test_C3h():
    atoms = read("boric_acid.xyz")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C3h"
    assert set(pg.textbook_names_g) == {"E", "1C3", "1C3^2", "1sh", "1S3", "1S3^5"}


def test_C3v():
    atoms = molecule("HCCl3")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C3v"
    assert set(pg.textbook_names_g) == {"E", "2C3", "3sv"}


def test_D3():
    atoms = read("D3.xyz")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "D3"
    assert set(pg.textbook_names_g) == {"E", "2C3", "3C2'"}


def test_D3h():
    atoms = molecule("AlF3")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "D3h"
    assert set(pg.textbook_names_g) == {"E", "2C3", "3C2'", "1sh", "2S3", "3sv"}


def test_D3d():
    atoms = molecule("Si2H6")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "D3d"
    assert set(pg.textbook_names_g) == {"E", "2C3", "3C2'", "i", "2S6", "3sd"}


def test_C4():
    atoms = Atoms(
        "H9",
        positions=[
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [0, 0, 0],
            [1.1, 1, -1],
            [-1, 1.1, -1],
            [1, -1.1, -1],
            [-1.1, -1, -1],
        ],
    )
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C4"
    assert set(pg.textbook_names_g) == {"E", "1C4", "1C2", "1C4^3"}


def test_C4h():
    atoms = Atoms(
        "H9",
        positions=[
            [1.1, 1, 1],
            [-1, 1.1, 1],
            [1, -1.1, 1],
            [-1.1, -1, 1],
            [0, 0, 0],
            [1.1, 1, -1],
            [-1, 1.1, -1],
            [1, -1.1, -1],
            [-1.1, -1, -1],
        ],
    )
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C4h"
    assert set(pg.textbook_names_g) == {
        "E",
        "1C4",
        "1C2",
        "1C4^3",
        "i",
        "1S4^3",
        "1sh",
        "1S4",
    }


def test_C4v():
    atoms = Atoms(
        "H5", positions=[[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [0, 0, 0]]
    )
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C4v"
    assert set(pg.textbook_names_g) == {"E", "2C4", "1C2", "2sv", "2sd"}


def test_S4():
    atoms = read("S4.xyz")
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "S4"
    assert set(pg.textbook_names_g) == {"E", "1S4", "1C2", "1S4^3"}


def test_D4():
    atoms = read("D4.xyz")
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "D4"
    assert set(pg.textbook_names_g) == {"E", "2C4", "1C2", "2C2'", "2C2''"}


def test_D4h():
    atoms = Atoms(
        "CH4", positions=[[0, 0, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]
    )
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "D4h"
    assert set(pg.textbook_names_g) == {
        "E",
        "2C4",
        "1C2",
        "2C2'",
        "2C2''",
        "i",
        "2S4",
        "1sh",
        "2sv",
        "2sd",
    }


def test_S6():
    atoms = read("S6.xyz")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "S6"
    assert set(pg.textbook_names_g) == {"E", "1C3", "1C3^2", "i", "1S6^5", "1S6"}


def test_C6():
    atoms = read("C6.xyz")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C6"
    assert set(pg.textbook_names_g) == {"E", "1C6", "1C3", "1C2", "1C3^2", "1C6^5"}


def test_C6h():
    atoms = read("C6h.xyz")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "C6h"
    assert set(pg.textbook_names_g) == {
        "E",
        "1C6",
        "1C3",
        "1C2",
        "1C3^2",
        "1C6^5",
        "i",
        "1S3^5",
        "1S6^5",
        "1sh",
        "1S6",
        "1S3",
    }


def test_D6h():
    atoms = molecule("C6H6")
    L = 10
    angle = 2 * np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    atoms.center()
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "D6h"
    assert set(pg.textbook_names_g) == {
        "E",
        "2C6",
        "2C3",
        "1C2",
        "3C2'",
        "3C2''",
        "i",
        "2S3",
        "2S6",
        "1sh",
        "3sd",
        "3sv",
    }


def test_Td():
    atoms = read("Td.xyz")
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "Td"
    assert set(pg.textbook_names_g) == {"E", "8C3", "3C2", "6S4", "6sd"}

def test_Oh():
    atoms = read("Al13.xyz")
    pg = symmetry_from(atoms)
    assert pg.spg_ops.pointgroup == "Oh"
    assert set(pg.textbook_names_g) == {"E", "8C3", "6C2", "6C4", "3C2",
                "i", "6S4", "8S6", "3sh", "6sd"}
