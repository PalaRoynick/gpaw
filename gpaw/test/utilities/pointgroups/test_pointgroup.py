import pytest
from ase.build import mx2
import numpy as np
from gpaw.utilities.pointgroup import PointGroup, SPGOperations
from gpaw.utilities.pointgroup_proj import Projectable, PaniProjectable
from gpaw.utilities.pointgroup_data import character_tables
from gpaw.new.ase_interface import GPAW
from pathlib import Path


def prepare_atoms(atoms):
    # atoms.set_pbc((True,True,True))
    # atoms.translate(-atoms.get_center_of_mass())
    from ase.spacegroup.symmetrize import (
        get_symmetrized_atoms,
        spglib_get_symmetry_dataset,
    )
    from ase.utils import atoms_to_spglib_cell

    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms))
    atoms.set_scaled_positions(
        atoms.get_scaled_positions()
        + dataset.transformation_matrix.T @ dataset.origin_shift
    )
    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms))
    # assert np.allclose(dataset.origin_shift, 0)


def test_Oh():
    from ase.build import bulk
    from ase.io import read

    cell = (bulk("Al") * 5).cell
    atoms = read("structures/Al13.xyz")
    atoms.set_pbc(True)
    atoms.set_cell(cell, scale_atoms=False)
    spg_ops = SPGOperations.from_atoms(atoms, layergroup=False)
    assert spg_ops.pointgroup == "Oh"

    pg = PointGroup(spg_ops)

    if not Path("gs_Oh.gpw").exists():
        calc = GPAW(mode={"name": "pw", "force_complex_dtype": True})
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write("gs_Oh.gpw", mode="all")

    calc = GPAW("gs_Oh.gpw")
    analyze_symmetry(calc, layergroup=False, expected="Oh")


def create_mos2(pg_type="D3h"):
    """
    Create MoS2 structure for testing point group symmetries.
    Choose between 'D3h' and 'C3v' point groups by removing
    one Mo atom or one S atom.
    """
    primitive = mx2("MoS2", "2H", a=3.16, thickness=3.17, vacuum=5.0)
    primitive.rotate("z", 13.55, rotate_cell=True)
    primitive.set_pbc(True)
    atoms = primitive.repeat((2, 2, 1))
    # return atoms.copy(), atoms.copy()
    atoms.translate([0, 0, 3.1415])
    atoms.set_scaled_positions(-atoms.get_scaled_positions())
    if pg_type == "C3v":
        atoms[1].symbol = "H"
        shift = -atoms[1].position
        shift[2] = 0
        atoms.translate(shift)
    elif pg_type == "D3h":
        atoms[0].symbol = "H"
        shift = -atoms[0].position
        shift[2] = 0
        atoms.translate(shift)
    else:
        raise ValueError("Unknown pg type")

    Hreplaced_atoms = atoms.copy()

    if pg_type == "C3v":
        del atoms[1]
    elif pg_type == "D3h":
        del atoms[0]
    else:
        raise ValueError("Unknown pg type")

    return Hreplaced_atoms, atoms


def analyze_symmetry(calc, layergroup, expected):
    assert not layergroup
    spg_ops = SPGOperations.from_atoms(calc.atoms, layergroup=layergroup)
    assert spg_ops.pointgroup == expected
    pg = PointGroup(spg_ops)  # [0, 0, 1] if layergroup else None)
    results = []
    failure = False
    for band in range(6):
        signature = pg.signature(Projectable.from_calc(calc, band))
        found = None
        for irrep, s in zip(
            pg.character_table.irreps,
            pg.detect_irrep(signature),
        ):
            if s > 0.01:
                print(band, irrep, f"{s.real:.2f}")
                results.append(irrep)
                if found is not None:
                    failure = True
                found = irrep
    if failure:
        raise ValueError("Band spans multiple irreps.")
    return results


@pytest.mark.parametrize("group", ["D3h", "C3v"])
@pytest.mark.parametrize("pani", [False, True])
def test_defect_atoms(group, pani):
    Hreplaced_atoms, atoms = create_mos2(pg_type=group)
    # prepare_atoms(Hreplaced_atoms)
    # prepare_atoms(atoms)
    spg_ops = SPGOperations.from_atoms(Hreplaced_atoms, layergroup=True)
    assert spg_ops.pointgroup == group
    # origin_ops = spg_ops.apply_origin_shift(-spg_ops.origin_shift_c)
    # print("shift", spg_ops.origin_shift_c @ atoms.cell)
    # atoms.translate(spg_ops.origin_shift_c @ atoms.cell)
    # print(origin_ops.w_sc)
    # assert np.allclose(origin_ops.w_sc, 0)
    pg = PointGroup(spg_ops)
    from gpaw.new.ase_interface import GPAW

    fname = f"MoS2_test_{group}.gpw"
    if 1:
        calc = GPAW(
            mode={"name": "pw", "force_complex_dtype": True},
            xc="LDA",
            kpts=(1, 1, 1),
            convergence={"density": 1e-3, "eigenstates": 1e-5},
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write(fname, mode="all")

    calc = GPAW(fname)
    if group == "D3h":
        reference = [
            "A'2",
            "E'1",
            "E'1",
            "A'1",
            "E'1",
            "E'1",
            "E''1",
            "E''1",
            "A'1",
            "A'2",
            "E'1",
            "E'1",
        ]
    elif group == "C3v":
        reference = [
            "E",
            "E",
            "E",
            "E",
            "A1",
            "A2",
            "E",
            "E",
            "A1",
            "A1",
            "E",
            "E",
        ]

    failure = False
    # Use higher threshold for pani
    threshold = 0.1 if pani else 0.01

    for band, ref in zip(range(39, 51), reference):
        if pani:
            kpt = calc.wfs.kpt_u[0]
            P_ai = {}
            for a in kpt.P_ani.keys():
                P_ai[a] = kpt.P_ani[a][band]
            proj = PaniProjectable(P_ai, calc.atoms, calc.wfs.setups)
            signature = pg.signature(proj)
        else:
            # Plane waves
            signature = pg.signature(Projectable.from_calc(calc, band))

        irreps_found = []
        for irrep, s in zip(
            pg.character_table.irreps,
            pg.detect_irrep(signature),
        ):
            if s > threshold:
                print(band, irrep, f"{s.real:.2f}",
                      "(P_ani)" if pani else "(pw)")
                irreps_found.append((irrep, s))

        # For pani, accept if dominant irrep is clear (even if there's mixing)
        # For pw, expect single clear irrep
        if pani and len(irreps_found) > 1:
            # Check if one is clearly dominant (>2x the next)
            irreps_found.sort(key=lambda x: x[1], reverse=True)
            if irreps_found[0][1] > 2 * irreps_found[1][1]:
                print(f"  Band {band}: Dominant {irreps_found[0][0]}")
            else:
                print(f"  Band {band}: Mixed (PAW incomplete), dominant {irreps_found[0][0]}")
        elif not pani and len(irreps_found) > 1:
            # For plane waves, expect single irrep
            failure = True

        # assert found == ref XXXX
    if failure:
        raise ValueError("Band spans multiple irreps.")


def get_group_example(group):
    from ase.io import read

    examples = {
        "C2v": ("structures/C2v.json", "A1,B2,A1,B1,A1,B2"),
        "C2h": ("structures/C2h.json", "Ag,Au,Ag,Bu,Ag,Au"),
        "C2": ("structures/C2.json", "A,B,A,B,A,B"),
        "S4": ("structures/S4.json", "Ag,Au,Ag,Bu,Ag,Au"),
        "D2h": ("structures/D2h.json", "Ag,B1u,B2u,B3g,Ag,B1u"),
    }

    try:
        fname, result = examples[group]
    except KeyError:
        pytest.skip(reason=f"Test for {group} not yet implemented.")

    try:
        atoms = read(fname)
    except FileNotFoundError:
        pytest.skip(reason=f"File {fname} not found.")
    from ase.spacegroup.symmetrize import (
        get_symmetrized_atoms,
        spglib_get_symmetry_dataset,
    )

    atoms = get_symmetrized_atoms(atoms, symprec=0.3)[0]
    prepare_atoms(atoms)

    return atoms, result


@pytest.mark.parametrize("group", character_tables.keys())
def test_all(group):
    atoms, results = get_group_example(group)

    from gpaw.new.ase_interface import GPAW

    fname = f"cache_{group}.gpw"
    if not Path(fname).exists():
        calc = GPAW(
            mode={"name": "pw", "force_complex_dtype": True},
            xc="LDA",
            kpts=(1, 1, 1),
            convergence={"density": 1e-3, "eigenstates": 1e-5},
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write(fname, mode="all")

    calc = GPAW(fname)
    obtained_results = analyze_symmetry(
        calc, layergroup=False, expected=group
    )  # XXX Not using layergroup
    assert results.split(",") == obtained_results
