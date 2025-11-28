import pytest
from ase.build import mx2
import numpy as np
from gpaw.utilities.pointgroup import PointGroup, Projectable, SPGOperations
from gpaw.utilities.pointgroup_data import character_tables
from gpaw.new.ase_interface import GPAW
from pathlib import Path

def test_Oh():
    from ase.build import bulk
    from ase.io import read
    cell = (bulk('Al') * 5).cell
    atoms = read('structures/Al13.xyz')
    atoms.set_pbc(True)
    atoms.set_cell(cell, scale_atoms=False)

    spg_ops = SPGOperations.from_atoms(atoms, layergroup=False)
    pg = PointGroup(spg_ops, principal_axis=None)

    if not Path('gs_Oh.gpw').exists():
        calc = GPAW(mode={'name': 'pw', 'force_complex_dtype':True})
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('gs_Oh.gpw', mode='all')

    calc = GPAW('gs_Oh.gpw')
    analyze_symmetry(calc, layergroup=False)

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


def analyze_symmetry(calc, layergroup):
    spg_ops = SPGOperations.from_atoms(calc.atoms, layergroup=layergroup)
    pg = PointGroup(spg_ops, [0,0,1] if layergroup else None)
    results = []
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
                assert found is None
                found = irrep
    return results


@pytest.mark.parametrize("group", ["D3h", "C3v"])
def test_defect_atoms(group):
    Hreplaced_atoms, atoms = create_mos2(pg_type=group)

    spg_ops = SPGOperations.from_atoms(Hreplaced_atoms, layergroup=True)
    origin_ops = spg_ops.apply_origin_shift(-spg_ops.origin_shift_c)
    # print("shift", spg_ops.origin_shift_c @ atoms.cell)
    atoms.translate(spg_ops.origin_shift_c @ atoms.cell)
    print(origin_ops.w_sc)
    assert np.allclose(origin_ops.w_sc, 0)
    pg = PointGroup(origin_ops)

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

    for band, ref in zip(range(39, 51), reference):
        signature = pg.signature(Projectable.from_calc(calc, band))
        found = None
        for irrep, s in zip(
            pg.character_table.irreps,
            pg.detect_irrep(signature),
        ):
            if s > 0.01:
                print(band, irrep, f"{s.real:.2f}")
                assert found is None
                found = irrep
        assert found == ref


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
        pytest.skip(msg=f"Test for {group} not yet implemented.")

    atoms = read(fname)
    return atoms, result

@pytest.mark.parametrize('group', character_tables.keys())
def test_all(group):
    atoms, results = get_group_example(group)

    from gpaw.new.ase_interface import GPAW
    fname = f'cache_{group}.gpw'
    if not Path(fname).exists():
        calc = GPAW(
            mode={"name": "pw", "force_complex_dtype": True},
            xc="LDA",
            kpts=(1, 1, 1),
            convergence={"density": 1e-3, "eigenstates": 1e-5},
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write(fname, mode='all')

    calc = GPAW(fname)
    obtained_results = analyze_symmetry(calc, layergroup=True)
    assert results.split(',') == obtained_results
