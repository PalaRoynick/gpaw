import pytest
from ase.build import mx2
import numpy as np
from gpaw.utilities.pointgroup import PointGroup, SPGOperations
from gpaw.utilities.pointgroup_proj import Projectable, PaniProjectable
from gpaw.utilities.pointgroup_data import character_tables
from gpaw.new.ase_interface import GPAW
from pathlib import Path


def prepare_atoms(atoms):
    return
    from ase.spacegroup.symmetrize import (
        spglib_get_symmetry_dataset,
    )
    from ase.utils import atoms_to_spglib_cell

    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms))
    atoms.set_scaled_positions(
        atoms.get_scaled_positions()
        + np.linalg.inv(dataset.transformation_matrix) @ dataset.origin_shift
    )
    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms))


def test_Oh():
    from ase.build import bulk
    from ase.io import read

    cell = (bulk("Al") * 5).cell
    atoms = read("structures/Al13.xyz")
    atoms.set_pbc(True)
    atoms.set_cell(cell, scale_atoms=False)
    spg_ops = SPGOperations.from_atoms(atoms, layergroup=False)
    assert spg_ops.pointgroup == "Oh"

    if not Path("gs_Oh.gpw").exists():
        calc = GPAW(mode={"name": "pw", "force_complex_dtype": True})
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write("gs_Oh.gpw", mode="all")

    calc = GPAW("gs_Oh.gpw")
    analyze_symmetry(calc, layergroup=False, expected="Oh")


def create_mos2(pg_type="D3h", n_sc=2):
    primitive = mx2("MoS2", "2H", a=3.16, thickness=3.17, vacuum=5.0)
    primitive.set_pbc(True)
    atoms = primitive.repeat((n_sc, n_sc, 1))

    if pg_type == "C3v":
        del atoms[1]
    elif pg_type == "D3h":
        del atoms[0]
    else:
        raise ValueError("Unknown pg type")

    return atoms


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
def test_defect_atoms(group):
    atoms = create_mos2(pg_type=group, n_sc=2)
    spg_ops = SPGOperations.from_atoms(atoms, layergroup=True)
    assert spg_ops.pointgroup == group

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
            "E'",
            "E'",
            "A'1",
            "E'",
            "E'",
            "E''",
            "E''",
            "A'1",
            "A'2",
            "E'",
            "E'",
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
    
    threshold = 0.01 # Worked with 1e-4 too!

    for band, ref in zip(range(39, 51), reference):
        kpt = calc.wfs.kpt_u[0]
        P_ai = {}
        for a in kpt.P_ani.keys():
            P_ai[a] = kpt.P_ani[a][band]
        proj = PaniProjectable(P_ai, calc.atoms, calc.wfs.setups)
        signature_pani = pg.signature(proj)
        # Plane waves
        signature_pw = pg.signature(Projectable.from_calc(calc, band))

        for pani, signature in [(True, signature_pani), (False, signature_pw)]:
            irreps_found = []
            for irrep, s in zip(
                pg.character_table.irreps,
                pg.detect_irrep(signature),
            ):
                if s > threshold:
                    print(band, irrep, f"{s.real:.2f}",
                          "(P_ani)" if pani else "(pw)")
                    irreps_found.append((irrep, s))
                    assert irrep == ref, \
                        f"Band {band} ({'P_ani' if pani else 'pw'}): " \
                        f"expected {ref}, got {irrep}"


@pytest.mark.parametrize("group", ["C3v", "D3h"])
def test_cluster_mapping(group):
    """Test finding point group from cluster and
    mapping P_ani from supercell."""
    # TODO: Make it more robust? In practice, the defect cluster
    # is made by trying different cutoffs and choosing the one with
    # highest symmetry.
    # TODO: Smaller supercell for the test!
    n_sc = 3
    atoms = create_mos2(pg_type=group, n_sc=n_sc)

    # Store pristine to find vacancy position
    pristine = mx2("MoS2", "2H", a=3.16, thickness=3.17, vacuum=5.0)
    pristine = pristine.repeat((n_sc, n_sc, 1))
    # Run GS on defect supercell
    calc = GPAW(
        mode={"name": "pw", "force_complex_dtype": True},
        kpts=(1, 1, 1),
        convergence={"density": 1e-3, "eigenstates": 1e-5},
    )
    atoms.calc = calc
    atoms.get_potential_energy()

    # Find vacancy position by comparing with pristine
    defect_positions = atoms.get_positions()
    pristine_positions = pristine.get_positions()
    dist_matrix = np.linalg.norm(pristine_positions[:, None, :]
                                 - defect_positions[None, :, :],
                                 axis=2)
    min_distances = np.min(dist_matrix, axis=1)
    missing = np.where(min_distances > 0.2)[0]
    vacancy_pos = pristine[missing[0]].position.copy()

    # Center atoms on defect
    defect_in_middle = atoms.cell.sum(0) / 2
    translation_xy = defect_in_middle[:2] - vacancy_pos[:2]
    translation = np.array([translation_xy[0], translation_xy[1], 0.0])
    atoms_centered = atoms.copy()
    atoms_centered.set_positions(atoms_centered.get_positions() + translation)
    atoms_centered.wrap()

    # Update vacancy position
    vacancy_pos_shifted = vacancy_pos.copy()
    vacancy_pos_shifted[:2] += translation_xy
    frac = np.linalg.solve(atoms.cell.T, vacancy_pos_shifted)
    frac %= 1.0
    vacancy_pos_shifted = atoms.cell.T @ frac

    # Extract cluster within cutoff
    cutoff_radius = 4.7  # 5.0 was including some extra atoms
    positions_xy = atoms_centered.get_positions()[:, :2]
    dist_xy = np.linalg.norm(positions_xy - vacancy_pos_shifted[:2], axis=1)
    cluster_mask = dist_xy < cutoff_radius
    cluster_indices = np.where(cluster_mask)[0]
    new_defect = atoms_centered[cluster_indices]

    # Embed cluster in large hexagonal cell
    hex_cell = pristine.repeat((2, 2, 1))
    lattice = hex_cell.cell.copy()
    lattice[2] = atoms.cell[2]
    positions = np.linalg.solve(lattice.T, new_defect.get_positions().T).T

    # Center vacancy in hexagonal cell
    vacancy_frac = np.linalg.solve(lattice.T, vacancy_pos_shifted)
    center_frac_xy = np.array([0.5, 0.5])
    delta_frac_xy = vacancy_frac[:2] - center_frac_xy
    positions[:, :2] -= delta_frac_xy
    positions[:, :2] %= 1.0

    # Create cluster
    from ase import Atoms
    cluster = Atoms(
        symbols=new_defect.get_chemical_symbols(),
        scaled_positions=positions,
        cell=lattice,
        pbc=[False, False, False]
    )

    spg_ops = SPGOperations.from_atoms(cluster, layergroup=True)
    assert spg_ops.pointgroup == group
    pg = PointGroup(spg_ops)

    # Map P_ani from supercell to cluster and analyze
    kpt = calc.wfs.kpt_u[0]
    n_bands = len(kpt.f_n)
    print(f"\nAnalyzing all {n_bands} bands:")
    for band in range(n_bands):
        # TODO: Does the naming make sense here?
        P_ai_cluster = {a: kpt.P_ani[cluster_indices[a]][band]
                        for a in range(len(cluster)) if cluster_indices[a]
                        in kpt.P_ani}
        cluster_setups = {a: calc.wfs.setups[cluster_indices[a]]
                          for a in range(len(cluster))}

        proj = PaniProjectable(P_ai_cluster, cluster, cluster_setups)
        signature = pg.signature(proj)
        irrep_weights = pg.detect_irrep(signature)

        # Format irreps with weights > 0.01
        irrep_str = " ".join([
            f"{irrep}:{w.real:.2f}"
            for irrep, w in zip(pg.character_table.irreps, irrep_weights)
            if abs(w) > 0.01
        ])
        if not irrep_str:
            irrep_str = "mixed"

        if group == "C3v":
            reference = [
                "A2",
                "E",
                "E",
                "A2",
                "A1",
                "A1",
                "E",
                "E",
                "E",
                "E",
                "A1",
                "E",
                "E",
                "A1",
                "E",
                "E",
                "E",
                "E",
                "A2",
            ]
        elif group == "D3h":
            reference = [
                "E'",
                "A'1",
                "A'2",
                "E'",
                "E'",
                "A'1",
                "E'",
                "E'",
                "E''",
                "E''",
                "A'1",
                "E'",
                "E'",
                "A'2",
                "A'1",
                "E'",
                "E'",
                "A'2",
                "A'1",
            ]

        # Assert that the dominant irrep matches reference
        if 100 <= band <= 118:
            ref_irrep = reference[band - 100]
            for irrep, w in zip(pg.character_table.irreps, irrep_weights):
                if abs(w) > 0.01:
                    assert irrep == ref_irrep


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
