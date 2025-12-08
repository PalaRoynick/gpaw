from ase.build import molecule
from gpaw.new.ase_interface import GPAW
from gpaw.utilities.pointgroup import PointGroup, SPGOperations
from gpaw.utilities.pointgroup_proj import PaniProjectable
from gpaw.utilities.pointgroup_projections import SymmetryEigenvalues
from pathlib import Path
import numpy as np
from numpy import pi, sin, cos
import pytest


def build_cell(atoms, group):
    """Prepare molecule cell same as in test_molecules.py"""
    if "3" in group or "6" in group:
        L = 10
        angle = 2 * pi / 3
        c, s = cos(angle), sin(angle)
        cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
        atoms.set_cell(cell)
        atoms.set_pbc((True, True, True))
        atoms.center()
    else:
        atoms.center(vacuum=4)

    atoms.translate(-atoms.get_center_of_mass())
    from ase.spacegroup.symmetrize import (
        spglib_get_symmetry_dataset,
    )
    from ase.utils import atoms_to_spglib_cell

    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms))
    atoms.set_scaled_positions(
        atoms.get_scaled_positions()
        + dataset.transformation_matrix.T @ dataset.origin_shift
    )
    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms))
    assert np.allclose(dataset.origin_shift, 0)


@pytest.mark.parametrize("translation", [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.5, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.7]),
    np.array([0.2, 1.3, 3.4]),
])
@pytest.mark.parametrize("rotation_axis", [
    None,  # No rotation
    np.array([0.0, 0.0, 1.0]),  # Rotate around z
    np.array([1.0, 0.0, 0.0]),  # Rotate around x
])
@pytest.mark.parametrize("permute_axes", [
    [0, 1, 2],  # No permutation
    [1, 0, 2],  # Swap x and y
    [2, 1, 0],  # Swap x and z
])
def test_h2o_pani(translation, rotation_axis, permute_axes):
    """Test H2O using PAniProjectable against turbomole reference"""
    name = "H2O"
    symmetry = "c2v"

    atoms = molecule(name)
    build_cell(atoms, symmetry)

    # Apply transformations
    # 1. Permute axes
    if permute_axes != [0, 1, 2]:
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        atoms.set_positions(pos[:, permute_axes])
        atoms.set_cell(cell[permute_axes][:, permute_axes])

    # 2. Rotate
    if rotation_axis is not None:
        from ase.build import rotate
        angle = 37.5  # arbitrary angle in degrees
        atoms.rotate(angle, rotation_axis, center='COM')

    # 3. Translate
    atoms.translate(translation)

    # Load turbomole reference
    tmole_json = Path(name + "_tmole.json")
    if not tmole_json.exists():
        raise FileNotFoundError(
            f"Turbomole reference {tmole_json} not found. "
            f"Run test_molecules.py::test_molecule[H2O] first to generate it."
        )

    tmole_states = SymmetryEigenvalues.load(tmole_json)
    tmole_states = tmole_states.unroll_degeneracies().occupied_states

    # Load GPAW calculation
    gpw_file = Path(name) / "wfs.gpw"
    if not gpw_file.exists():
        raise FileNotFoundError(
            f"GPAW file {gpw_file} not found. "
            f"Run test_molecules.py::test_molecule[H2O] first to generate it."
        )

    calc = GPAW(str(gpw_file))

    # Get symmetry operations
    spg_ops = SPGOperations.from_atoms(calc.atoms, layergroup=False)
    assert spg_ops.pointgroup.upper() == symmetry.upper()

    pg = PointGroup(spg_ops)
    kpt = calc.wfs.kpt_u[0]

    # Analyze bands using PAniProjectable
    eig_n = calc.get_eigenvalues()
    occ_n = calc.get_occupation_numbers()

    pani_irreps = []
    for n, (eig, occ) in enumerate(zip(eig_n, occ_n)):
        if occ > 0.01:  # Only occupied states
            P_ai = {}
            for a in kpt.P_ani.keys():
                P_ai[a] = kpt.P_ani[a][n]

            proj = PaniProjectable(P_ai, calc.atoms, calc.wfs.setups)
            signature = pg.signature(proj)
            irrep_weights = pg.detect_irrep(signature)

            # Find dominant irrep
            found = None
            for irrep, s in zip(pg.character_table.irreps, irrep_weights):
                if s.real > 0.01:
                    if found is not None:
                        raise ValueError(f"Band {n} spans multiple irreps.")
                    found = irrep

            if found is not None:
                pani_irreps.append(found)
                print(f"{n} {found} {eig:.4f}")

    print(f"P_ani irreps: {pani_irreps}")
    print(f"TMOLE irreps: {[s.irrep for s in tmole_states]}")

    # Compare with turbomole
    comparable = min(len(pani_irreps), len(tmole_states))
    pani_irreps_cmp = pani_irreps[-comparable:]
    tmole_states_cmp = tmole_states[-comparable:]

    print(f"Comparing P_ani: {pani_irreps_cmp}")
    print(f"Comparing TMOLE: {[s.irrep for s in tmole_states_cmp]}")

    for pani_irrep, tmole_state in zip(pani_irreps_cmp, tmole_states_cmp):
        assert sorted(tmole_state.irrep.upper()) == sorted(pani_irrep.upper())


if __name__ == '__main__':
    test_h2o_pani()
