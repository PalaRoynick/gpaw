from ase.build import mx2
import numpy as np
from gpaw.utilities.pointgroup import PointGroup, Projectable
from gpaw.utilities.spg_operations import SPGOperations


def create_mos2(pg_type="D3h"):
    """
    Create MoS2 structure for testing point group symmetries.
    Choose between 'D3h' and 'C3v' point groups by removing
    one Mo atom or one S atom.
    """
    primitive = mx2("MoS2", "2H", a=3.16, thickness=3.17, vacuum=5.0)
    primitive.rotate("z", 90, rotate_cell=True)
    primitive.set_pbc(True)
    # transform = np.array([[4, -2, 0],
    #                       [1,  4, 0],
    #                       [0,  0, 1]])
    # atoms = make_supercell(primitive, transform, wrap=True)
    atoms = primitive.repeat((2, 2, 1))
    # atoms.translate([0, 0, 3.1415])
    if pg_type == "C3v":
        atoms[1].symbol = "H"
    elif pg_type == "D3h":
        atoms[0].symbol = "H"

    spg_ops = SPGOperations.from_atoms(atoms)
    origin_ops = spg_ops.apply_origin_shift(-spg_ops.origin_shift_c)
    atoms = atoms.copy()
    print("shift", spg_ops.origin_shift_c @ atoms.cell)
    atoms.translate(spg_ops.origin_shift_c @ atoms.cell)
    print(spg_ops)
    print(origin_ops)
    assert np.allclose(origin_ops.w_sc, 0)

    pg = PointGroup(origin_ops, None)

    if pg_type == "C3v":
        del atoms[1]
    elif pg_type == "D3h":
        del atoms[0]

    from gpaw import GPAW

    if 0:
        calc = GPAW(
            mode={"name": "pw", "force_complex_dtype": True},
            xc="LDA",
            kpts=(1, 1, 1)
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write(f"MoS2_test_{pg_type}.gpw", mode="all")

    calc = GPAW(f"MoS2_test_{pg_type}.gpw")

    for band in range(100):
        for irrep, s in zip(
            pg.character_table.irreps,
            pg.detect_irrep(pg.signature(Projectable.from_calc(calc, band))),
        ):
            print(band, irrep, f"{s.real:.2f}")
    