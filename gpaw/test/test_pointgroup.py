import pytest
from ase.build import mx2
import numpy as np
from gpaw.utilities.pointgroup import PointGroup, Projectable, SPGOperations


def create_mos2(pg_type="D3h"):
    """
    Create MoS2 structure for testing point group symmetries.
    Choose between 'D3h' and 'C3v' point groups by removing
    one Mo atom or one S atom.
    """
    primitive = mx2("MoS2", "2H", a=3.16, thickness=3.17, vacuum=5.0)
    primitive.rotate("z", 90, rotate_cell=True)
    primitive.set_pbc(True)
    atoms = primitive.repeat((2, 2, 1))
    #atoms.translate([0, 0, 3.1415])
    if pg_type == "C3v":
        atoms[1].symbol = "H"
    elif pg_type == "D3h":
        atoms[0].symbol = "H"
    else:
        raise ValueError('Unknown pg type')

    Hreplaced_atoms = atoms.copy()
    
    if pg_type == "C3v":
        del atoms[1]
    elif pg_type == "D3h":
        del atoms[0]
    else:
        raise ValueError('Unknown pg type')

    return Hreplaced_atoms, atoms

@pytest.mark.parametrize('group', ['D3h', 'C3v'])
def test_point_groups(group):
    Hreplaced_atoms, atoms = create_mos2(pg_type=group)

    spg_ops = SPGOperations.from_atoms(Hreplaced_atoms)
    origin_ops = spg_ops.apply_origin_shift(-spg_ops.origin_shift_c)
    #print("shift", spg_ops.origin_shift_c @ atoms.cell)
    atoms.translate(spg_ops.origin_shift_c @ atoms.cell)
    print(origin_ops.w_sc)
    assert np.allclose(origin_ops.w_sc, 0)
    pg = PointGroup(origin_ops, None)


    from gpaw.new.ase_interface import GPAW

    fname = f"MoS2_test_{group}.gpw"
    if 0:
        calc = GPAW(
            mode={"name": "pw", "force_complex_dtype": True},
            xc="LDA",
            kpts=(1, 1, 1),
            convergence={'density':1e-3, 'eigenstates': 1e-5},
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write(fname, mode="all")

    calc = GPAW(fname)

    reference = """A'2 
E'1 
E'1 
A'1 
E'1 
E'1 
E''1
E''1
A'1 
A'2 
E'1 
E'1""".split('\n')

    for band, ref in zip(range(39, 51), reference):
        ref = ref.strip() # Kartik
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


