"""Direct unit tests for Symmetries.analyze_displacements and
Symmetries.symmetrize_displaced_forces (gpaw/new/symmetry.py).

All Symmetries objects are built by hand (cell + integer fractional
rotations + atom maps) so every expectation can be verified on paper.
Conventions: row vectors, e' = e @ R_vv, F'[amap[a]] = F[a] @ R_vv,
R_vv = inv(cell) @ U_cc @ cell.
"""
import numpy as np
import pytest
from ase.build import bulk

from gpaw.new.symmetry import Symmetries, create_symmetries_object

I3 = np.eye(3, dtype=int)
C2Z = np.diag([-1, -1, 1])  # 180 deg about z (cartesian == fractional
#                             for the cells used below)


def c2z_symmetries(n_atoms=1):
    """{E, C2z} on a cubic cell; every op fixes every atom."""
    return Symmetries(cell=[1, 1, 1],
                      rotations=[I3, C2Z],
                      atommaps=[list(range(n_atoms))] * 2)


def swap_symmetries():
    """{E, sigma_z} with sigma_z exchanging the two atoms."""
    return Symmetries(cell=[1, 1, 2],
                      rotations=[I3, np.diag([1, 1, -1])],
                      atommaps=[[0, 1], [1, 0]])


def hexagonal_c2v_symmetries():
    """{E, C2z, sigma(30 deg), sigma(120 deg)} on a hexagonal cell.

    The mirror lines lie at 30/120 degrees from the x-axis, so the two
    mirrors map x/y onto non-cardinal directions (their fractional
    matrices are exact integers; the cartesian ones mix x and y).
    """
    cell = [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 2]]
    sigma30 = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]])
    sigma120 = np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]])
    return Symmetries(cell=cell,
                      rotations=[I3, C2Z, sigma30, sigma120],
                      atommaps=[[0]] * 4)


# ----------------------------------------------------------------------
# analyze_displacements
# ----------------------------------------------------------------------

def test_analyze_needs_atom_maps():
    sym = Symmetries.from_cell([1, 1, 1])  # no atom maps
    with pytest.raises(ValueError, match='atom maps'):
        sym.analyze_displacements()


def test_analyze_structure():
    sym = c2z_symmetries(n_atoms=2)
    disp = sym.analyze_displacements()

    # One key per atom, axis and sign; entries per retained op, in order.
    assert set(disp) == {f'{a}{ax}{s}'
                         for a in (0, 1) for ax in 'xyz' for s in '+-'}
    entry = disp['0x+']
    assert set(entry) == {'rotation_scc', 'atommap_sa', 'source_keys'}
    assert entry['rotation_scc'].shape == (2, 3, 3)
    assert np.issubdtype(entry['rotation_scc'].dtype, np.integer)
    assert entry['atommap_sa'].shape == (2, 2)
    # Result is also stored on the object.
    assert sym.displaced_symmetries is disp


def test_analyze_source_keys_signed_permutation():
    # C2z: x -> -x, y -> -y, z -> z, atoms fixed.
    disp = c2z_symmetries().analyze_displacements()
    assert disp['0x+']['source_keys'] == ['0x+', '0x-']
    assert disp['0y-']['source_keys'] == ['0y-', '0y+']
    assert disp['0z+']['source_keys'] == ['0z+', '0z+']


def test_analyze_atom_swap_uses_preimage_atom():
    # sigma_z exchanges the atoms and flips z: the source of a target on
    # atom 0 is the corresponding displacement of atom 1.
    disp = swap_symmetries().analyze_displacements()
    assert disp['0x+']['source_keys'] == ['0x+', '1x+']
    assert disp['0z+']['source_keys'] == ['0z+', '1z-']
    assert disp['1z-']['source_keys'] == ['1z-', '0z+']


def test_analyze_drops_non_cardinal_sources():
    # The 30/120-degree mirrors map x/y onto mixed directions -> dropped;
    # z is fixed by all four ops -> full group retained.
    sym = hexagonal_c2v_symmetries()
    disp = sym.analyze_displacements()

    assert disp['0x+']['source_keys'] == ['0x+', '0x-']  # E, C2z only
    assert disp['0y+']['source_keys'] == ['0y+', '0y-']
    assert disp['0z+']['source_keys'] == ['0z+'] * 4     # all ops
    assert disp['0x+']['rotation_scc'].shape == (2, 3, 3)


def test_analyze_structure_si_cubic():
    """Si diamond: full cubic point group, everything retained."""
    sym = create_symmetries_object(bulk('Si', 'diamond', cubic=False))
    disp = sym.analyze_displacements()

    # 2 atoms * 3 axes * 2 signs
    assert len(disp) == 12
    # Contributions are restricted to ops whose source displacement is one
    # of the six measured cardinal displacements (stored as its key).  For a
    # cubic group every op is a signed axis permutation, so all 24 are
    # retained.  Duplicate source keys are expected.
    entry = disp['0x+']
    assert set(entry) == {'rotation_scc', 'atommap_sa', 'source_keys'}
    assert entry['rotation_scc'].shape == (24, 3, 3)
    # Stored rotations are the exact integer (fractional) operations.
    assert np.issubdtype(entry['rotation_scc'].dtype, np.integer)
    assert entry['atommap_sa'].shape == (24, 2)
    assert len(entry['source_keys']) == 24
    # Identity op contributes the target itself; cubic axis permutations
    # pull from other directions too.
    assert '0x+' in entry['source_keys']
    assert {'0y-', '0z-'} <= set(entry['source_keys'])


def test_analyze_hexagonal_drops_mixing():
    """Ops with non-cardinal source directions are excluded by design.

    Hexagonal 3-/6-fold rotations map x onto directions mixing x and y;
    since only the six cardinal displacements are measured, those ops are
    dropped for in-plane targets (z targets keep the full group).
    """
    sym = create_symmetries_object(bulk('Mg', 'hcp', a=3.21, c=5.21))
    disp = sym.analyze_displacements()
    n_ops = len(sym)

    entry_x = disp['0x+']
    assert 0 < len(entry_x['rotation_scc']) < n_ops
    assert '0x+' in entry_x['source_keys']  # identity always retained

    # Every op maps z onto +-z, so the z target retains the whole group.
    assert len(disp['0z+']['rotation_scc']) == n_ops


def test_analyze_identity_always_retained():
    # The identity maps every displacement onto itself, so every target
    # contains its own key (the stabilizer is never empty).
    for sym in (c2z_symmetries(), swap_symmetries(),
                hexagonal_c2v_symmetries()):
        for key, entry in sym.analyze_displacements().items():
            assert key in entry['source_keys']


# ----------------------------------------------------------------------
# symmetrize_displaced_forces
# ----------------------------------------------------------------------

def test_symmetrize_requires_analysis():
    sym = c2z_symmetries()
    with pytest.raises(KeyError, match='analyze_displacements'):
        sym.symmetrize_displaced_forces(['0x+'], {'0x+': np.zeros((1, 3))})


def test_symmetrize_missing_source_key_raises():
    sym = c2z_symmetries()
    sym.analyze_displacements()
    # C2z needs '0x-' as a source for '0x+'; it must not be skipped.
    with pytest.raises(KeyError):
        sym.symmetrize_displaced_forces(['0x+'],
                                        {'0x+': np.zeros((1, 3))})


def test_symmetrize_average_by_hand():
    # F_sym('0x+') = (F['0x+'] @ E + F['0x-'] @ C2z) / 2
    #              = ([a, b, c] + [-d, -e, f]) / 2
    sym = c2z_symmetries()
    sym.analyze_displacements()
    forces = {'0x+': [[1.0, 2.0, 3.0]],
              '0x-': [[4.0, 5.0, 6.0]]}
    F_sym = sym.symmetrize_displaced_forces(['0x+'], forces)['0x+']
    assert np.allclose(F_sym, [[(1 - 4) / 2, (2 - 5) / 2, (3 + 6) / 2]])


def test_symmetrize_permutes_rows_on_atom_swap():
    # sigma_z maps atom 1 -> atom 0, so the rotated rows of the source
    # '1x+' land row-permuted: row a of F['1x+'] contributes to row
    # amap[a] of the target.
    sym = swap_symmetries()
    sym.analyze_displacements()
    forces = {'0x+': np.array([[1.0, 0, 0], [2.0, 0, 0]]),
              '1x+': np.array([[3.0, 0, 0], [4.0, 0, 0]])}
    F_sym = sym.symmetrize_displaced_forces(['0x+'], forces)['0x+']
    # sigma_z leaves x-components alone; rows of '1x+' swap:
    # ([1, 2] + [4, 3]) / 2 = [2.5, 2.5]
    assert np.allclose(F_sym[:, 0], [2.5, 2.5])
    assert np.allclose(F_sym[:, 1:], 0)


def test_symmetrize_supercell_rows_in_place():
    # More force rows than symmetry atoms (primitive symmetry, supercell
    # forces): rows are rotated in place, never permuted.
    sym = c2z_symmetries(n_atoms=1)
    sym.analyze_displacements()
    rng = np.random.default_rng(7)
    forces = {'0x+': rng.standard_normal((5, 3)),
              '0x-': rng.standard_normal((5, 3))}
    F_sym = sym.symmetrize_displaced_forces(['0x+'], forces)['0x+']
    assert F_sym.shape == (5, 3)
    expected = 0.5 * (forces['0x+'] + forces['0x-'] @ C2Z)
    assert np.allclose(F_sym, expected)


def test_symmetrize_symmetric_input_is_fixed_point():
    # An isotropic on-site response is invariant under the full cubic
    # group: symmetrizing must return it unchanged.
    ops = Symmetries.from_cell([1, 1, 1]).rotation_scc  # 48 ops
    sym = Symmetries(cell=[1, 1, 1], rotations=ops,
                     atommaps=[[0]] * len(ops))
    sym.analyze_displacements()
    forces = {}
    for k, ax in enumerate('xyz'):
        e = np.zeros((1, 3))
        e[0, k] = -1.0  # F = -u for unit stiffness
        forces[f'0{ax}+'] = e
        forces[f'0{ax}-'] = -e
    F_sym = sym.symmetrize_displaced_forces(list(forces), forces)
    for key in forces:
        assert np.allclose(F_sym[key], forces[key])


def test_symmetrize_is_idempotent():
    # Averaging over the retained contributions is a projector when they
    # form the full group (cubic): applying it twice equals once.
    ops = Symmetries.from_cell([1, 1, 1]).rotation_scc
    sym = Symmetries(cell=[1, 1, 1], rotations=ops,
                     atommaps=[[0]] * len(ops))
    sym.analyze_displacements()
    rng = np.random.default_rng(3)
    forces = {f'0{ax}{s}': rng.standard_normal((1, 3))
              for ax in 'xyz' for s in '+-'}
    once = sym.symmetrize_displaced_forces(list(forces), forces)
    twice = sym.symmetrize_displaced_forces(list(forces), once)
    for key in forces:
        assert np.allclose(once[key], twice[key], atol=1e-12)


def test_symmetrize_cubic_full_group_stays_on_axis():
    # Full cubic averaging of a 1-atom response leaves only the on-axis
    # component of an axis-aligned displacement.
    ops = Symmetries.from_cell([1, 1, 1]).rotation_scc
    sym = Symmetries(cell=[1, 1, 1], rotations=ops,
                     atommaps=[[0]] * len(ops))
    sym.analyze_displacements()
    rng = np.random.default_rng(11)
    base = rng.standard_normal((1, 3))
    forces = {f'0{ax}{s}': (1 if s == '+' else -1) * base
              for ax in 'xyz' for s in '+-'}
    F_sym = sym.symmetrize_displaced_forces(['0x+'], forces)['0x+']
    assert np.allclose(F_sym[0, 1:], 0, atol=1e-12)
