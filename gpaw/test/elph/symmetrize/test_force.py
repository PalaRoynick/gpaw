import numpy as np
import pytest
from ase.build import bulk

from gpaw.elph.symmetrize.force import ForceSymmetrizer
from gpaw.test.elph.symmetrize.fixtures import create_cache


@pytest.fixture
def cache(tmp_path):
    return create_cache(tmp_path / 'data')


@pytest.fixture
def atoms():
    return bulk('Si', 'diamond', cubic=False)


@pytest.fixture
def forces(cache):
    return {
        k: cache[k]['forces']
        for k in cache.keys()
        if 'eq' not in k and 'info' not in k
    }


def test_symmetrize_eq_force(atoms, cache):
    F_av = cache['eq']['forces']
    F_av_sym = ForceSymmetrizer(atoms).symmetrize_equilibrium(F_av)

    assert F_av_sym.shape == F_av.shape
    assert np.allclose(F_av_sym, 0, atol=1e-8)


def test_symmetrize_displaced_force(atoms, forces):
    disp_key = '0x-'
    F_av_sym = ForceSymmetrizer(atoms).symmetrize_displaced(disp_key, forces)

    expected = np.array(
        [[0.312424743852, 0.0, 0.0],
         [-0.311996209487, 0.0, 0.0]]
    )
    assert F_av_sym.shape == forces[disp_key].shape
    assert np.allclose(F_av_sym, expected, atol=1e-8)


def test_symmetrize_displaced_force_stays_on_axis(atoms, forces):
    """Full-group averaging drives the off-axis force components to zero."""
    fs = ForceSymmetrizer(atoms)
    F_av_sym = fs.symmetrize_displaced('0x-', forces)
    assert np.allclose(F_av_sym[:, 1:], 0, atol=1e-8)


def test_symmetrize_restores_degeneracy(atoms, forces):
    """The on-site force-constant block is isotropic -> triply degenerate."""
    fs = ForceSymmetrizer(atoms)
    sym = fs.symmetrize_all_displaced(set(forces), forces)

    delta = 0.01
    cols = [-(sym[f'0{ax}+'][0] - sym[f'0{ax}-'][0]) / (2 * delta)
            for ax in 'xyz']
    phi = np.column_stack(cols)
    phi = 0.5 * (phi + phi.T)
    eig = np.linalg.eigvalsh(phi)
    assert np.allclose(eig, eig.mean(), atol=1e-6)


def test_symmetrize_displaced_forces(atoms, forces):
    disp_keys = {'0x-', '1x+'}
    fs = ForceSymmetrizer(atoms)
    result = fs.symmetrize_all_displaced(disp_keys, forces)

    expected_0xm = np.array(
        [[0.312424743852, 0.0, 0.0],
         [-0.311996209487, 0.0, 0.0]]
    )
    expected_1xp = np.array(
        [[0.311996294692, 0.0, 0.0],
         [-0.312424865258, 0.0, 0.0]]
    )
    assert np.allclose(result['0x-'], expected_0xm, atol=1e-8)
    assert np.allclose(result['1x+'], expected_1xp, atol=1e-8)


def test_symmetrize_cache_displaced_forces(atoms, cache, tmp_path):
    sym = ForceSymmetrizer(atoms)
    disp_keys = {k for k in cache if k not in ('eq', 'info')}
    forces_before = {k: cache[k]['forces'] for k in disp_keys}

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    expected = sym.symmetrize_all_displaced(disp_keys, forces_before)
    for key in disp_keys:
        assert np.array_equal(out[key]['forces'], expected[key]), \
            f'forces mismatch for {key}'


def test_symmetrize_cache_eq_forces(atoms, cache, tmp_path):
    sym = ForceSymmetrizer(atoms)
    F_eq_before = np.array(cache['eq']['forces'])

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    expected = sym.symmetrize_equilibrium(F_eq_before)
    result = out['eq']['forces']
    assert np.array_equal(result, expected)
    assert np.allclose(result, 0, atol=1e-8)


def test_symmetrize_cache_preserves_other_fields(atoms, cache, tmp_path):
    sym = ForceSymmetrizer(atoms)
    dh_before = {k: {a: np.array(arr)
                     for a, arr in cache[k]['dH_all_asp'].items()}
                 for k in cache if k != 'info'}

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    for key, dh_orig in dh_before.items():
        dh_out = out[key]['dH_all_asp']
        assert set(dh_out) == set(dh_orig)
        for a, arr in dh_orig.items():
            assert np.array_equal(dh_out[a], arr), \
                f'dH_all_asp changed for {key}'


def test_symmetrize_cache_info_copied(atoms, cache, tmp_path):
    sym = ForceSymmetrizer(atoms)
    info_before = dict(cache['info'])

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    assert dict(out['info']) == info_before


def test_symmetrize_cache_input_unchanged(atoms, cache, tmp_path):
    sym = ForceSymmetrizer(atoms)
    forces_before = {k: np.array(cache[k]['forces'])
                     for k in cache if k != 'info'}

    sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    for key, F_before in forces_before.items():
        assert np.array_equal(cache[key]['forces'], F_before), \
            f'input cache forces for {key!r} were modified'


_SC_DISP_KEYS = [
    f'{a}{ax}{s}'
    for a in (0, 1)
    for ax in ('x', 'y', 'z')
    for s in ('+', '-')
]


@pytest.fixture
def supercell_forces():
    rng = np.random.default_rng(0)
    return {k: rng.standard_normal((54, 3)) for k in _SC_DISP_KEYS}


def test_symmetrize_eq_force_supercell(atoms):
    rng = np.random.default_rng(42)
    F_av = rng.standard_normal((54, 3))

    F_sym = ForceSymmetrizer(atoms).symmetrize_equilibrium(F_av)

    assert F_sym.shape == (54, 3)
    assert np.allclose(F_sym, 0, atol=1e-10)


def test_symmetrize_displaced_supercell_forces_shape(atoms, supercell_forces):
    # Primitive symmetry (2 atoms) with forces on the 3x3x3 supercell
    # (54 atoms): the rotated response rows are accumulated in place, so the
    # output keeps the supercell shape.
    fs = ForceSymmetrizer(atoms)
    F_sym = fs.symmetrize_displaced('0x-', supercell_forces)
    assert F_sym.shape == (54, 3)


def test_symmetrize_all_displaced_supercell_forces(atoms, supercell_forces):
    disp_keys = {'0x-', '1x+'}
    result = ForceSymmetrizer(atoms).symmetrize_all_displaced(
        disp_keys, supercell_forces)

    assert set(result.keys()) == disp_keys
    for k in disp_keys:
        assert result[k].shape == (54, 3)


def test_symmetrize_displaced_supercell_restores_degeneracy(atoms,
                                                            supercell_forces):
    """On-site block from supercell forces is isotropic -> triply degenerate.

    The primitive point group is applied to supercell forces (no atom-row
    permutation).  Even for arbitrary input, the group average makes each
    displaced atom's on-site 3x3 force-constant block isotropic, which is what
    keeps the Gamma optical mode triply degenerate.
    """
    fs = ForceSymmetrizer(atoms)
    sym = fs.symmetrize_all_displaced(set(supercell_forces), supercell_forces)

    delta = 0.01
    for a in (0, 1):
        cols = [-(sym[f'{a}{ax}+'][a] - sym[f'{a}{ax}-'][a]) / (2 * delta)
                for ax in 'xyz']
        phi = np.column_stack(cols)
        phi = 0.5 * (phi + phi.T)
        eig = np.linalg.eigvalsh(phi)
        assert np.allclose(eig, eig.mean(), atol=1e-10)
