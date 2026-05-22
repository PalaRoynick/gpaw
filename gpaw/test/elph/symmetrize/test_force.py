import shutil

import numpy as np
import pytest
from ase.build import bulk
from ase.utils.filecache import MultiFileJSONCache
from pathlib import Path

from gpaw.new.symmetry import create_symmetries_object
from gpaw.elph.symmetrize.atom_mapper import AtomMapper
from gpaw.elph.symmetrize.force import ForceSymmetrizer

TEST_DATA_DIR_NAME = Path(__file__).parent / 'data'


@pytest.fixture
def cache(tmp_path):
    shutil.copytree(TEST_DATA_DIR_NAME, tmp_path / 'data')
    return MultiFileJSONCache(tmp_path / 'data')


@pytest.fixture
def atom_mapper():
    atoms = bulk('Si', 'diamond', cubic=False)
    symmetries = create_symmetries_object(atoms)
    return AtomMapper(symmetries)


@pytest.fixture
def forces(cache):
    return {
        k: cache[k]['forces']
        for k in cache.keys()
        if 'eq' not in k and 'info' not in k
    }


def test_symmetrize_eq_force(atom_mapper, cache):
    F_av = cache['eq']['forces']
    F_av_sym = ForceSymmetrizer(atom_mapper).symmetrize_equilibrium(F_av)

    assert F_av_sym.shape == F_av.shape
    assert np.allclose(F_av_sym, 0, atol=1e-8)


def test_symmetrize_displaced_force(atom_mapper, forces):
    disp_key = '0x-'
    F_av_sym = ForceSymmetrizer(atom_mapper).symmetrize_displaced(disp_key, forces)

    expected = np.array(
        [[0.31060372, 0.00179978, 0.00179978],
         [-0.31018268, -0.00180007, -0.00180007]]
    )
    assert F_av_sym.shape == forces[disp_key].shape
    assert np.allclose(F_av_sym, expected, atol=0)


def test_symmetrize_displaced_forces(atom_mapper, forces):
    disp_keys = {'0x-', '1x+'}
    result = ForceSymmetrizer(atom_mapper).symmetrize_all_displaced(disp_keys, forces)

    expected_0xm = np.array(
        [[0.31060372, 0.00179978, 0.00179978],
         [-0.31018268, -0.00180007, -0.00180007]]
    )
    expected_1xp = np.array(
        [[0.31018333, 0.00179903, 0.00179903],
         [-0.31060429, -0.00179872, -0.00179872]]
    )
    assert np.allclose(result['0x-'], expected_0xm, atol=0)
    assert np.allclose(result['1x+'], expected_1xp, atol=0)

def test_symmetrize_cache_displaced_forces(atom_mapper, cache, tmp_path):
    sym = ForceSymmetrizer(atom_mapper)
    disp_keys = {k for k in cache if k not in ('eq', 'info')}
    forces_before = {k: cache[k]['forces'] for k in disp_keys}

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    expected = sym.symmetrize_all_displaced(disp_keys, forces_before)
    for key in disp_keys:
        assert np.array_equal(out[key]['forces'], expected[key]), \
            f'forces mismatch for {key}'


def test_symmetrize_cache_eq_forces(atom_mapper, cache, tmp_path):
    sym = ForceSymmetrizer(atom_mapper)
    F_eq_before = np.array(cache['eq']['forces'])

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    expected = sym.symmetrize_equilibrium(F_eq_before)
    result = out['eq']['forces']
    assert np.array_equal(result, expected)
    assert np.allclose(result, 0, atol=1e-8)


def test_symmetrize_cache_preserves_other_fields(atom_mapper, cache, tmp_path):
    sym = ForceSymmetrizer(atom_mapper)
    vt_before = {k: np.array(cache[k]['Vt_sG'])
                 for k in cache if k != 'info'}

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    for key, vt_orig in vt_before.items():
        assert np.array_equal(out[key]['Vt_sG'], vt_orig), \
            f'Vt_sG changed for {key}'


def test_symmetrize_cache_info_copied(atom_mapper, cache, tmp_path):
    sym = ForceSymmetrizer(atom_mapper)
    info_before = dict(cache['info'])

    out = sym.symmetrize_cache(cache, str(tmp_path / 'out'))

    assert dict(out['info']) == info_before


def test_symmetrize_cache_input_unchanged(atom_mapper, cache, tmp_path):
    sym = ForceSymmetrizer(atom_mapper)
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


def test_symmetrize_eq_force_supercell(atom_mapper):
    rng = np.random.default_rng(42)
    F_av = rng.standard_normal((54, 3))

    F_sym = ForceSymmetrizer(atom_mapper).symmetrize_equilibrium(F_av)

    assert F_sym.shape == (54, 3)
    assert np.allclose(F_sym, 0, atol=1e-10)


def test_symmetrize_displaced_supercell_forces_shape(atom_mapper, supercell_forces):
    F_sym = ForceSymmetrizer(atom_mapper).symmetrize_displaced('0x-', supercell_forces)
    assert F_sym.shape == (54, 3)


def test_symmetrize_all_displaced_supercell_forces(atom_mapper, supercell_forces):
    disp_keys = {'0x-', '1x+'}
    result = ForceSymmetrizer(atom_mapper).symmetrize_all_displaced(disp_keys, supercell_forces)

    assert set(result.keys()) == disp_keys
    for k in disp_keys:
        assert result[k].shape == (54, 3)
