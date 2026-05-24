import numpy as np
import pytest
from ase.build import bulk
from ase.utils.filecache import MultiFileJSONCache
from pathlib import Path

from gpaw.new.symmetry import create_symmetries_object
from gpaw.elph.symmetrize.atom_mapper import AtomMapper
from gpaw.elph.symmetrize.force import ForceSymmetrizer

TEST_DATA_DIR_NAME = Path(__file__).parent / 'data'
CACHE = MultiFileJSONCache(TEST_DATA_DIR_NAME)


@pytest.fixture
def atom_mapper():
    atoms = bulk('Si', 'diamond', cubic=False)
    symmetries = create_symmetries_object(atoms)
    return AtomMapper(symmetries)


@pytest.fixture
def forces():
    return {
        k: CACHE[k]['forces']
        for k in CACHE.keys()
        if 'eq' not in k and 'info' not in k
    }


def test_symmetrize_eq_force(atom_mapper):
    F_av = CACHE['eq']['forces']
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
