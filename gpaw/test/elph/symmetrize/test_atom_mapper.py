"""
Tests for AtomMapper.disp_map_sd.

The mock Symmetries object uses a 4-atom H4 square-lattice system with four
hand-crafted operations so that expected values can be verified analytically.

Atom positions (scaled):
    0: [0.0, 0.0, 0.0]
    1: [0.5, 0.0, 0.0]
    2: [0.5, 0.5, 0.0]
    3: [0.0, 0.5, 0.0]

Operations (row-vector convention, x' = x @ R):
    s=0  identity                  atommap [0,1,2,3]
    s=1  180° rotation about z     atommap [0,1,2,3]
    s=2  reflection x ↔ y         atommap [0,3,2,1]
    s=3  reflection x ↔ -y        atommap [0,3,2,1]
"""

import types
import numpy as np
import pytest

from gpaw.elph.symmetrize.atom_mapper import AtomMapper, disp_key_to_index


@pytest.fixture
def mock_symmetries():
    rotation_scc = np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],    # s=0 identity
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # s=1 180° z
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],    # s=2 reflect x=y
        [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],  # s=3 reflect x=-y
    ], dtype=int)

    atommap_sa = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 3, 2, 1],
        [0, 3, 2, 1],
    ], dtype=int)

    return types.SimpleNamespace(rotation_scc=rotation_scc, atommap_sa=atommap_sa)


@pytest.fixture
def mapper(mock_symmetries):
    return AtomMapper(mock_symmetries)


def test_disp_map_sd_shape(mapper, mock_symmetries):
    n_ops, n_atoms = mock_symmetries.atommap_sa.shape
    assert mapper.disp_map_sd.shape == (n_ops, n_atoms * 6)


def test_identity_op_maps_to_self(mapper):
    """Identity operation must map every displacement to itself."""
    row = mapper.disp_map_sd[0]
    assert np.array_equal(row, np.arange(len(row)))


def test_no_sentinel_values(mapper):
    """All entries should have valid images for these purely symmorphic ops."""
    assert (mapper.disp_map_sd != -1).all()


def test_bijective_per_op(mapper, mock_symmetries):
    """Each row of disp_map_sd must be a permutation (no duplicates, no gaps)."""
    n_ops, n_atoms = mock_symmetries.atommap_sa.shape
    n_disps = n_atoms * 6
    for s in range(n_ops):
        row = mapper.disp_map_sd[s]
        assert sorted(row) == list(range(n_disps)), \
            f'op {s} is not a permutation: {row}'


# R2 (reflect x=y) maps source "3y+" → target "1x+".
# R3 (reflect x=-y) maps source "3y-" → target "1x+" (sign flips under R3).
def test_spot_check_1xp_sources(mapper):
    d_tgt = disp_key_to_index('1x+')
    assert mapper.disp_map_sd[2, d_tgt] == disp_key_to_index('3y+')
    assert mapper.disp_map_sd[3, d_tgt] == disp_key_to_index('3y-')


# R2 maps source "1x+" → target "3y+".
# R3 maps source "1x-" → target "3y+" (sign flips under R3).
def test_spot_check_3yp_sources(mapper):
    d_tgt = disp_key_to_index('3y+')
    assert mapper.disp_map_sd[2, d_tgt] == disp_key_to_index('1x+')
    assert mapper.disp_map_sd[3, d_tgt] == disp_key_to_index('1x-')


# All four ops map "0z+" to itself (z-axis is invariant under all ops here).
def test_spot_check_0zp_self(mapper):
    d = disp_key_to_index('0z+')
    assert (mapper.disp_map_sd[:, d] == d).all()


# R2 and R3 map "3z-" → "1z-" (atom swap, z direction preserved).
def test_spot_check_1zm_from_3zm(mapper):
    d_tgt = disp_key_to_index('1z-')
    d_src = disp_key_to_index('3z-')
    assert mapper.disp_map_sd[2, d_tgt] == d_src
    assert mapper.disp_map_sd[3, d_tgt] == d_src


# s=1 (180° z) flips both x and y signs; atom 0 maps to itself.
# "0x+" under s=1: [1,0,0] @ R1 = [-1,0,0] → target "0x-", source "0x+".
def test_spot_check_180z_flips_sign(mapper):
    d_tgt = disp_key_to_index('0x-')
    d_src = disp_key_to_index('0x+')
    assert mapper.disp_map_sd[1, d_tgt] == d_src
