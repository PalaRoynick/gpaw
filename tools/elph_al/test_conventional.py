"""Cheap independent checks for conventional-cell geometry and phonons."""
import itertools

import numpy as np
from ase.utils.filecache import MultiFileJSONCache

from conventional import geometry, unfolded_kpoints, ConventionalPhonons


def test_reciprocal_unfolding():
    for n, k in [(2, 6), (3, 4), (4, 3)]:
        primitive, atoms, matrix, offset, R = geometry(4.04, n)
        assert round(abs(np.linalg.det(matrix))) == len(atoms)
        assert np.allclose(R[offset], 0)
        points = unfolded_kpoints(matrix, k)
        assert points.shape == (6912, 3)
        # Every unfolded point folds back onto the input Gamma mesh.
        folded = points @ matrix.T * k
        assert np.allclose(folded, np.rint(folded), atol=1e-8)
        assert np.allclose(atoms.cell.array, matrix @ primitive.cell.array)


def test_conventional_force_constants(tmp_path):
    a = 4.04
    primitive, atoms, matrix, center, R = geometry(a, 2)
    vectors = R @ primitive.cell.array
    spring = np.zeros((len(atoms), 3, 3))
    nearest = abs(np.linalg.norm(vectors, axis=1) - a / np.sqrt(2)) < 1e-8
    assert nearest.sum() == 12
    for i in np.flatnonzero(nearest):
        direction = vectors[i] / np.linalg.norm(vectors[i])
        spring[i] = -np.outer(direction, direction)
    spring[center] = -spring.sum(axis=0)
    cache = MultiFileJSONCache(tmp_path / 'fd')
    delta = .01
    cache['info'] = dict(delta=delta)
    for v, label in enumerate('xyz'):
        cache[f'0{label}-'] = dict(forces=delta * spring[:, v, :])
        cache[f'0{label}+'] = dict(forces=-delta * spring[:, v, :])
    ph = ConventionalPhonons(cache, a, 2)
    assert np.allclose(ph.dynamical_matrix(np.zeros(3)), 0, atol=1e-14)
    for q in itertools.product([.1, .25, .5], repeat=3):
        q = np.array(q)
        exact = np.einsum('n,nij->ij', np.exp(-2j * np.pi * R @ q),
                          spring) / primitive.get_masses()[0]
        assert np.allclose(ph.dynamical_matrix(q), exact, atol=1e-14)
        assert np.linalg.eigvalsh(exact).min() >= -1e-14
