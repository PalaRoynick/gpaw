"""Tests for elph/supercell/_read_displacement_cache.

Covers
------
* PBC and non-PBC geometries — the latter exercises the
  ``zerobc`` boundary on the non-periodic axis.
* Cache layouts produced by both new GPAW (shape ``size_c - zerobc_c``)
  and a legacy GPAW format (full ``size_c`` with a leading zero plane on
  non-periodic axes).
"""
import ase.units as units
import numpy as np
import pytest
from ase.build import bulk
from ase.utils.filecache import MultiFileJSONCache

from gpaw import GPAW
from gpaw.elph import DisplacementRunner, Supercell
from gpaw.mpi import world


def _build_atoms(pbc: bool):
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    if pbc:
        return atoms, {'size': (2, 2, 2), 'gamma': False}
    atoms.pbc = (True, True, False)
    atoms.cell[2, 2] += 6.0
    atoms.center(axis=2)
    return atoms, {'size': (2, 2, 1), 'gamma': True}


def _to_legacy_format(Vt_sG, pbc_c):
    """Prepend a zero plane on each non-periodic axis (legacy shape)."""
    out = Vt_sG
    for axis, p in enumerate(pbc_c, start=1):  # +1: skip leading spin axis
        if not p:
            pad = np.zeros_like(np.take(out, [0], axis=axis))
            out = np.concatenate([pad, out], axis=axis)
    return out


@pytest.mark.elph
@pytest.mark.parametrize('pbc', [True, False])
@pytest.mark.parametrize('legacy_format', [False, True])
def test_gradient(in_tmp_dir, pbc, legacy_format):
    atoms, kpts = _build_atoms(pbc)
    assert len(atoms) == 2
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts=kpts,
                txt='elph_displacements.txt')
    atoms.calc = calc

    elph = DisplacementRunner(atoms=atoms, calc=atoms.calc,
                              supercell=(1, 1, 1), name='elph',
                              calculate_forces=False)
    elph.indices = []
    elph.run()
    # Barrier will be included in elph.run() in the future:
    # https://gitlab.com/ase/ase/-/merge_requests/2903
    world.barrier()

    Vt_sG = elph.cache['eq']['Vt_sG']
    dH_all_asp = elph.cache['eq']['dH_all_asp']
    pbc_c = np.array(atoms.get_pbc(), dtype=bool)

    # create displaced entries without calculation to save time
    V1expected = []
    for a in (0, 1):
        for i in range(3):
            V1expected.append(2 * 0.1 * (a + 1) * (i + 1) /
                              (2 * 0.01 / units.Bohr))
            for sign in [-1, 1]:
                disp = elph._disp(a, i, sign)
                with elph.cache.lock(disp.name) as handle:
                    if handle is None:
                        continue
                    Vfake = Vt_sG + sign * 0.1 * (a + 1) * (i + 1)
                    if legacy_format:
                        Vfake = _to_legacy_format(Vfake, pbc_c)
                    handle.save({'Vt_sG': Vfake, 'dH_all_asp': dH_all_asp})

    sc = Supercell(atoms)
    cache = MultiFileJSONCache('elph')
    grid = atoms.calc.dft.density.nt_sR.desc
    expected_shape = tuple(grid.size_c - grid.zerobc_c)

    V1t_xsG = []
    for a in (0, 1):
        for v in range(3):
            V1t_sG, _ = sc._read_displacement_cache(
                cache, a, v, grid, filter=None)
            assert V1t_sG.shape[-3:] == expected_shape
            V1t_xsG.append(V1t_sG)
    V1t_xsG = np.array(V1t_xsG)

    assert np.array(V1expected) == pytest.approx(V1t_xsG[:, 0, 0, 0, 0])
