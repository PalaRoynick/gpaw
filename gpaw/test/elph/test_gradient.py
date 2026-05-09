"""Basic test of elph/supercell/_read_displacement_cache

Check that the per-displacement cache loader is capable of reading
the FD entries properly.
"""
import ase.units as units
import numpy as np
import pytest
from ase.build import bulk
from ase.utils.filecache import MultiFileJSONCache

from gpaw import GPAW
from gpaw.elph import DisplacementRunner, Supercell
from gpaw.mpi import world


@pytest.mark.elph
def test_gradient(in_tmp_dir):
    # 2 atoms with one 1 valence electron each
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    assert len(atoms) == 2
    # Note: usually we need to disable point group symmetry for displacements,
    # but we won't run actual displacements, so it saves time not to bother.
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts={'size': (2, 2, 2), 'gamma': False},
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
                    handle.save({'Vt_sG': Vfake, 'dH_all_asp': dH_all_asp})

    sc = Supercell(atoms)
    cache = MultiFileJSONCache('elph')
    grid = atoms.calc.dft.density.nt_sR.desc

    V1t_xsG = []
    for a in (0, 1):
        for v in range(3):
            V1t_sG, _ = sc._read_displacement_cache(cache, a, v, grid,
                                                   filter=None)
            V1t_xsG.append(V1t_sG)
    V1t_xsG = np.array(V1t_xsG)

    assert np.array(V1expected) == pytest.approx(V1t_xsG[:, 0, 0, 0, 0])
