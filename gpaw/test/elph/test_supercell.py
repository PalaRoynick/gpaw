"""Basic test of elph/supercell/

"""
import numpy as np
import pytest

from gpaw.elph import Supercell
from gpaw.mpi import world

SUPERCELL = (2, 1, 1)


g00 = np.array(
    [[[[2.54075420e-08, 1.36509302e-08, 1.55902055e-02, 6.68467085e-03],
       [1.36509302e-08, 7.80434310e-09, 4.92696165e-03, 2.17286796e-03],
       [1.55902055e-02, 4.92696165e-03, 4.13169693e-02, 1.94515015e-02],
       [6.68467085e-03, 2.17286796e-03, 1.94515015e-02, 9.74567004e-03]],
      [[3.66048080e-02, 1.46628274e-02, 1.55901883e-02, 6.68466538e-03],
       [1.34728203e-02, 5.32696910e-03, 4.92695352e-03, 2.17286534e-03],
       [1.65245564e-02, 5.90874979e-03, -1.78388956e-08, -3.36865383e-04],
       [7.15892262e-03, 2.69239734e-03, 3.36852348e-04, -2.51035479e-09]]],
     [[[3.66048080e-02, 1.34728203e-02, 1.65245564e-02, 7.15892262e-03],
       [1.46628274e-02, 5.32696910e-03, 5.90874979e-03, 2.69239734e-03],
       [1.55901883e-02, 4.92695352e-03, -1.78388956e-08, 3.36852348e-04],
       [6.68466538e-03, 2.17286534e-03, -3.36865383e-04, -2.51035479e-09]],
      [[-2.54036782e-08, -1.36495323e-08, -1.65245736e-02, -7.15892808e-03],
       [-1.36495323e-08, -7.80413427e-09, -5.90875792e-03, -2.69239995e-03],
       [-1.65245736e-02, -5.90875792e-03, -4.13169693e-02, -1.94515015e-02],
       [-7.15892808e-03, -2.69239995e-03, -1.94515015e-02, -9.74567004e-03]]]])


# 5 kpoints for parallelisation; 4 basis functions
@pytest.mark.slow
@pytest.mark.skipif(world.size > 4,
                    reason='world.size > 4')
@pytest.mark.elph
def test_supercell(session_tmp_path, supercell_cache):
    # Generate supercell_cache
    supercell_cache

    # read supercell matrix
    g_xsNNMM, basis_info = Supercell.load_supercell_matrix()
    assert g_xsNNMM.shape == (6, 1, 2, 2, 4, 4)
    print(g_xsNNMM[0, 0])
    assert g_xsNNMM[0, 0] == pytest.approx(g00, rel=1e-2, abs=1e-4)
