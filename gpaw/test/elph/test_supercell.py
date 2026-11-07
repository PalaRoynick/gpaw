"""Basic test of elph/supercell/

"""
import numpy as np
import pytest

from gpaw.elph import Supercell
from gpaw.mpi import world

SUPERCELL = (2, 1, 1)

# Centered-atom pseudo-Hamiltonian derivative, after fixing projector mapping.
# Independently checked by test_paw.py against a frozen-operator derivative.
g00 = np.array([[[[1.24685685e-08, 6.49166500e-03],
                  [6.49166500e-03, 2.36718512e-02]],
                 [[1.75091078e-02, 6.49165787e-03],
                  [7.63447985e-03, -8.60311860e-09]]],
                [[[1.75091078e-02, 7.63447985e-03],
                  [6.49165787e-03, -8.60311860e-09]],
                 [[-1.18115568e-08, -7.63448697e-03],
                  [-7.63448697e-03, -2.36718518e-02]]]])


@pytest.mark.slow
@pytest.mark.skipif(world.size > 2,
                    reason='world.size > 2')
@pytest.mark.elph
def test_supercell(session_tmp_path, supercell_cache, comm):
    # Generate supercell_cache
    supercell_cache

    # read supercell matrix
    g_xsNNMM, basis_info = Supercell.load_supercell_matrix()
    assert g_xsNNMM.shape == (6, 1, 2, 2, 2, 2)
    print(g_xsNNMM[0, 0])
    assert g_xsNNMM[0, 0] == pytest.approx(g00, rel=1e-2, abs=1e-4)
