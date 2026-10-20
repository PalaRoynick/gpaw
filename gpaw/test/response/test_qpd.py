import numpy as np

from gpaw.old.grid_descriptor import GridDescriptor
from gpaw.response.qpd import SingleQPWDescriptor, SingleZQPWDescriptor


def test_single_z_qpw_descriptor_is_subset_of_full_basis(mpi):
    gd = GridDescriptor((8, 8, 10), [6.0, 6.0, 8.0], comm=mpi.comm)
    ecut = 3.0
    q_c = [0.0, 0.0, 0.0]

    full_qpd = SingleQPWDescriptor.from_q(q_c, ecut, gd)
    z_qpd = SingleQPWDescriptor.from_q(
        q_c,
        {'class': SingleZQPWDescriptor,
         'kwargs': {'ecut_z': ecut}},
        gd)

    full_Gv = full_qpd.get_reciprocal_vectors(add_q=False)
    z_Gv = z_qpd.get_reciprocal_vectors(add_q=False)

    assert np.allclose(z_Gv[:, :2], 0.0)
    assert len(z_Gv) < len(full_Gv)

    z_in_full_G = []
    for G_v in z_Gv:
        d_G = np.linalg.norm(full_Gv - G_v, axis=1)
        G = np.argmin(d_G)
        assert d_G[G] < 1e-12
        z_in_full_G.append(G)

    z_in_full_G = np.array(z_in_full_G)
    assert np.allclose(full_Gv[z_in_full_G], z_Gv)
