import numpy as np
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.mpi import world
from gpaw.new.brillouin import MonkhorstPackKPoints
from gpaw.new.symmetry import Symmetries
from gpaw.spinorbit import (basis_change, diagonalize, distribute_projections,
                            to_spinors)


def test_dist():
    layout = AtomArraysLayout([5, 13], world)
    l_aj = [[0, 0, 1], [0, 0, 1, 1, 2]]
    P_ani = layout.zeros(3)
    P_ksani = {(0, 0): P_ani}
    rank_ks = np.zeros((1, 1), int)
    sym = Symmetries(cell=np.eye(3), atommaps=np.arange(2).reshape((1, 2)))
    ibz = MonkhorstPackKPoints([1, 1, 1]).reduce(sym)
    P_Ksani, _, _ = distribute_projections(
        P_ksani, l_aj, rank_ks, ibz, world, 1, 3)
    P_amsi, eig_m = to_spinors(P_Ksani[0], np.ones((1, 2)))
    C_ss, s_vss = basis_change(theta=0.0, phi=0.0)
    dVL_avii = {a: np.ones((3,) + shape)
                for a, shape in enumerate(layout.shape_a)}
    diagonalize(P_amsi, eig_m, dVL_avii, s_vss, C_ss)


if __name__ == '__main__':
    test_dist()
