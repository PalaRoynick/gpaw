from gpaw.soc import distribute_projections
from gpaw.new.brillouin import MonkhorstPackKPoints
from gpaw.new.symmetry import Symmetries
from gpaw.core.atom_arrays import AtomArraysLayout
import numpy as np
from gpaw.mpi import world


def test_dist():
    layout = AtomArraysLayout([5, 13])
    l_aj = [[0, 0, 1], [0, 0, 1, 1, 2]]
    P_ani = layout.zeros(3)
    P_ksani = {(0, 0): P_ani}
    rank_ks = np.zeros((1, 1), int)
    sym = Symmetries(cell=np.eye(3), atommaps=np.arange(2).reshape((1, 2)))
    ibz = MonkhorstPackKPoints([1, 1, 1]).reduce(sym)
    P = distribute_projections(P_ksani, l_aj, rank_ks, ibz, world, 1, 3)
    print(P)
