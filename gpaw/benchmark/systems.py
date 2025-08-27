import numpy as np
from scipy.linalg import block_diag

graphene_bilayers = {
    244: [[[-5, -9], [4, -5]], [[-4, -9], [5, -4]]],
    364: [[[-6, -11], [5, -6]], [[-5, -11], [6, -5]]],
    508: [[[-13, -6], [-7, -13]], [[-13, -7], [-6, -13]]],
    676: [[[-7, 8], [-15, -7]], [[-8, 7], [-15, -8]]],
    868: [[[-17, -8], [-9, -17]], [[-17, -9], [-8, -17]]],
    1084: [[[-10, -19], [9, -10]], [[-9, -19], [10, -9]]],
    1324: [[[-10, 11], [-21, -10]], [[-11, 10], [-21, -11]]],
    1588: [[[-23, -11], [-12, -23]], [[-23, -12], [-11, -23]]],
    1876: [[[-13, -25], [12, -13]], [[-12, -25], [13, -12]]],
    2188: [[[27, 13], [14, 27]], [[27, 14], [13, 27]]],
    2524: [[[-15, -29], [14, -15]], [[-14, -29], [15, -14]]],
    2884: [[[15, -16], [31, 15]], [[16, -15], [31, 16]]],
    3268: [[[-16, 17], [-33, -16]], [[-17, 16], [-33, -17]]],
    3676: [[[18, 35], [-17, 18]], [[17, 35], [-18, 17]]],
    4564: [[[-39, -19], [-20, -39]], [[-39, -20], [-19, -39]]],
    5548: [[[21, -22], [43, 21]], [[22, -21], [43, 22]]],
    10444: [[[29, -30], [59, 29]], [[30, -29], [59, 30]]],
}


def get_bilayer(N):
    from gpaw.benchmark.generate_twisted import make_heterostructure
    from ase.build import graphene

    atoms = graphene(vacuum=5)
    transa_cc, transb_cc = graphene_bilayers[N]
    atoms = make_heterostructure(
        atoms,
        atoms,
        transa_cc=block_diag(transa_cc, [[1]]),
        transb_cc=block_diag(transb_cc, [[1]]),
        straina_vv=np.eye(3),
        interlayer_dist=3.35,
    )
    return atoms


def system_H2():
    from ase.build import molecule

    atoms = molecule("H2")
    atoms.center(vacuum=3)
    return atoms


def system_C60():
    from ase.build import molecule

    atoms = molecule("C60")
    atoms.center(vacuum=5)
    return atoms


def system_diamond():
    from ase.build import bulk

    atoms = bulk("C")
    return atoms


def system_MoS2_tube():
    from math import pi
    import numpy as np
    from ase.build import mx2

    # Create tube of MoS2:
    atoms = mx2("MoS2", size=(3, 2, 1))
    atoms.cell[1, 0] = 0
    atoms = atoms.repeat((1, 10, 1))
    p = atoms.positions
    p2 = p.copy()
    L = atoms.cell[1, 1]
    r0 = L / (2 * pi)
    angle = p[:, 1] / L * 2 * pi
    p2[:, 1] = (r0 + p[:, 2]) * np.cos(angle)
    p2[:, 2] = (r0 + p[:, 2]) * np.sin(angle)
    atoms.positions = p2
    atoms.cell = [atoms.cell[0, 0], 0, 0]
    atoms.center(vacuum=6, axis=[1, 2])
    atoms.pbc = True

    return atoms


def system_magbulk():
    from ase.build import bulk

    atoms = bulk("Fe") * 2
    atoms.set_initial_magnetic_moments([3] * len(atoms))
    return atoms


def system_metalslab():
    from ase.build import fcc111

    slab = fcc111("Al", size=(3, 4, 8), vacuum=6.0)
    return slab


systems = {
    "C60": system_C60,
    "diamond": system_diamond,
    "H2": system_H2,
    "MoS2_tube": system_MoS2_tube,
    "magbulk": system_magbulk,
    "metalslab": system_metalslab,
}

for N in graphene_bilayers:
    systems["C" + str(N)] = lambda: get_bilayer(N)
    assert N == len(get_bilayer(N)), (N, len(get_bilayer(N)))
systems["magic_graphene"] = systems["C10444"]


def parse_system(name):
    return systems[name]()
