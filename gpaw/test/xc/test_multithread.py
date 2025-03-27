from time import time

from gpaw.core import UGDesc
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.mpi import world
from gpaw.new.density import Density
from gpaw.new.external_potential import ExternalPotential
from gpaw.new.pot_calc import calculate_non_local_potential
from gpaw.new.xc import create_functional
from gpaw.setup import create_setup


def run():
    natoms = 30
    setups = [create_setup('Pt', lmax=2)] * natoms
    grid = UGDesc.from_cell_and_grid_spacing(cell=[1, 1, 1], grid_spacing=0.1)
    xc = create_functional('PBE', grid)
    nt_sR = grid.empty(1)
    D_asii = AtomArraysLayout([(18, 18)] * natoms).zeros(1)
    for D_sii in D_asii.values():
        D_sii[0, 0, 0] = 2.0
    Q_aL = AtomArraysLayout([(9,)] * natoms).zeros()
    density = Density.from_data_and_setups(
        nt_sR, None, D_asii, 0.0, setups, None, None)
    t1 = time()
    calculate_non_local_potential(
        setups, density, xc, ExternalPotential(),
        Q_aL,
        soc=False,
        kpt_band_comm=world)
    t2 = time()
    print((t2 - t1) / natoms)


if __name__ == '__main__':
    run()
