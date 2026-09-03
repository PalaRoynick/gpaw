import numpy as np
import pytest
from ase.spectrum.band_structure import BandStructure

from gpaw import GPAW, mpi
from gpaw.utilities.bandstructure import (create_bandstructure,
                                          diagonalize_soc_lcao,
                                          plot_soc_bandstructures)


def test_diagonalize_spinpolarized_lcao_soc_hamiltonian():
    H_skMM = np.array(
        [
            [[[1.0, 0.0], [0.0, 2.0]]],
            [[[3.0, 0.0], [0.0, 4.0]]],
        ],
        dtype=complex,
    )
    S_kMM = np.eye(2)[None]
    V_kvMM = np.zeros((1, 3, 2, 2), complex)

    energies_kn, sz_kn = diagonalize_soc_lcao(
        H_skMM, S_kMM, V_kvMM
    )

    assert energies_kn[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert sz_kn[0] == pytest.approx([1.0, 1.0, -1.0, -1.0])


def test_soc_mixes_spin_channels():
    H_skMM = np.zeros((1, 1, 1, 1), complex)
    S_kMM = np.ones((1, 1, 1), complex)
    V_kvMM = np.zeros((1, 3, 1, 1), complex)
    V_kvMM[0, 0, 0, 0] = 0.25

    energies_kn, sz_kn = diagonalize_soc_lcao(
        H_skMM, S_kMM, V_kvMM
    )

    assert energies_kn[0] == pytest.approx([-0.25, 0.25])
    assert sz_kn[0] == pytest.approx([0.0, 0.0], abs=1e-14)


def test_rank0_lcao_interpolation_and_plot(gpw_files, tmp_path):
    """Exercise rank-zero interpolation from a serial calculator under MPI."""
    calc = None
    path = None
    if mpi.world.rank == 0:
        calc = GPAW(gpw_files['diamond_lcao'], txt=None,
                    communicator=mpi.serial_comm)
        path = calc.atoms.cell.bandpath('GX', npoints=3)

    bs = create_bandstructure(calc, path, world=mpi.world)
    if mpi.world.rank == 0:
        assert isinstance(bs, BandStructure)
        sz_kn = np.zeros_like(bs.energies[0])
        plot_input = [('SOC', bs, sz_kn)]
    else:
        assert bs is None
        plot_input = []

    filename = tmp_path / 'soc.png'
    result = plot_soc_bandstructures(
        plot_input, filename, world=mpi.world
    )
    if mpi.world.rank == 0:
        assert result is not None
        assert filename.is_file()
    else:
        assert result is None
