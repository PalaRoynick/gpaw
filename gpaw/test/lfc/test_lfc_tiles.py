import numpy as np
import pytest
from ase.build import bulk, molecule

from gpaw.dft import Parameters
from gpaw.lfc_tiles import tiles_available
from gpaw.mpi import serial_comm


def h2o():
    atoms = molecule('H2O', vacuum=2.5)  # s, p and d functions
    atoms.positions[0, 0] += 0.13  # break symmetry, odd grid offsets
    return atoms


def li_bulk():
    # Cutoff spheres larger than the cell: overlapping periodic images
    atoms = bulk('Li', 'bcc', a=3.51, cubic=True)
    atoms.positions[0] += 0.07
    return atoms


@pytest.mark.ci
@pytest.mark.parametrize('create_atoms', [h2o, li_bulk])
def test_lfc_tiles(monkeypatch, create_atoms):
    """Tiled LCAO kernels must reproduce the LFC kernels."""
    if not tiles_available():
        pytest.skip('GPAW compiled without BLAS')
    basis = Parameters(mode='lcao', basis='dzp', h=0.23, kpts=(1, 1, 1)
                       ).dft_component_builder(
        create_atoms(), comm=serial_comm).create_basis_set()
    assert basis.gamma and basis.dtype == float
    rng = np.random.default_rng(7)
    vt_G = basis.gd.zeros()
    vt_G[:] = rng.random(vt_G.shape) - 0.5
    C_MM = rng.random((basis.Mmax, basis.Mmax)) - 0.5
    rho_MM = C_MM @ C_MM.T

    results = []
    for flag in ['0', '1']:
        monkeypatch.setenv('GPAW_LFC_TILES', flag)
        V_MM = basis.calculate_potential_matrices(vt_G)[0]
        nt_G = basis.gd.zeros()
        basis.construct_density(rho_MM, nt_G, q=-1)
        results.append((V_MM, nt_G))
    assert basis._tiles() is not None

    for x0, x1 in zip(*results):
        assert x1 == pytest.approx(x0, abs=1e-12 * abs(x0).max())
