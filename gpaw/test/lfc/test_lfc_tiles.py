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
        F_av = basis.calculate_force_contribution(vt_G, rho_MM, q=-1)
        psit_nG = basis.gd.zeros(5)
        basis.lcao_to_grid(C_MM[:5].copy(), psit_nG, q=-1)
        results.append((V_MM, nt_G, F_av, psit_nG))
    assert basis._tiles() is not None

    for x0, x1 in zip(*results):
        assert x1 == pytest.approx(x0, abs=1e-12 * abs(x0).max())


def si2():
    # Small non-orthogonal cell: many images per tile (Bloch-summed tiles)
    atoms = bulk('Si', 'diamond', a=5.43)
    atoms.positions[0] += [0.05, -0.02, 0.03]
    return atoms


def h2o_periodic():
    # Large cell: few images per tile (per-volume tiles with phases)
    atoms = h2o()
    atoms.pbc = True
    return atoms


@pytest.mark.ci
@pytest.mark.parametrize('create_atoms, kpts', [(si2, (2, 1, 1)),
                                                (si2, (1, 2, 2)),
                                                (h2o_periodic, (2, 1, 1))])
def test_lfc_tiles_kpts(monkeypatch, create_atoms, kpts):
    """Tiled kernels with Bloch phases (complex density matrices)."""
    if not tiles_available():
        pytest.skip('GPAW compiled without BLAS')
    atoms = create_atoms()
    basis = Parameters(mode='lcao', basis='dzp', kpts=kpts
                       ).dft_component_builder(
        atoms, comm=serial_comm).create_basis_set()
    assert not basis.gamma
    nq = len(basis.phase_qW)
    rng = np.random.default_rng(8)
    vt_G = basis.gd.zeros()
    vt_G[:] = rng.random(vt_G.shape) - 0.5
    M = basis.Mmax
    C_MM = rng.random((M, M)) - 0.5 + 1j * (rng.random((M, M)) - 0.5)
    rho_MM = C_MM @ C_MM.conj().T  # Hermitian

    results = []
    for flag in ['0', '1']:
        monkeypatch.setenv('GPAW_LFC_TILES', flag)
        V_xMM = basis.calculate_potential_matrices(vt_G)
        nt_qG = []
        F_qav = []
        psit_qnG = []
        for q in range(nq):
            nt_G = basis.gd.zeros()
            basis.construct_density(rho_MM, nt_G, q)
            nt_qG.append(nt_G)
            F_qav.append(basis.calculate_force_contribution(
                vt_G, rho_MM.T.copy(), q))
            psit_nG = basis.gd.zeros(5, dtype=complex)
            basis.lcao_to_grid(C_MM[:5].copy(), psit_nG, q)
            psit_qnG.append(psit_nG)
        results.append((V_xMM, np.array(nt_qG), np.array(F_qav),
                        np.array(psit_qnG)))
    tiles = basis._tiles()
    assert tiles.per_volume
    used = tiles.nf_t > 0
    bloch = tiles.nf_t > 2 * tiles.nfM_t  # see use_bloch_sum() in C
    if create_atoms is si2:
        assert bloch[used].all()
    else:
        assert not bloch[used].all()

    for x0, x1 in zip(*results):
        assert x1 == pytest.approx(x0, abs=1e-12 * abs(x0).max())
