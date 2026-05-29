import numpy as np
import pytest
from ase import Atoms

from gpaw.dft import Parameters


def build_density(comm, mode: str, parallel: dict):
    atoms = Atoms('H2',
                  positions=[(1.35, 1.5, 1.5), (1.65, 1.5, 1.5)],
                  cell=[3.0, 3.0, 3.0],
                  pbc=True)

    modeobj = {'name': 'fd'} if mode == 'fd' else {'name': 'pw', 'ecut': 80}
    params = Parameters(mode=modeobj,
                        h=0.5,
                        nbands=2,
                        kpts={'size': [4, 1, 1], 'gamma': True},
                        symmetry='off',
                        parallel=parallel)
    builder = params.dft_component_builder(atoms, log=None, comm=comm)
    basis = builder.create_basis_set()
    density = builder.density_from_superposition(basis)
    return builder, density


def redist_density(density, source_builder, target_builder):
    """Move density between two GPAW k/b/d parallel layouts.

    ``comm1`` and ``comm2`` are the complements of the source and target
    domain communicators.  In GPAW communicator names this is ``D``, the
    combined k-point/band communicator for ranks that share a domain slice.
    """
    return density.redist(target_builder.grid,
                          target_builder.interpolation_desc,
                          target_builder.atomdist,
                          source_builder.communicators['D'],
                          target_builder.communicators['D'])


def snapshot(density):
    nt_sR = density.nt_sR.gather(broadcast=True)
    D_asii = density.D_asii.gather(broadcast=True, copy=True)
    nct_R = density.nct_R.gather(broadcast=True)
    tauct_R = density.tauct_R.gather(broadcast=True)
    assert nt_sR is not None
    assert D_asii is not None
    assert nct_R is not None
    assert tauct_R is not None
    taut_sR = None
    if density.taut_sR is not None:
        taut_sR = density.taut_sR.gather(broadcast=True)
        assert taut_sR is not None
        taut_sR = taut_sR.data.copy()
    return (nt_sR.data.copy(),
            taut_sR,
            D_asii.data.copy(),
            nct_R.data.copy(),
            tauct_R.data.copy())


def assert_same_density(density, reference):
    data = snapshot(density)
    for actual, expected in zip(data, reference):
        if actual is None:
            assert expected is None
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.ci
@pytest.mark.parametrize('mode', ['fd', 'pw'])
def test_density_redist_between_parallel_layouts(comm, mode):
    if comm.size != 4:
        pytest.skip('Test needs exactly 4 MPI ranks')

    source_builder, source_density = build_density(
        comm, mode, {'domain': 2, 'kpt': 2, 'band': 1})
    target_builders = [
        build_density(comm, mode, {'domain': 4, 'kpt': 1, 'band': 1})[0],
        build_density(comm, mode, {'domain': 1, 'kpt': 1, 'band': 4})[0],
        build_density(comm, mode, {'domain': 1, 'kpt': 4, 'band': 1})[0],
        build_density(comm, mode, {'domain': 2, 'kpt': 1, 'band': 2})[0]]

    assert source_builder.grid.comm.size == 2
    assert source_builder.communicators['D'].size == 2
    assert [builder.grid.comm.size for builder in target_builders] == [
        4, 1, 1, 2]
    assert [builder.communicators['D'].size
            for builder in target_builders] == [1, 4, 4, 2]

    source_data = snapshot(source_density)

    for target_builder in target_builders:
        target_density = redist_density(source_density,
                                        source_builder,
                                        target_builder)
        assert target_density.grid.comm.size == target_builder.grid.comm.size
        assert (target_density.D_asii.layout.atomdist.comm.size ==
                target_builder.atomdist.comm.size)
        assert_same_density(target_density, source_data)

        roundtrip_density = redist_density(target_density,
                                           target_builder,
                                           source_builder)
        assert (roundtrip_density.grid.comm.size ==
                source_builder.grid.comm.size)
        assert_same_density(roundtrip_density, source_data)
