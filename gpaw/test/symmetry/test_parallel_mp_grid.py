import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from gpaw.bztools import (_parallel_grid_function,
                          optimal_monkhorst_pack_grid)
from gpaw.mpi import world


def test_distributed_grid_exception():
    def evaluate(grids):
        if world.rank == world.size - 1:
            raise ValueError('candidate failed')
        return np.zeros(len(grids), dtype=int)

    with pytest.raises(ValueError, match='candidate failed'):
        _parallel_grid_function(evaluate, np.ones((2, 3), int), world)
    world.barrier()


@pytest.mark.parametrize('system', ['orthogonal', 'silicon'])
@pytest.mark.parametrize('options', [
    {},
    {'minimize_ibz_points': True},
    {'is_symmetric_mp_grid': True, 'minimize_ibz_points': True},
    {'force_gamma': True, 'force_even': True,
     'contains_ibz_vertices': True, 'is_symmetric_mp_grid': True,
     'nmaxperdim': 6,
     'minimize_ibz_points': True},
    {'nmaxperdim': 0},
])
def test_parallel_optimal_mp_grid(system, options):
    atoms = (Atoms('H', cell=[6, 3, 8], pbc=True)
             if system == 'orthogonal' else bulk('Si', 'diamond', a=5.43))
    kwargs = dict(kptdensity=0.5, nmaxperdim=2, tolerance=2e-5)
    kwargs.update(options)
    serial = optimal_monkhorst_pack_grid(atoms, **kwargs)
    parallel = optimal_monkhorst_pack_grid(atoms, comm=world, **kwargs)
    np.testing.assert_array_equal(parallel['size'], serial['size'])
    assert parallel['gamma'] == serial['gamma']


def test_parallel_no_compliant_grid():
    atoms = bulk('Si', 'diamond', a=5.43)
    with pytest.raises(RuntimeError, match='contains_ibz_vertices_predicate'):
        optimal_monkhorst_pack_grid(
            atoms, kptdensity=0.5, nmaxperdim=0, force_even=True,
            force_gamma=True, contains_ibz_vertices=True, comm=world)
    world.barrier()
