import pytest
from ase import Atoms

from gpaw import GPAW_NO_C_EXTENSION
from gpaw.dft import GPAW
from gpaw.mpi import world


@pytest.mark.gpu
@pytest.mark.parametrize(
    'num_domains',
    [
        pytest.param(1, id="domains1"),
        pytest.param(2, id="domains2", marks=pytest.mark.skipif(
            world.size < 2, reason="Needs >= 2 MPI tasks")),
    ])
def test_h2(num_domains: int):
    """Converges a simple system with GPU-LCAO"""
    if GPAW_NO_C_EXTENSION:
        pytest.skip('GPAW_NO_C_EXTENSION')

    # Leftover ranks are automatically used for band parallelization
    parallel = {'gpu': True, 'domain': num_domains}

    h2 = Atoms('H2',
               [[0, 0, 0],
                [0, 0, 0.75]],
               pbc=False)
    h2.center(vacuum=2.5)
    h2.calc = GPAW(
        legacy_gpaw=False,
        mode='lcao',
        basis='dzp',
        parallel=parallel,
        experimental={'pw_pot_calc': True})
    e = h2.get_potential_energy()
    assert e == pytest.approx(-6.514335)
