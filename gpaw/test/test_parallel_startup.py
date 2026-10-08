import os
import pytest
import subprocess
import sys


@pytest.fixture(autouse=True)
def require_no_mpi(comm):
    if comm.backend != 'serial':
        # (This condition should work in the test suite but not necessarily
        # in other contexts)
        pytest.skip('This test cannot run from an MPI process')


def test_startup_serial():
    output = subprocess.check_output(
        [sys.executable, __file__], encoding='utf-8')
    assert output.strip() == 'serial 1'


@pytest.mark.parametrize('backend', [None, 'cgpaw', 'mpi4py'])
def test_startup_mpiexec(backend):
    if backend == 'mpi4py':
        pytest.importorskip('mpi4py')
    elif backend == 'cgpaw':
        from gpaw.cgpaw import have_mpi

        if not have_mpi:
            pytest.skip('no cgpaw mpi')

    mpiexec = os.environ.get('GPAW_TEST_MPIEXEC_COMMAND')
    if mpiexec is None:
        pytest.skip('Cannot test parallel startup without envvar.  '
                    'Try setting $GPAW_TEST_MPIEXEC_COMMAND=mpiexec '
                    'or similar.')

    env = dict(os.environ)
    if backend is not None:
        env['GPAW_MPI_BACKEND'] = backend

    commsize = '2'

    output = subprocess.check_output(
        [*mpiexec.split(), '-n', commsize, 'gpaw', 'python', __file__],
        encoding='utf-8', env=env)
    actual_backend, actual_commsize = output.split()

    if backend is not None:
        assert backend == actual_backend
    assert commsize == actual_commsize


def main():
    from gpaw.mpi import world

    assert world.sum_scalar(1) == world.size
    if world.rank == 0:
        print(world.backend, world.size)


if __name__ == '__main__':
    main()
