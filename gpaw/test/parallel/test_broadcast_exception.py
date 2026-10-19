import pytest

from gpaw.mpi import broadcast_exception, world


class MyError(RuntimeError):
    pass


def failonranks(failranks, comm):
    with broadcast_exception(comm):
        if comm.rank in failranks:
            raise MyError(f'{comm.rank}')

    return 42


def test_ok(comm):
    assert failonranks([], comm) == 42


@pytest.mark.skipif(world.size < 4, reason='world.size < 4')
def test_broadcast_exception(comm):
    with pytest.raises(MyError, match='1'):
        failonranks([1], comm)

    # The highest rank should be the one that broadcasts its error:
    with pytest.raises(MyError, match='3'):
        failonranks([1, 3], comm)
