import pytest

from gpaw.core import UGDesc
from gpaw.mpi import world
from gpaw.new.builder import create_communicators


@pytest.mark.ci
def test_fft_interpolation():
    a = UGDesc(cell=[1, 1, 1], size=(4, 4, 4)).zeros()
    b = UGDesc(cell=[1, 1, 1], size=(8, 8, 8)).zeros()
    a.data[2, 2, 2] = 1.0
    a.interpolate(out=b)
    assert (b.data[::2, ::2, ::2] == a.data).all()

    b.fft_restrict(out=a)
    assert a.integrate() == pytest.approx(b.integrate(), abs=1e-12)


def _serial_interpolation(size1=(4, 4, 4), size2=(8, 8, 8)):
    a = UGDesc(cell=[1, 2, 3], size=size1).empty()
    i, j, k = a.desc.xyz().transpose((3, 0, 1, 2))
    a.data[:] = (1.0 + 0.3 * i + 0.2 * j - 0.1 * k +
                 0.05 * i * j + 0.04 * j * k)

    b = UGDesc(cell=[1, 2, 3], size=size2).empty()
    a.interpolate(out=b)
    return a.data.copy(), b.data.copy()


def _serial_restriction(size1=(4, 4, 4), size2=(8, 8, 8)):
    fine = UGDesc(cell=[1, 2, 3], size=size2).empty()
    i, j, k = fine.desc.xyz().transpose((3, 0, 1, 2))
    fine.data[:] = (1.0 + 0.07 * i - 0.05 * j + 0.03 * k +
                    0.02 * i * k)

    coarse = UGDesc(cell=[1, 2, 3], size=size1).empty()
    fine.fft_restrict(out=coarse)
    return fine.data.copy(), coarse.data.copy()


def _set_domain_data(a_R, serial_data):
    start_c = a_R.desc.start_c
    end_c = a_R.desc.end_c
    a_R.data[:] = serial_data[
        start_c[0]:end_c[0],
        start_c[1]:end_c[1],
        start_c[2]:end_c[2]]


def _assert_local_data(a_R, serial_data):
    start_c = a_R.desc.start_c
    end_c = a_R.desc.end_c
    expected = serial_data[
        start_c[0]:end_c[0],
        start_c[1]:end_c[1],
        start_c[2]:end_c[2]]
    assert a_R.data == pytest.approx(expected, abs=1e-12)


@pytest.mark.ci
@pytest.mark.skipif(world.size == 1, reason='Only run in parallel')
@pytest.mark.parametrize('domain_size', [1, 2, 4])
def test_fft_interpolation_with_input_complementary_comm(domain_size):
    if world.size % domain_size:
        pytest.skip('world size must be divisible by domain size')

    comms = create_communicators(world,
                                 domain=domain_size,
                                 kpt=world.size // domain_size,
                                 band=1)
    domain_comm = comms['d']
    complementary_comm = comms['D']
    grid = UGDesc(cell=[1, 2, 3],
                  size=(4, 4, 4),
                  comm=domain_comm)
    fine_grid = grid.new(size=(8, 8, 8), comm=world)

    serial_data, expected = _serial_interpolation()
    a_R = grid.empty()
    _set_domain_data(a_R, serial_data)
    b_r = fine_grid.empty()

    a_R.interpolate(out=b_r, in_complementary_comm=complementary_comm)

    result = b_r.gather()
    if world.rank == 0:
        assert result is not None
        assert result.data == pytest.approx(expected, abs=1e-12)


@pytest.mark.ci
@pytest.mark.skipif(world.size == 1, reason='Only run in parallel')
@pytest.mark.parametrize('domain_size', [1, 2, 4])
def test_fft_interpolation_with_output_complementary_comm(domain_size):
    if world.size % domain_size:
        pytest.skip('world size must be divisible by domain size')

    comms = create_communicators(world,
                                 domain=domain_size,
                                 kpt=world.size // domain_size,
                                 band=1)
    domain_comm = comms['d']
    complementary_comm = comms['D']
    grid = UGDesc(cell=[1, 2, 3],
                  size=(4, 4, 4),
                  comm=world)
    fine_grid = grid.new(size=(8, 8, 8), comm=domain_comm)

    serial_data, expected = _serial_interpolation()
    a_R = grid.empty()
    _set_domain_data(a_R, serial_data)
    b_r = fine_grid.empty()

    a_R.interpolate(out=b_r, out_complementary_comm=complementary_comm)

    _assert_local_data(b_r, expected)


@pytest.mark.ci
@pytest.mark.skipif(world.size == 1, reason='Only run in parallel')
@pytest.mark.parametrize('domain_size', [1, 2, 4])
def test_fft_restrict_with_output_complementary_comm(domain_size):
    if world.size % domain_size:
        pytest.skip('world size must be divisible by domain size')

    comms = create_communicators(world,
                                 domain=domain_size,
                                 kpt=world.size // domain_size,
                                 band=1)
    domain_comm = comms['d']
    complementary_comm = comms['D']
    fine_grid = UGDesc(cell=[1, 2, 3],
                       size=(8, 8, 8),
                       comm=world)
    grid = fine_grid.new(size=(4, 4, 4), comm=domain_comm)

    fine_data, expected = _serial_restriction()
    a_r = fine_grid.empty()
    _set_domain_data(a_r, fine_data)
    b_R = grid.empty()

    a_r.fft_restrict(out=b_R, out_complementary_comm=complementary_comm)

    _assert_local_data(b_R, expected)


@pytest.mark.ci
@pytest.mark.skipif(world.size != 4, reason='Needs exactly four ranks')
def test_fft_interpolation_with_shifted_input_complementary_comm():
    domain_ranks = [0, 2] if world.rank % 2 == 0 else [1, 3]
    complementary_ranks = [0, 1] if world.rank < 2 else [2, 3]
    domain_comm = world.new_communicator(domain_ranks)
    complementary_comm = world.new_communicator(complementary_ranks)

    grid = UGDesc(cell=[1, 2, 3],
                  size=(4, 4, 4),
                  comm=domain_comm)
    fine_grid = grid.new(size=(8, 8, 8), comm=world)

    serial_data, expected = _serial_interpolation()
    a_R = grid.empty()
    _set_domain_data(a_R, serial_data)
    b_r = fine_grid.empty()

    a_R.interpolate(out=b_r, in_complementary_comm=complementary_comm)

    result = b_r.gather()
    if world.rank == 0:
        assert result is not None
        assert result.data == pytest.approx(expected, abs=1e-12)
