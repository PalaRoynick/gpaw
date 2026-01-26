from gpaw.cgpaw.gpu import Spline32, Spline64
from gpaw.spline import Spline  # legacy spline class

import pytest
import numpy as np

# can't use legacy splines with 32bit floats :(

#@pytest.fixture(scope="module", params=[np.float32, np.float64])
@pytest.fixture(scope="module", params=[np.float64])
def fixt_random_data(request) -> np.ndarray:
    """Tests will fit an interpolating spline to this data."""
    rng = np.random.default_rng(99)
    out = rng.random((6), dtype=request.param)
    # Last value must be 0.0 (value at the cutoff)
    out[-1] = 0.0
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legacy_conversion(dtype: np.dtype):
    """Test that we can correctly convert from legacy splines.
    """

    rng = np.random.default_rng(99)

    # The legacy spline class only works with 64bit floats
    data = rng.random((25), dtype=np.float64)

    x_max = 42.2
    legacy_spline = Spline.from_data(0, x_max, data)

    if dtype == np.float32:
        spline = Spline32.from_legacy_spline(legacy_spline.spline)
    else:
        spline = Spline64.from_legacy_spline(legacy_spline.spline)

    assert spline.get_num_bins() == legacy_spline.get_npoints() - 1
    assert spline.get_x_start() == 0.0
    if dtype == np.float64:
        assert spline.get_x_end() == legacy_spline.get_cutoff()
    else:
        assert spline.get_x_end() == pytest.approx(legacy_spline.get_cutoff())

    # Test at a few points, including some that go beyond the cutoff
    x = rng.uniform(0.0, 1.2*x_max, (15))
    y = spline.evaluate(x)
    y_old = legacy_spline.map(x)

    rtol = 1e-12 if dtype == np.float64 else 1e-6
    atol = 1e-6
    np.testing.assert_allclose(y, y_old, rtol=rtol, atol=atol)


def test_radial_spline(fixt_random_data: np.ndarray):
    """"""

    data = fixt_random_data
    dtype = data.dtype
    r_start = 0.0
    r_end = 5.8

    if dtype == np.float32:
        spline = Spline32(r_end, data)
    else:
        spline = Spline64(r_end, data)

    rtol = 1e-12 if dtype == np.float64 else 1e-8
    atol = 1e-12

    # Check that values at the breakpoints match the input values.
    # ie. did we choose breakpoints correctly
    breakpoints = spline.get_breakpoints()
    assert breakpoints[0] == pytest.approx(r_start)
    assert breakpoints[-1] == pytest.approx(r_end)
    np.testing.assert_allclose(spline.evaluate(breakpoints), data, atol=atol)

    # Test out of bounds evaluation (should give 0)
    r = np.asarray([r_end, r_end + 0.1])
    np.testing.assert_allclose(spline.evaluate(r),
                               np.zeros_like(r))

    legacy_spline = Spline.from_data(0, r_end, fixt_random_data)

    # test at random values of r
    rng = np.random.default_rng(99)
    r = rng.uniform(r_start, r_end, (8)).view(dtype)

    s_r = spline.evaluate(r)
    s_r_legacy = legacy_spline.map(r).view(dtype)

    np.testing.assert_allclose(s_r, s_r_legacy, rtol=rtol, atol=atol)


    from scipy.interpolate import CubicSpline
    scipy_spline = CubicSpline(breakpoints, data, bc_type="clamped")
    np.testing.assert_allclose(scipy_spline(breakpoints), data, atol=atol)

    np.testing.assert_allclose(scipy_spline(r), spline.evaluate(r),
                               rtol=rtol, atol=atol)
