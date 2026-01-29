import pytest
import numpy as np

from gpaw.cgpaw.gpu import cpp_test_ndarray, cpp_test_ndarray_typed
from gpaw.gpu import cupy as cp

@pytest.mark.parametrize("xp", [np])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_type_casts(xp, dtype: np.dtype):
    """"""
    a = xp.empty((1, 2), dtype=dtype)

    cpp_test_ndarray(a)
    if dtype != np.float32:
        with pytest.raises(TypeError):
            cpp_test_ndarray_typed(a)
    else:
        cpp_test_ndarray_typed(a)
