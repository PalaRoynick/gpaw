import pytest
import numpy as np
from typing import cast, Callable

from gpaw.cgpaw.gpu import arraytests
from gpaw.gpu import cupy as cp, cupy_is_fake


def test_non_array_input():
    """"""
    a = np.zeros((1,1))

    not_arrays = [42, [1., 2.], [[42.]]]
    for not_array in not_arrays:
        with pytest.raises(TypeError):
           arraytests.accept_any(not_array, a)


@pytest.mark.parametrize("xp", [np, cp])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64,
                                   np.complex128])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("shape", [(2), (2, 3), (1, 2, 3)])
def test_type_casts(xp, shape: tuple, dtype: np.dtype, order: str):
    """Tests bindings to C++ functions that take gpaw::NDArray as input.
    Function call should fail if called with incompatible array => TypeError.
    """
    if xp == cp and cupy_is_fake:
        pytest.skip("Cupy is fake")

    a = xp.empty(shape=shape, order=order, dtype=dtype)
    a = cast(np.ndarray, a) if xp is np else cast(cp.ndarray, a)

    # Call a bunch of test functions from C++. These all assume a specific
    # kind of NDArray input and will throw TypeError on incompatible input.
    # They also that imported C++ array has correct metadata and throw
    # RuntimeError on failure.
    # The former is a user error, latter is a bug in gpaw::NDArray importer

    def maybe_raises(a, testfunc: Callable, success_condition: bool):

        # Pass same array twice. First arg gets converted to gpaw::NDArray.
        # Second arg is kept as PyObject handle and used for metadata checks
        if not success_condition:
            with pytest.raises(TypeError):
                testfunc(a, a)
        else:
            testfunc(a, a)

    maybe_raises(a, arraytests.accept_any, True)
    maybe_raises(a, arraytests.accept_cpu, xp is np)
    maybe_raises(a, arraytests.accept_gpu, xp is not np)
    maybe_raises(a, arraytests.accept_c_contig, a.flags.c_contiguous)
    maybe_raises(a, arraytests.accept_float, dtype is np.float32)
    maybe_raises(a, arraytests.accept_complexdouble, dtype is np.complex128)
    maybe_raises(a, arraytests.accept_2D, a.ndim == 2)
    maybe_raises(a, arraytests.accept_shape_any, a.ndim == 1)
    maybe_raises(a, arraytests.accept_shape_1_2_3, a.shape == (1, 2, 3))
    maybe_raises(a, arraytests.accept_shape_2_any,
                 a.shape[0] == 2 and a.ndim == 2)
    maybe_raises(a, arraytests.accept_double_c_contig_3D,
                 dtype is np.float64 and a.flags.c_contiguous and a.ndim == 3)
    maybe_raises(a, arraytests.accept_complexdouble_f_contig_cpu_shape_2_3,
                 xp is np and dtype is np.complex128 and a.flags.f_contiguous
                 and a.shape == (2, 3))
    maybe_raises(a, arraytests.accept_complexfloat_c_contig_gpu_shape_any_any,
                 dtype == np.complex64 and a.flags.c_contiguous
                 and xp is cp and a.ndim == 2)
