import cupy
import numpy
import numpy.typing
import typing
from . import magma as magma

class SplineTester32:
    """Utility class for exposing GPU splines for Python-side testing."""
    def __init__(self, r_cutoff: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """__init__(self: _gpaw.gpu.SplineTester32, r_cutoff: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None"""
    def evaluate_cpu(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]:
        """evaluate_cpu(self: _gpaw.gpu.SplineTester32, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]"""
    def evaluate_gpu(self, arg0: cupy.ndarray) -> numpy.typing.NDArray[numpy.float32]:
        """evaluate_gpu(self: _gpaw.gpu.SplineTester32, arg0: cupy.ndarray) -> numpy.typing.NDArray[numpy.float32]"""

class SplineTester64:
    """Utility class for exposing GPU splines for Python-side testing."""
    def __init__(self, r_cutoff: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """__init__(self: _gpaw.gpu.SplineTester64, r_cutoff: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None"""
    def evaluate_cpu(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """evaluate_cpu(self: _gpaw.gpu.SplineTester64, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]"""
    def evaluate_gpu(self, arg0: cupy.ndarray) -> numpy.typing.NDArray[numpy.float64]:
        """evaluate_gpu(self: _gpaw.gpu.SplineTester64, arg0: cupy.ndarray) -> numpy.typing.NDArray[numpy.float64]"""

def test_array_metadata(array: object, same_array: cupy.ndarray) -> None:
    """test_array_metadata(array: object, same_array: cupy.ndarray) -> None

    Tests that the array type caster correctly copies metadata from cupy ndarray. Use by passing the same array as both inputs. Raises RuntimeError on failure.
    """
def test_cupy_input(a: cupy.ndarray) -> None:
    """test_cupy_input(a: cupy.ndarray) -> None

    Simply tests that our type caster works, ie. that we can pass a Cupy ndarray to a C++ function accepting gpaw::PyDeviceArray
    """
