import numpy
import numpy.typing
import typing

class Spline32:
    '''Implements cubic spline interpolation for radial functions, uniform spacing.
        The spline is a C2-smooth piecewise polynomial
            S(x) = S_i(x) for x_i < x <= x_{i+1},
        such that
            S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3.
        This is intended for radial functions only, ie. x is in range [0, x_max].
        Boundary conditions are hardcoded:
           - "Clamped" boundaries, ie. first derivatives at endpoints are zero. S\'(0) = S\'(x_max) = 0.
           - S(x) = S(0) for x <= 0
           - S(x) = 0 for x >= x_max.'''
    def __init__(self, r_max: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """__init__(self: _gpaw.new.Spline32, r_max: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None

        Create a new spline from input data. Last value in the input array should correspond to spline value at r = r_max and MUST equal 0.0.
        """
    def evaluate(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]:
        """evaluate(self: _gpaw.new.Spline32, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]

        Vectorized evaluation of the spline. Output array shape will match that of the input array.
                Note that this may create a copy of the input array if its dtype does not match dtype of the spline.
        
        """
    @staticmethod
    def from_legacy_spline(spline: object) -> Spline32:
        """from_legacy_spline(spline: object) -> _gpaw.new.Spline32

        Construct new Spline object from a legacy GPAW Spline object.
        """
    def get_breakpoints(self) -> numpy.typing.NDArray[numpy.float32]:
        """get_breakpoints(self: _gpaw.new.Spline32) -> numpy.typing.NDArray[numpy.float32]

        Returns spline breakpoints {x_i}.
        """
    def get_num_bins(self) -> int:
        """get_num_bins(self: _gpaw.new.Spline32) -> int"""
    def get_x_end(self) -> float:
        """get_x_end(self: _gpaw.new.Spline32) -> float"""
    def get_x_start(self) -> float:
        """get_x_start(self: _gpaw.new.Spline32) -> float"""
    def __call__(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]:
        """__call__(self: _gpaw.new.Spline32, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]

        Equivalent to evaluate(x).
        
        """

class Spline64:
    '''Implements cubic spline interpolation for radial functions, uniform spacing.
        The spline is a C2-smooth piecewise polynomial
            S(x) = S_i(x) for x_i < x <= x_{i+1},
        such that
            S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3.
        This is intended for radial functions only, ie. x is in range [0, x_max].
        Boundary conditions are hardcoded:
           - "Clamped" boundaries, ie. first derivatives at endpoints are zero. S\'(0) = S\'(x_max) = 0.
           - S(x) = S(0) for x <= 0
           - S(x) = 0 for x >= x_max.'''
    def __init__(self, r_max: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """__init__(self: _gpaw.new.Spline64, r_max: typing.SupportsFloat, f_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create a new spline from input data. Last value in the input array should correspond to spline value at r = r_max and MUST equal 0.0.
        """
    def evaluate(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """evaluate(self: _gpaw.new.Spline64, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Vectorized evaluation of the spline. Output array shape will match that of the input array.
                Note that this may create a copy of the input array if its dtype does not match dtype of the spline.
        
        """
    @staticmethod
    def from_legacy_spline(spline: object) -> Spline64:
        """from_legacy_spline(spline: object) -> _gpaw.new.Spline64

        Construct new Spline object from a legacy GPAW Spline object.
        """
    def get_breakpoints(self) -> numpy.typing.NDArray[numpy.float64]:
        """get_breakpoints(self: _gpaw.new.Spline64) -> numpy.typing.NDArray[numpy.float64]

        Returns spline breakpoints {x_i}.
        """
    def get_num_bins(self) -> int:
        """get_num_bins(self: _gpaw.new.Spline64) -> int"""
    def get_x_end(self) -> float:
        """get_x_end(self: _gpaw.new.Spline64) -> float"""
    def get_x_start(self) -> float:
        """get_x_start(self: _gpaw.new.Spline64) -> float"""
    def __call__(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """__call__(self: _gpaw.new.Spline64, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Equivalent to evaluate(x).
        
        """
