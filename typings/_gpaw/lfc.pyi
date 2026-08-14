import collections.abc
import cupy
import numpy
import numpy.typing
import typing

class LFCBuilder:
    """Helper class for building C++ side LFC objects"""
    def __init__(self) -> None:
        """__init__(self: _gpaw.lfc.LFCBuilder) -> None"""
    def add_block(self, block_start_c, block_phi_idx_to_global: collections.abc.Sequence[typing.SupportsInt]) -> None:
        '''add_block(self: _gpaw.lfc.LFCBuilder, block_start_c: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"], block_phi_idx_to_global: collections.abc.Sequence[typing.SupportsInt]) -> None'''
    def add_phi(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2, arg3: typing.SupportsInt, arg4) -> None:
        '''add_phi(self: _gpaw.lfc.LFCBuilder, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg3: typing.SupportsInt, arg4: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"]) -> None'''
    def add_spline(self, cutoff: typing.SupportsFloat, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """add_spline(self: _gpaw.lfc.LFCBuilder, cutoff: typing.SupportsFloat, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None"""
    def build_gpu(self, arg0: SplinePoolGpu) -> LFCGpu:
        """build_gpu(self: _gpaw.lfc.LFCBuilder, arg0: SplinePoolGpu) -> _gpaw.lfc.LFCGpu

        Builds GPU-side LFC object
        """
    def build_spline_pool(self) -> SplinePool:
        """build_spline_pool(self: _gpaw.lfc.LFCBuilder) -> _gpaw.lfc.SplinePool

        Finalizes spline creation and returns a shared SplinePool object.
        """
    def define_grid(self, start_c, end_c, spacing_cv, dV: typing.SupportsFloat) -> None:
        '''define_grid(self: _gpaw.lfc.LFCBuilder, start_c: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"], end_c: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"], spacing_cv: typing.Annotated[collections.abc.Sequence[typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]], "FixedSize(3)"], dV: typing.SupportsFloat) -> None'''
    def set_block_size(self, block_size) -> None:
        '''set_block_size(self: _gpaw.lfc.LFCBuilder, block_size: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(3)"]) -> None'''

class LFCGpu:
    """Control object for GPU-LFC calculations (new LFC). No Python constructor: must be constructed using LFCBuilder"""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def add_to_density(self, nt_sG: cupy.ndarray, f_sM: cupy.ndarray) -> None:
        """add_to_density(self: _gpaw.lfc.LFCGpu, nt_sG: cupy.ndarray, f_sM: cupy.ndarray) -> None"""
    def calculate_potential_matrix(self, vt_G: cupy.ndarray, out_V_MN: cupy.ndarray, mu_start: typing.SupportsInt, mu_end: typing.SupportsInt) -> None:
        """calculate_potential_matrix(self: _gpaw.lfc.LFCGpu, vt_G: cupy.ndarray, out_V_MN: cupy.ndarray, mu_start: typing.SupportsInt, mu_end: typing.SupportsInt) -> None"""
    def construct_density(self, nt_G: cupy.ndarray, rho_MM: cupy.ndarray, mu_start: typing.SupportsInt, mu_end: typing.SupportsInt) -> None:
        """construct_density(self: _gpaw.lfc.LFCGpu, nt_G: cupy.ndarray, rho_MM: cupy.ndarray, mu_start: typing.SupportsInt, mu_end: typing.SupportsInt) -> None"""

class SplinePool:
    """C++ side object that holds CPU splines. Does not have a Python constructor."""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class SplinePoolGpu:
    """C++ side object that holds GPU splines."""
    def __init__(self, spline_pool_cpu: SplinePool) -> None:
        """__init__(self: _gpaw.lfc.SplinePoolGpu, spline_pool_cpu: _gpaw.lfc.SplinePool) -> None

        Construct from SplinePoolCpu
        """
