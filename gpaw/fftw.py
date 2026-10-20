"""
Python wrapper for FFTW3 library
================================

.. autoclass:: FFTPlans

"""
from __future__ import annotations

import weakref
from types import ModuleType

import numpy as np
from scipy.fft import fftn, ifftn, irfftn, rfftn

import gpaw.cgpaw as cgpaw
from gpaw.gpu import is_hip
from gpaw.typing import Array3D, DTypeLike, IntVector
from gpaw.utilities import as_complex_dtype, as_real_dtype, get_dtype_precision

ESTIMATE = 64
MEASURE = 0
PATIENT = 32
EXHAUSTIVE = 8

_plan_cache: dict[tuple, weakref.ReferenceType] = {}


def have_fftw() -> bool:
    """Did we compile with FFTW?"""
    return hasattr(cgpaw, 'FFTWPlan')


def check_fft_size(n: int, factors=[2, 3, 5, 7]) -> bool:
    """Check if n is an efficient fft size.

    Efficient means that n can be factored into small primes (2, 3, 5, 7).

    >>> check_fft_size(17)
    False
    >>> check_fft_size(18)
    True
    """

    if n == 1:
        return True
    for x in factors:
        if n % x == 0:
            return check_fft_size(n // x, factors)
    return False


def get_efficient_fft_size(N: int, n=1, factors=[2, 3, 5, 7]) -> int:
    """Return smallest efficient fft size.

    Must be greater than or equal to N and divisible by n.

    >>> get_efficient_fft_size(17)
    18
    """
    N = -(-N // n) * n
    while not check_fft_size(N, factors):
        N += n
    return N


def empty(shape, dtype=float):
    """numpy.empty() equivalent with 16 byte alignment."""
    assert np.issubdtype(dtype, np.complexfloating)

    real_dtype = as_real_dtype(dtype)
    complex_dtype = as_complex_dtype(dtype)

    N = np.prod(shape)
    a = np.empty(2 * N + 16 // real_dtype.itemsize - 1, real_dtype)
    offset = (a.ctypes.data % 16) // real_dtype.itemsize
    a = a[offset:2 * N + offset].view(complex_dtype)
    return a.reshape(shape)


def create_plans(size_c: IntVector,
                 dtype: DTypeLike,
                 flags: int = MEASURE,
                 xp: ModuleType = np,
                 blocksize: int = 1) -> FFTPlans:
    """Create plan-objects for FFT and inverse FFT."""
    key = (tuple(size_c), dtype, flags, xp, blocksize)
    # Look up weakref to plan:
    if key in _plan_cache:
        plan = _plan_cache[key]()
        # Check if plan is still "alive":
        if plan is not None:
            return plan
    # Create new plan:
    if xp is not np:
        plan = CuPyFFTPlans(size_c, dtype, blocksize=blocksize)
    elif have_fftw() and get_dtype_precision(dtype) != 'single':
        plan = FFTWPlans(size_c, dtype, flags, blocksize=blocksize)
    else:
        plan = NumpyFFTPlans(size_c, dtype, blocksize=blocksize)
    _plan_cache[key] = weakref.ref(plan)
    return plan


class FFTPlans:
    xp = np

    def __init__(self,
                 size_c: IntVector,
                 dtype: DTypeLike,
                 empty=empty,
                 blocksize: int = 1):
        self.size_c = size_c
        self.shape: tuple[int, ...]
        self.blocksize = blocksize

        if np.issubdtype(dtype, np.floating):
            if blocksize > 1:
                self.shape = (blocksize, size_c[0], size_c[1],
                              size_c[2] // 2 + 1)
            else:
                self.shape = (size_c[0], size_c[1],
                              size_c[2] // 2 + 1)
            self.tmp_Q = empty(self.shape, as_complex_dtype(dtype))
            # Real data with padding. NB: Numpy marks this as non-c-contiguous
            self.tmp_R = self.tmp_Q.view(dtype)[..., :, :, :size_c[2]]
        else:
            if blocksize > 1:
                self.shape = (blocksize, *tuple(size_c))
            else:
                self.shape = tuple(size_c)
            self.tmp_Q = empty(self.shape, dtype)
            self.tmp_R = self.tmp_Q

    def fft(self) -> None:
        """Do FFT from ``tmp_R`` to ``tmp_Q``.

        >>> plans = create_plans([4, 1, 1], float)
        >>> plans.tmp_R[:, 0, 0] = [1, 0, 1, 0]
        >>> plans.fft()
        >>> plans.tmp_Q[:, 0, 0]
        array([2.+0.j, 0.+0.j, 2.+0.j, 0.+0.j])
        """
        raise NotImplementedError

    def ifft(self) -> None:
        """Do inverse FFT from ``tmp_Q`` to ``tmp_R``.

        >>> plans = create_plans([4, 1, 1], complex)
        >>> plans.tmp_Q[:, 0, 0] = [0, 1j, 0, 0]
        >>> plans.ifft()
        >>> plans.tmp_R[:, 0, 0]
        array([ 0.+1.j, -1.+0.j,  0.-1.j,  1.+0.j])
        """
        raise NotImplementedError

    def ifft_sphere(self, coef_nG, pw, out_nR=None):
        if coef_nG is None:
            out_nR.scatter_from(None)
            return
        self.tmp_Q[:] = 0
        if self.blocksize > 1:
            blocksize = coef_nG.shape[0]
            assert self.blocksize >= blocksize
            pw.paste(coef_nG, self.tmp_Q[:blocksize], self.size_c)
        else:
            pw.paste(coef_nG, self.tmp_Q, self.size_c)

        if np.issubdtype(pw.dtype, np.floating):
            t = self.tmp_Q[..., 0]
            if out_nR is not None:
                assert (
                    out_nR.desc.size_c[:2] == self.tmp_R.shape[-3:-1]).all()
            n, m = (s // 2 - 1 for s in self.tmp_R.shape[-3:-1])
            t[..., 0, -m:] = t[..., 0, m:0:-1].conj()
            t[..., n:0:-1, -m:] = t[..., -n:, m:0:-1].conj()
            t[..., -n:, -m:] = t[..., n:0:-1, m:0:-1].conj()
            t[..., -n:, 0] = t[..., n:0:-1, 0].conj()

        self.ifft()
        if out_nR is not None:
            out_nR.scatter_from(self.tmp_R)

    def fft_sphere(self, in_nR, pw):
        self.tmp_R[:] = in_nR
        self.fft()
        coefs = pw.cut(self.tmp_Q) * (self.blocksize / self.tmp_R.size)
        return coefs


class FFTWPlans(FFTPlans):
    # The test suite likes to override the FFTW flags since methods
    # like MEASURE are not guaranteed reproducible.
    _overwrite_flags = None

    """FFTW3 3d transforms."""
    def __init__(self, size_c, dtype, flags=MEASURE,
                 blocksize: int = 1):
        if not have_fftw():
            raise ImportError('Not compiled with FFTW.')
        super().__init__(size_c, dtype, blocksize=blocksize)
        if self._overwrite_flags is not None:
            flags = self._overwrite_flags

        assert self.tmp_R.dtype != np.complex64
        assert self.tmp_R.dtype != np.float32
        self._fftplan = cgpaw.FFTWPlan(self.tmp_R, self.tmp_Q, -1, flags)
        self._ifftplan = cgpaw.FFTWPlan(self.tmp_Q, self.tmp_R, 1, flags)

    def fft(self):
        cgpaw.FFTWExecute(self._fftplan)

    def ifft(self):
        cgpaw.FFTWExecute(self._ifftplan)

    def __del__(self):
        # Attributes will not exist if execution stops during FFTW planning
        if hasattr(self, '_fftplan'):
            cgpaw.FFTWDestroy(self._fftplan)
        if hasattr(self, '_ifftplan'):
            cgpaw.FFTWDestroy(self._ifftplan)


class NumpyFFTPlans(FFTPlans):
    """Numpy fallback."""
    def fft(self):
        if np.issubdtype(self.tmp_R.dtype, np.floating):
            self.tmp_Q[:] = rfftn(self.tmp_R, axes=(-3, -2, -1),
                                  overwrite_x=True)
        else:
            self.tmp_Q[:] = fftn(self.tmp_R, axes=(-3, -2, -1),
                                 overwrite_x=True)

    def ifft(self):
        if np.issubdtype(self.tmp_R.dtype, np.floating):
            self.tmp_R[:] = irfftn(self.tmp_Q, self.tmp_R.shape[-3:],
                                   norm='forward', overwrite_x=True)
        else:
            self.tmp_R[:] = ifftn(self.tmp_Q, self.tmp_R.shape[-3:],
                                  norm='forward', overwrite_x=True)


def rfftn_patch(tmp_R):
    from gpaw.gpu import cupyx
    return cupyx.scipy.fft.fftn(
        tmp_R, axes=(-3, -2, -1))[:, :, :tmp_R.shape[-1] // 2 + 1]


def irfftn_patch(B, shape):
    import cupy as xp

    from gpaw.gpu import cupyx
    A = xp.empty(shape, dtype=complex)
    A[:, :, :B.shape[2]] = B
    inv_ind1 = -xp.arange(B.shape[0])[:, None, None]
    inv_ind2 = -xp.arange(B.shape[1])[None, :, None]
    inv_ind3 = -xp.arange(1, B.shape[2])[None, None, :]
    A[:, :, -(B.shape[2] - 1):] = B[inv_ind1, inv_ind2, inv_ind3].conj()
    return cupyx.scipy.fft.ifftn(A, axes=(-3, -2, -1)).real


class CuPyFFTPlans(FFTPlans):
    def __init__(self,
                 size_c: IntVector,
                 dtype: DTypeLike,
                 blocksize: int = 1):
        from gpaw.gpu import cupy as cp
        self.dtype = dtype
        super().__init__(size_c, dtype, empty=cp.empty,
                         blocksize=blocksize)
        self.xp = cp

    def fft(self):
        from gpaw.gpu import cupyx
        if np.issubdtype(self.tmp_R.dtype, np.floating):
            if is_hip:
                self.tmp_Q[:] = rfftn_patch(self.tmp_R)
            else:
                self.tmp_Q[:] = cupyx.scipy.fft.rfftn(
                    self.tmp_R, axes=(-3, -2, -1), overwrite_x=True)
        else:
            self.tmp_Q[:] = cupyx.scipy.fft.fftn(
                self.tmp_R, axes=(-3, -2, -1), overwrite_x=True)

    def ifft(self):
        from gpaw.gpu import cupyx
        if np.issubdtype(self.tmp_R.dtype, np.floating):
            if is_hip:
                self.tmp_R[:] = irfftn_patch(
                    self.tmp_Q, self.tmp_R.shape[-3:]) * self.tmp_nR.size
            else:
                self.tmp_R[:] = cupyx.scipy.fft.irfftn(
                    self.tmp_Q, self.tmp_R.shape[-3:],
                    norm='forward',
                    overwrite_x=True)
        else:
            self.tmp_R[:] = cupyx.scipy.fft.ifftn(
                self.tmp_Q, self.tmp_R.shape[-3:],
                norm='forward',
                overwrite_x=True)


# The rest of this file will be removed in the future ...

def check_fftw_inputs(in_R, out_R):
    for arr in in_R, out_R:
        # Note: Arrays not necessarily contiguous due to 16-byte alignment
        assert arr.ndim == 3  # We can perhaps relax this requirement
        assert arr.dtype == float or arr.dtype == complex

    if in_R.dtype == out_R.dtype == complex:
        assert in_R.shape == out_R.shape
    else:
        # One real and one complex:
        R, C = (in_R, out_R) if in_R.dtype == float else (out_R, in_R)
        assert C.dtype == complex
        assert R.shape[:2] == C.shape[:2]
        assert C.shape[2] == 1 + R.shape[2] // 2


class FFTPlan:
    """FFT 3d transform."""
    def __init__(self,
                 in_R: Array3D,
                 out_R: Array3D,
                 sign: int,
                 flags: int = MEASURE):
        check_fftw_inputs(in_R, out_R)
        self.in_R = in_R
        self.out_R = out_R
        self.sign = sign
        self.flags = flags

    def execute(self) -> None:
        raise NotImplementedError


class FFTWPlan(FFTPlan):
    """FFTW3 3d transform."""

    def __init__(self, in_R, out_R, sign, flags=MEASURE):
        if not have_fftw():
            raise ImportError('Not compiled with FFTW.')
        # fftw_plan handle (integer)
        self._handle = cgpaw.FFTWPlan(in_R, out_R, sign, flags)
        super().__init__(in_R, out_R, sign, flags)

    def execute(self):
        """"""
        cgpaw.FFTWExecute(self._handle)

    def __del__(self):
        if getattr(self, '_handle', None):
            cgpaw.FFTWDestroy(self._handle)
        self._handle = None


class NumpyFFTPlan(FFTPlan):
    """Numpy fallback."""
    def execute(self):
        if self.in_R.dtype == float:
            self.out_R[:] = np.fft.rfftn(self.in_R)
        elif self.out_R.dtype == float:
            self.out_R[:] = np.fft.irfftn(self.in_R,
                                          self.out_R.shape,
                                          [0, 1, 2])
            self.out_R *= self.out_R.size
        elif self.sign == 1:
            self.out_R[:] = np.fft.ifftn(self.in_R,
                                         self.out_R.shape,
                                         [0, 1, 2])
            self.out_R *= self.out_R.size
        else:
            self.out_R[:] = np.fft.fftn(self.in_R)


def create_plan(in_R: Array3D,
                out_R: Array3D,
                sign: int,
                flags: int = MEASURE) -> FFTPlan:
    if have_fftw() and get_dtype_precision(in_R.dtype) != 'single':
        return FFTWPlan(in_R, out_R, sign, flags)
    return NumpyFFTPlan(in_R, out_R, sign, flags)
