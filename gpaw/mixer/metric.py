import numpy as np

from gpaw.core.arrays import XArray
from gpaw.typing import ArrayND
from gpaw.core.plane_waves import PWDesc
from gpaw.core.uniform_grid import UGDesc, UGArray


class BaseMetric:
    def __init__(self,
                 g_ss: ArrayND | None,
                 ncomponents: int,
                 grid: UGDesc,
                 xp=np):
        """Base class for mixer metrics.

        This metric operates locally in real space, optionally applying a
        mixing matrix in spin space.

        Parameters
        ----------
        g_ss:
            Spin-space metric. Can be None (identity) or a matrix of shape
            (ncomponents, ncomponents).
        ncomponents:
            Number of density components.
        grid:
            Grid descriptor for the density.
        xp:
            Array namespace (numpy or cupy).
        """
        self.g_ss = g_ss
        self.ncomponents = ncomponents
        self.xp = xp
        if self.g_ss is None or len(self.g_ss) != ncomponents:
            self.g_ss = xp.eye(ncomponents)
        else:
            self.g_ss = self.xp.array(self.g_ss)
        self.grid = grid
        self.pw = PWDesc(ecut=0.99 * grid.ekin_max(),
                         cell=grid.cell,
                         comm=grid.comm)

    def __call__(self, n_sX: XArray,
                 out: XArray | None = None) -> XArray:
        if out is None:
            out = n_sX.new()
        out.matrix.data[:] = self.g_ss @ n_sX.matrix.data
        return out

    def __str__(self) -> str:
        assert isinstance(self.g_ss, self.xp.ndarray)
        if (self.g_ss == self.xp.eye(self.ncomponents)).all():
            return 'No metric'
        else:
            return f'Spin metric: {self.g_ss.tolist()})'


class FFTMetric(BaseMetric):
    def __init__(self,
                 weight: float,
                 sigma: float,
                 g_ss: ArrayND | None = None,
                 **kwargs):
        """Fourier-space metric for density mixing.

        Applies a weight to the density in reciprocal space to suppress
        long-wavelength charge sloshing (Kerker-style preconditioning).

        Parameters
        ----------
        weight:
            Preconditioning weight factor.
        sigma:
            Softening parameter for the Fourier-space weight.
        g_ss:
            Spin-space metric matrix.
        **kwargs:
            Additional arguments passed to
            :class:`~gpaw.mixer.metric.BaseMetric`.
        """
        self.weight = weight
        self.sigma = sigma
        super().__init__(g_ss, **kwargs)

    def __call__(self, n_sX: XArray,
                 out: XArray | None = None) -> XArray:
        if out is None:
            out = n_sX.new()
        # We will now assume that the density is represented
        # in real space. This may change in the future.

        ekin_G = self.xp.asarray(self.pw.ekin_G)
        w_G = (self.weight * self.sigma + ekin_G) / (self.sigma + ekin_G)

        assert isinstance(n_sX, UGArray)

        n_sX_pbc = n_sX.to_pbc_grid()
        if n_sX_pbc is n_sX:
            n_sX_pbc = n_sX_pbc.copy()
        n_sG = n_sX_pbc.fft(pw=self.pw)
        n_sG.data *= w_G
        n_sG.ifft(grid=n_sX_pbc, out=n_sX_pbc)
        out.from_pbc_grid(n_sX_pbc)

        out.matrix.data[:] = self.g_ss @ out.matrix.data

        return out

    def __str__(self) -> str:
        base = f'Fourier metric: weight={self.weight}, sigma={self.sigma}'
        super_str = super().__str__()
        if super_str != 'No metric':
            base += f'\n  {super_str}'
        return base
