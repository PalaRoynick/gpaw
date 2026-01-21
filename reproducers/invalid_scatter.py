from gpaw.gpu import cupy as cp
from gpaw.mpi import world
import numpy as np
import numpy.typing as npt

"""Simpler version of scatter_from() in gpaw/core/plane_waves.py.
This function allows the input data to be None in non-root ranks,
in which case it creates a temporary array to use as the recv buffer
and eventually copies that to a class-internal array.

scatter_from(None) is called eg. from gpaw/core/uniform_grid.py::fft
"""

assert world.size > 1, "Please run this reproducer with at least 2 ranks"


def mpi_print(msg: str):
    print(f"Rank {world.rank}: " + msg)


class Scatterer:

    comm = world
    my_size = 1000
    dtype = np.float64

    def __init__(self, xp):
        self.xp = xp
        self.local_array = self.xp.empty((self.my_size), dtype=self.dtype)
        # fill with rank+1 so that we don't have zeros anywhere
        self.local_array.fill(self.comm.rank + 1)

    def scatter_from(self, data: npt.ArrayLike | None = None) -> None:
        """"""

        if self.comm.rank == 0:
            assert data is not None
            assert data.flags.c_contiguous
            assert self.local_array.flags.c_contiguous
            self.comm.scatter(data, self.local_array, 0)
        else:
            buf = self.xp.empty(self.my_size, dtype=self.dtype)
            assert buf.flags.c_contiguous
            self.comm.scatter(None, buf, 0)
            assert buf.flags.c_contiguous
            self.local_array[:] = buf[:len(self.local_array)]

    def test_scatter(self) -> None:
        """"""

        a = None
        if self.comm.rank == 0:
            #a = self.xp.asanyarray(np.arange(self.my_size * self.comm.size), dtype=self.dtype)
            a = self.xp.empty((self.my_size * self.comm.size), dtype=self.dtype)
            a.fill(42)

        self.scatter_from(a)


for xp in [np, cp]:

    if xp is np:
        print("Testing with Numpy arrays")
    else:
        print("Testing with Cupy arrays")

    scatterer = Scatterer(xp)

    mpi_print(f"Data before scatter: {scatterer.local_array}")
    scatterer.test_scatter()
    mpi_print(f"Data after scatter: {scatterer.local_array}")
