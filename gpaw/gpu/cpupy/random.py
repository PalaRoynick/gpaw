import numpy as np


def default_rng(seed):
    return Generator(np.random.default_rng(seed))


class Generator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def random(self, size=None, dtype=float, out=None):
        from gpaw.gpu.cpupy import ndarray
        np_out = None if out is None else out._data
        return ndarray(self.rng.random(size, dtype=dtype, out=np_out))
