import numpy as np
import pytest
from ase.units import Bohr

from gpaw.core import UGDesc
from gpaw.extensions.electrocatalysis.sjm import _write_trace_in_z

SIZE = (2, 2, 8)          # tiny: the z axis is all that matters
CELL = [2.0, 2.0, 8.0]    # bohr, orthorhombic -> h = 1 bohr in z


def make_trace(zerobc, path):
    """Write a z-trace of a ramp on a grid with/without a zero z boundary."""
    grid = UGDesc(cell=CELL, size=SIZE, pbc=(True, True, False),
                  zerobc=(False, False, zerobc))
    a_r = grid.empty()
    # A value that varies only along z, so the xy average is exact:
    a_r.data[:] = np.arange(a_r.data.shape[2], dtype=float) + 1.0
    _write_trace_in_z(a_r, str(path))
    return np.loadtxt(path)


@pytest.mark.serial
def test_zerobc_trace_is_padded_to_full_grid(tmp_path):
    """fd-style (zerobc) and pw-style grids must write the same z column."""
    fd = make_trace(True, tmp_path / 'fd.txt')
    pw = make_trace(False, tmp_path / 'pw.txt')

    nz = SIZE[2]
    h = CELL[2] / nz * Bohr

    # Same number of rows and the same z column in both modes:
    assert len(fd) == len(pw) == nz
    assert fd[0, 0] == 0.0 and pw[0, 0] == 0.0
    assert fd[:, 0] == pytest.approx(pw[:, 0])
    assert fd[-1, 0] == pytest.approx(CELL[2] * Bohr - h)

    # The zerobc grid stores nz - 1 points; row 0 is a copy of row 1:
    assert fd[0, 1] == fd[1, 1]
    assert fd[1:, 1] == pytest.approx(np.arange(nz - 1) + 1.0)

    # The full grid is written unchanged - no padding there:
    assert pw[:, 1] == pytest.approx(np.arange(nz) + 1.0)
    assert pw[0, 1] != pw[1, 1]
