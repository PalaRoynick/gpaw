import numpy as np

from gpaw.projections import Projections
from gpaw.ibz2bz import get_overlap
from gpaw.mpi import serial_comm
from gpaw.utilities.partition import AtomPartition


class _GD:
    """get_overlap() only reads gd.dv, so a stub avoids a real grid."""
    dv = 1.0


def test_get_overlap_band_index_equivalence():
    """Regression for the spinor band-index bug in Wannier90.write_overlaps.

    Overlaps must be invariant to whether band selection happens *before*
    get_overlap (pre-sliced arrays + local indices, as the fix does) or
    *inside* it (full arrays + global indices). The spinor path previously
    pre-sliced the wavefunctions but indexed them again with global band
    numbers, giving wrong/out-of-range overlaps when a band window was used.
    """
    rng = np.random.default_rng(0)

    nbands = 6
    sel = [2, 3, 5]          # band window, e.g. from exclude_bands
    ni = 3                   # partial waves on a single atom
    ngrid = 4 * 4 * 4

    # spinor: the extra spin axis is flattened inside get_overlap
    shape = (nbands, 2, ngrid)

    def cnoise(s):
        return rng.standard_normal(s) + 1j * rng.standard_normal(s)

    ut1 = cnoise(shape)
    ut2 = cnoise(shape)

    # single atom, serial layout
    atom_partition = AtomPartition(serial_comm, np.array([0]))
    p1 = Projections(nbands, [ni], atom_partition, collinear=False)
    p2 = Projections(nbands, [ni], atom_partition, collinear=False)
    p1.array[:] = cnoise(p1.array.shape)
    p2.array[:] = cnoise(p2.array.shape)

    dO_aii = {0: np.eye(ni)}
    gd = _GD()

    # (a) full arrays, global band indices
    M_global = get_overlap(sel, gd, ut1, ut2, p1, p2, dO_aii)

    # (b) pre-slice, then local indices  -- what the fix does
    q1 = p1.new(nbands=len(sel))
    q1.array[:] = p1.array[sel]
    q2 = p2.new(nbands=len(sel))
    q2.array[:] = p2.array[sel]
    M_local = get_overlap(range(len(sel)), gd,
                          ut1[sel], ut2[sel], q1, q2, dO_aii)

    assert np.allclose(M_global, M_local)
