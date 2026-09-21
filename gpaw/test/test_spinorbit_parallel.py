"""Regression test: soc_eigenstates() with more kpt_comm ranks than IBZ
k-points.

get_both_spins() (gpaw/spinorbit.py) pairs each k-point's two spin
channels across kpt_comm by forwarding the spin-down data onto its
k-point's spin-up rank and then dropping the sender's own copy.  Once
kpt_comm has more ranks than IBZ k-points, at least one rank is left
with zero completed (k, spin) pairs after that hand-off -- expected and
harmless for the point-to-point, per-BZ-k-point bookkeeping further
downstream, but extract_ibz_wave_functions(), BZWaveFunctions.__init__()
and OccupationNumberCalculator.calculate() each unconditionally indexed
into that rank's (possibly empty) local data and crashed with an
IndexError.
"""
import pytest
from ase.build import bulk

from gpaw import PW
from gpaw.mpi import world
from gpaw.spinorbit import soc_eigenstates


@pytest.mark.soc
@pytest.mark.parallel
@pytest.mark.skipif(world.size < 2, reason='needs at least 2 ranks')
def test_soc_eigenstates_more_ranks_than_ibz_kpoints(mpi, in_tmp_dir):
    """world.size - 1 k-points, pure k-point/spin parallelization over
    all of world.size ranks: kpt_comm.size (== world.size) is then
    always one more than nibzkpts, guaranteeing at least one rank owns
    no local (k, spin) pair, for any world.size >= 2."""
    n = world.size - 1
    atoms = bulk('Fe', 'bcc', a=2.87)
    atoms.set_initial_magnetic_moments([2.2])
    atoms.calc = mpi.GPAW(mode=PW(200), xc='PBE',
                          kpts=(n, 1, 1),
                          symmetry='off',
                          occupations={'name': 'fermi-dirac', 'width': 0.1},
                          nbands=10, convergence={'bands': 8},
                          parallel={'domain': 1, 'band': 1},
                          txt=None)
    atoms.get_potential_energy()

    assert atoms.calc.wfs.kd.comm.size == world.size
    assert atoms.calc.wfs.kd.nibzkpts == n
    assert world.size > n  # the triggering condition itself

    soc = soc_eigenstates(atoms.calc, theta=0.0, phi=0.0)
    eig_km = soc.eigenvalues()
    assert eig_km.shape == (n, 20)
