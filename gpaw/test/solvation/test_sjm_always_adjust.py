"""always_adjust in constant-charge mode.

The charge stepping itself needs a converged constant-potential run, so it is
tested in gpaw/test/big/solvation/sjm/extension.py.
"""
import pytest
from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrochemistry import SJM
from gpaw.test.solvation.test_modes import NoCavity, Vacuum


def test_always_adjust_without_target_potential(in_tmp_dir):
    """It is a no-op in constant-charge mode, and says so."""
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 12.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    atoms.calc = GPAW(
        mode='fd', kpts=(2, 2, 1), occupations=FermiDirac(0.2), txt=None,
        extensions=[SJM(cavity=NoCavity(), dielectric=Vacuum(),
                        interactions=[], excess_electrons=-0.01,
                        always_adjust=True)])
    with pytest.warns(UserWarning, match='always_adjust'):
        atoms.get_potential_energy()
    assert atoms.calc.dft.sjm.excess_electrons == -0.01
    assert atoms.calc.dft.sjm.potentiostat.pending_excess_electrons is None
