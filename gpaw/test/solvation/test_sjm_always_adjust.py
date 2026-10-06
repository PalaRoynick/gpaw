"""always_adjust in constant-charge mode.

The charge stepping itself needs a converged constant-potential run, so it is
tested in gpaw/test/big/solvation/sjm/extension.py.
"""
import pytest

from gpaw.test.solvation.test_sjm_change import h_atoms


def test_always_adjust_without_target_potential(in_tmp_dir):
    """It is a no-op in constant-charge mode, and says so."""
    atoms, _ = h_atoms(excess_electrons=-0.01, always_adjust=True)
    with pytest.warns(UserWarning, match='always_adjust'):
        atoms.get_potential_energy()
    pot = atoms.calc.dft.potentiostat
    assert pot.excess_electrons == -0.01
    assert pot.pending_excess_electrons is None
