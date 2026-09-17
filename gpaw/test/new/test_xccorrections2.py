from types import SimpleNamespace

import numpy as np
import pytest

from gpaw.new.xccorrections2 import evaluate_paw_xc
from gpaw.new.xc import Functional
from gpaw.setup import create_setup
from gpaw.utilities.tensor_batch import gather_xc_paw, scatter_xc_paw
from gpaw.xc import XC


def test_xc_corrections2_rejects_other_functionals():
    functional = SimpleNamespace(
        use_xc_corrections2=True,
        name='RPBE')
    with pytest.raises(AssertionError):
        Functional.calculate_paw_corrections(functional, [], {})


@pytest.mark.parametrize('name', ['LDA', 'PBE'])
@pytest.mark.parametrize('nspins', [1, 2])
def test_xc_corrections2_matches_atom_loop(name, nspins):
    """The batched result graph reproduces established energies and dE/dD."""
    rng = np.random.default_rng(17)
    xc = XC(name)
    setup = create_setup('N', xc)
    npacked = setup.ni * (setup.ni + 1) // 2
    D_asp = {
        a: 0.2 + 0.1 * rng.random((nspins, npacked))
        for a in range(2)}

    expected_energy_a = {}
    expected_gradient_asp = {}
    for a, D_sp in D_asp.items():
        gradient_sp = np.zeros_like(D_sp)
        expected_energy_a[a] = xc.calculate_paw_correction(
            setup, D_sp, gradient_sp, a=a)
        expected_gradient_asp[a] = gradient_sp

    functional = SimpleNamespace(xp=np, name=name, xc=xc)
    batches = gather_xc_paw(
        np, [setup, setup], D_asp, gga=name == 'PBE')
    results = tuple(evaluate_paw_xc(functional, batch)
                    for batch in batches)
    energy_a, gradient_asp = scatter_xc_paw(batches, results)

    for a in D_asp:
        assert energy_a[a] == pytest.approx(
            expected_energy_a[a], abs=2.0e-11)
        # The result graph differentiates the zero-weight origin consistently.
        # GPAW's legacy PBE transpose extrapolates an origin potential and
        # therefore includes a tiny additional derivative contribution.
        gradient_tolerance = 5.0e-7 if name == 'PBE' else 2.0e-10
        assert gradient_asp[a] == pytest.approx(
            expected_gradient_asp[a], abs=gradient_tolerance)
