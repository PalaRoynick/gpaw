from types import SimpleNamespace

import numpy as np
import pytest

from gpaw.new.xc import Functional
from gpaw.setup import create_setup
from gpaw.new.xc.paw_xc_batch import pack_xc_batches, unpack_xc_batches
from gpaw.xc import XC
from gpaw.utilities import pack_density
from gpaw.core.atom_arrays import AtomArraysLayout


def test_xc_corrections2_rejects_other_functionals():
    functional = SimpleNamespace(
        use_xc_corrections2=True,
        name='RPBE')
    assert Functional.calculate_paw_corrections(
        functional, [], {}, None) is None


class MockSetups(list):
    @property
    def id_a(self):
        return [setup.fingerprint for setup in self]


@pytest.mark.parametrize('name', ['LDA', 'PBE'])
@pytest.mark.parametrize('nspins', [1, 2])
def test_xc_corrections2_matches_atom_loop(name, nspins):
    """The batched result graph reproduces established energies and dE/dD."""
    rng = np.random.default_rng(17)
    xc = XC(name)
    setup = create_setup('N', xc)
    setups = MockSetups([setup, setup])
    D_asii = AtomArraysLayout([(setup.ni, setup.ni) for _ in range(2)],
                              xp=np).empty(1)
    ni = setup.ni
    ii_data = rng.uniform(-1.0, 1.0, (ni, ni))
    ii_data += ii_data.T.conj()
    ii_data = np.array([ii_data, ii_data])
    D_asii.data[:] = ii_data.flatten()
    V_aL = AtomArraysLayout([(setup.Delta_pL.shape[1]) for _ in range(2)],
                            xp=np).empty()
    V_aL.data[:] = rng.uniform(-1.0, 1.0, V_aL.data.shape)

    expected_energy_a = {}
    expected_gradient_asp = {}
    for a, D_sii in D_asii.items():
        D_sp = pack_density(D_sii)
        gradient_sp = np.zeros_like(D_sp)
        expected_energy_a[a] = xc.calculate_paw_correction(
            setup, D_sp, gradient_sp, a=a)
        expected_gradient_asp[a] = gradient_sp
        D_p = D_sp.sum(0)
        dH_p = (setup.K_p + setup.M_p +
                setup.MB_p + 2.0 * D_p @ setup.M_pp.T +
                V_aL[a] @ setup.Delta_pL.T)
        expected_gradient_asp[a] += dH_p

    functional = SimpleNamespace(xp=np, name=name, xc=xc,
                                 backwards_compatible=True)
    batches = pack_xc_batches(
        D_asii, V_aL, setups, functional=functional)
    results = [batch.evaluate() for batch in batches]
    xc_corr = unpack_xc_batches(batches, results)

    for a, xc in xc_corr.items():
        energy = xc['e_xc']
        assert energy == pytest.approx(
            expected_energy_a[a], abs=2.0e-14)
        # The result graph differentiates the zero-weight origin consistently.
        # GPAW's legacy PBE transpose extrapolates an origin potential and
        # therefore includes a tiny additional derivative contribution.
        gradient_sp = xc['gradient_sp']
        gradient_tolerance = 2.0e-12
        assert gradient_sp == pytest.approx(
            expected_gradient_asp[a], abs=gradient_tolerance)
