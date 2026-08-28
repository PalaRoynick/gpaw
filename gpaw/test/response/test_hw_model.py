import numpy as np
import pytest

from gpaw.response.frequencies import NonLinearFrequencyDescriptor
from gpaw.response.screened_interaction import (FullFrequencyHWModel,
                                                HWModel)


@pytest.mark.response
@pytest.mark.parametrize('myslice', [slice(0, 7), slice(2, 5)])
def test_contract_hw_matches_get_hw(myslice):
    rng = np.random.default_rng(42)
    wd = NonLinearFrequencyDescriptor(0.05, 0.5, 2.0)
    nw = len(wd)
    nG = 7
    nlocal_g = myslice.stop - myslice.start
    shape = (2, nw, nlocal_g, nG)
    HW_swgG = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    model = FullFrequencyHWModel(wd, HW_swgG, factor=0.1234)

    deps_m = np.concatenate(
        [rng.uniform(-2.2, 2.2, 24),
         [0.0, -0.0, 1e-16, -1e-16, 5.0, -5.0,  # deps = 0 and beyond grid
          wd.omega_w[-1], -wd.omega_w[-1]]])
    f_m = rng.choice([0.0, 0.5, 1.0], size=len(deps_m))  # f=0.5: s branch 0
    n_mG = (rng.standard_normal((len(deps_m), nG))
            + 1j * rng.standard_normal((len(deps_m), nG)))

    sigma_m, dsigma_m = model.contract_HW(deps_m, f_m, n_mG, myslice)

    # Reference 1: explicit loop over get_HW.
    for m, (deps, f) in enumerate(zip(deps_m, f_m)):
        S_gG, dS_gG = model.get_HW(deps, f)
        if S_gG is None:
            ref = dref = 0.0
        else:
            ref = n_mG[m, myslice] @ S_gG @ n_mG[m].conj()
            dref = n_mG[m, myslice] @ dS_gG @ n_mG[m].conj()
        assert sigma_m[m] == pytest.approx(ref, rel=1e-12, abs=1e-14)
        assert dsigma_m[m] == pytest.approx(dref, rel=1e-12, abs=1e-14)

    # Reference 2: the generic base-class fallback (the ppa/mpa path).
    fb_m, dfb_m = HWModel.contract_HW(model, deps_m, f_m, n_mG, myslice)
    assert np.allclose(sigma_m, fb_m, rtol=1e-12, atol=1e-14)
    assert np.allclose(dsigma_m, dfb_m, rtol=1e-12, atol=1e-14)
