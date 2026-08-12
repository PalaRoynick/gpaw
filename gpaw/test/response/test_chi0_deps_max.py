import numpy as np
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
import pytest
from gpaw.response.pair import get_gs_and_context
from gpaw.mpi import world
from gpaw.response.bse import BSE
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.coulomb_kernels import CoulombKernel


@pytest.mark.response
def test_chi0_emax(in_tmp_dir, gpw_files, monkeypatch):
    ''' This test makes a chi0 calculation with the RPA code and
    with the BSE code in RPA mode (and no use of Tamm-Dancoff) to
    test the implmentation of an energy cutoff parameter (deps_max)
    in the chi0-code.'''

    calc = gpw_files['si_pw_nbands10_converged']
    gs, context = get_gs_and_context(
        calc, txt=None, world=world, timer=None)
    ecut = 20
    eshift = 0.2
    eta = 0.1
    q_c = [0.0, 0.0, 0.0]
    bse_nbands = 10
    bse = BSE(calc, ecut=ecut,
              q_c=q_c,
              valence_bands=range(0, 8),
              conduction_bands=range(0, 8),
              eshift=eshift,
              mode='RPA',
              deps_max=3,
              nbands=bse_nbands)

    w_w = np.array([-3, 0, 6])

    chi0_RPA_WGG_from_bse = bse.get_chi_wGG(eta=eta,
                                            optical=True,
                                            irreducible=True,
                                            w_w=w_w)

    wd = get_frequency_descriptor(w_w, gs=gs)
    # chi0 calculator
    chi0calc_small = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=slice(0, 8),
                                    intraband=True,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    deps_max=3,
                                    eshift=eshift)

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation=None)

    # create chi0 with RPA code
    chi0_data_small = chi0calc_small.calculate(q_c)
    dyson_eqs_small = Chi0DysonEquations(chi0_data_small, coulomb_kernel,
                                         xc_kernel=None, cd=gs.cd)
    chi0_small_wGG = dyson_eqs_small.get_chi0_wGG(direction='x')
    chi0_small_WGG = dyson_eqs_small.wblocks.all_gather(chi0_small_wGG)

    if world.rank == 0:
        assert chi0_RPA_WGG_from_bse == pytest.approx(chi0_small_WGG,
                                                      rel=1e-6, abs=1e-8)
