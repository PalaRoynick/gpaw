import numpy as np
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
import pytest
from gpaw.response.pair import get_gs_and_context
from gpaw.mpi import world
from gpaw.response.bse import BSE, BSE_Plus
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.coulomb_kernels import CoulombKernel


@pytest.mark.response
def test_bse_plus(in_tmp_dir, gpw_files):
    """
    This test makes a BSE plus calculation with the BSE_Plus class and
    manually to test that the BSE_Plus code is working. It tests that the
    assertions work.
    """
    gs, context = get_gs_and_context(
        gpw_files['sic_pw'], txt=None, world=world, timer=None)
    ecut = 25
    eshift = 0
    eta = 0.1
    q_c = [0.0, 0.0, 0.0]
    bse = BSE(gpw_files['sic_pw'], ecut=ecut,
              q_c=q_c,
              valence_bands=range(2, 4),
              conduction_bands=range(4, 6),
              eshift=eshift,  # integrate_gamma = 0,
              mode='BSE',
              nbands=14)

    w_w = np.linspace(0, 10, 10)
    wd = get_frequency_descriptor(w_w, gs=gs)

    chi0calc_small = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=6, n1=2,
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=None)

    chi0calc_large = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=14,
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=None)

    bse_plus = BSE_Plus(bse, chi0calc_large, eta=eta, w_w=w_w,
                        q_c=q_c, direction='x', truncation=None)

    chi_bse_plus = bse_plus.get_chi_wGG(optical=True)

    if world.rank == 0:
        chi_BSE_plus_WGG = chi_bse_plus

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation=None)

    chi_irr_BSE_wGG = bse.get_chi_wGG(eta=eta,
                                      optical=True,
                                      no_coulomb=True,
                                      w_w=w_w)

    if world.rank == 0:
        chi_irr_BSE_WGG = chi_irr_BSE_wGG
    else:
        chi_irr_BSE_WGG = None

    chi0_data_small = chi0calc_small.calculate(q_c)
    dyson_eqs_small = Chi0DysonEquations(chi0_data_small, coulomb_kernel,
                                         xc_kernel=None, cd=gs.cd)
    chi0_small_wGG = dyson_eqs_small.get_chi0_wGG(direction='x')
    chi0_small_WGG = dyson_eqs_small.wblocks.all_gather(chi0_small_wGG)

    chi0_data_large = chi0calc_large.calculate(q_c)
    dyson_eqs_large = Chi0DysonEquations(chi0_data_large, coulomb_kernel,
                                         xc_kernel=None, cd=gs.cd)
    chi0_large_wGG = dyson_eqs_large.get_chi0_wGG(direction='x')
    chi0_large_WGG = dyson_eqs_large.wblocks.all_gather(chi0_large_wGG)

    v_G = coulomb_kernel.V(chi0_data_large.qpd)

    v_G[0] = 0.0

    if world.rank == 0:
        chi_irr_BSE_plus_WGG = \
            chi_irr_BSE_WGG - chi0_small_WGG + chi0_large_WGG

        eye = np.eye(len(chi_irr_BSE_plus_WGG[1]))

        chi_BSE_plus_WGG_manuel = \
            np.linalg.solve(eye - chi_irr_BSE_plus_WGG @ np.diag(v_G),
                            chi_irr_BSE_plus_WGG)

        assert chi_BSE_plus_WGG_manuel == pytest.approx(chi_BSE_plus_WGG,
                                                        rel=1e-3, abs=1e-4)

    # assertion error if different shapes of frequency grids are passed
    with pytest.raises(AssertionError, match=r'Shape mismatch:*'):
        chi0calc_wrong_w_1 = Chi0Calculator(gs, context,
                                            wd=get_frequency_descriptor(
                                                np.linspace(0, 20, 100),
                                                gs=gs),
                                            nbands=14,
                                            intraband=False,
                                            hilbert=False,
                                            eta=eta,
                                            ecut=ecut,
                                            eshift=None)

        BSE_Plus(bse, chi0calc_wrong_w_1,
                 eta=eta, w_w=np.linspace(0, 10, 10),
                 q_c=q_c, direction='x', truncation=None)

    # assertion error if different frequency values are passed
    with pytest.raises(AssertionError, match=r'Use the same frequency grid*'):
        chi0calc_wrong_w_2 = Chi0Calculator(gs, context,
                                            wd=get_frequency_descriptor(
                                                np.linspace(0, 20, 10), gs=gs),
                                            nbands=14,
                                            intraband=False,
                                            hilbert=False,
                                            eta=eta,
                                            ecut=ecut,
                                            eshift=None)

        BSE_Plus(bse, chi0calc_wrong_w_2,
                 eta=eta, w_w=np.linspace(0, 10, 10),
                 q_c=q_c, direction='x', truncation=None)

    # assertion error if different q_c's
    with pytest.raises(AssertionError, match=r'Use the same q_c*'):
        BSE_Plus(bse, chi0calc_large,
                 eta=eta, w_w=w_w,
                 q_c=[0.1, 0, 0], direction='x', truncation=None)

    # assertion error if calculated in different directions
    with pytest.raises(AssertionError, match=r'BSE and chi0 have to be*'):
        BSE_Plus(bse, chi0calc_large,
                 eta=eta, w_w=w_w,
                 q_c=q_c, direction='y', truncation=None)

    # assertion error if coulomb trunction is only used in one of them
    with pytest.raises(AssertionError, match=r'Inconsistent use of*'):
        bse_truncation_2d = BSE(gpw_files['sic_pw'], ecut=ecut,
                                q_c=q_c,
                                valence_bands=range(2, 4),
                                conduction_bands=range(4, 6),
                                eshift=eshift,  # integrate_gamma = 0,
                                mode='BSE',
                                nbands=14,
                                truncation='2D')

        BSE_Plus(bse_truncation_2d, chi0calc_large,
                 eta=eta, w_w=w_w,
                 q_c=q_c, direction='x', truncation=None)

    # assertion error if ecuts do not match
    with pytest.raises(AssertionError, match=r'Cutoff energies should*'):
        chi0calc_ecut_10 = Chi0Calculator(gs, context,
                                          wd=wd,
                                          nbands=14,
                                          intraband=False,
                                          hilbert=False,
                                          eta=eta,
                                          ecut=10,
                                          eshift=None)

        BSE_Plus(bse, chi0calc_ecut_10,
                 eta=eta, w_w=w_w,
                 q_c=q_c, direction='x', truncation=None)

    # assertion error if more bands in the bse calculation
    with pytest.raises(AssertionError, match=r'Large chi0 calculation*'):
        chi0calc_bands_5 = Chi0Calculator(gs, context,
                                          wd=wd,
                                          nbands=5,
                                          intraband=False,
                                          hilbert=False,
                                          eta=eta,
                                          ecut=ecut,
                                          eshift=None)

        BSE_Plus(bse, chi0calc_bands_5,
                 eta=eta, w_w=w_w,
                 q_c=q_c, direction='x', truncation=None)
