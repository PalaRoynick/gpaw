import numpy as np
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
import pytest
from gpaw.response.pair import get_gs_and_context
from gpaw.mpi import world
from gpaw.response.bse import BSE, BSE_Plus
from gpaw import GPAW
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.coulomb_kernels import CoulombKernel


@pytest.mark.response
def test_bse_plus(in_tmp_dir, gpw_files):
    """
    This test makes a BSE plus calculation with the BSE_Plus class and
    manually to test that the BSE_Plus code is working. It tests that the
    assertions work.
    """
    calc = GPAW(gpw_files['si_pw'])

    gs, context = get_gs_and_context(
        gpw_files['si_pw'], txt=None, world=world, timer=None)
    ecut = 20
    eshift = 0
    eta = 0.1
    q_c = [0.0, 0.0, 0.0]
    bse = BSE(calc, ecut=ecut,
              q_c=q_c,
              valence_bands=range(1, 4),
              conduction_bands=range(4, 6),
              eshift=eshift,  # integrate_gamma = 0,
              mode='BSE',
              nbands=60,
              txt='bse_calculation.txt')

    wd = get_frequency_descriptor(np.linspace(0, 10, 101), gs=gs)

    chi0calc_small = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=6, n1=1,
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=None)

    # i tvivl om vi her skal bruge calc_es.write(calc, mode='all') eller om vi kan slippe.

    chi0calc_large = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=30,
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=None)

    w_w = np.linspace(0, 10, 101)

    chi_BSE_plus_WGG = BSE_Plus(bse, chi0calc_large, eta=eta, w_w=w_w,
                        q_c=q_c, direction='x', truncation=None)

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