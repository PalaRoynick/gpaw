import numpy as np

import meinsum
from gpaw.new.paw_xc_energy import spin_paired_pbe_paw_energy
from gpaw.setup import create_setup
from gpaw.sphere.lebedev import weight_n
from gpaw.xc import XC


def arguments(xcc, D_p):
    r_g = xcc.rgd.r_g
    inverse_r2_g = np.zeros_like(r_g)
    inverse_r2_g[1:] = r_g[1:] ** -2
    return (D_p, xcc.B_pqL, xcc.n_qg, xcc.nt_qg,
            xcc.nc_g, xcc.nct_g, xcc.Y_nL, xcc.rnablaY_nLv,
            xcc.rgd.dr_g, inverse_r2_g, weight_n, xcc.rgd.dv_g,
            xcc.e_xc0)


def test_pbe_energy_and_automatic_hamiltonian_match_gpaw():
    xc = XC('PBE')
    setup = create_setup('Mo', xc)
    xcc = setup.xc_correction
    D_sp = np.zeros((1, xcc.nii))
    D_sp[0, 0] = 1.0
    expected_dH_sp = np.zeros_like(D_sp)
    expected_energy = xc.calculate_paw_correction(
        setup, D_sp, expected_dH_sp)
    args = arguments(xcc, D_sp[0])

    direct_energy = spin_paired_pbe_paw_energy(*args)
    program = meinsum.differentiate_tensor_function(
        spin_paired_pbe_paw_energy)
    energy, (dH_p,) = meinsum.execute_tensor_program(program, *args)

    np.testing.assert_allclose(direct_energy, expected_energy,
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(energy, expected_energy,
                               rtol=2e-12, atol=2e-12)
    # GPAW's hand derivative has a small inconsistency at the radial origin.
    # Keep it as a compatibility check, but use the energy's finite difference
    # below as the definition of the correct gradient.
    np.testing.assert_allclose(dH_p, expected_dH_sp[0],
                               rtol=2e-4, atol=2e-6)

    rng = np.random.default_rng(7)
    direction = rng.normal(size=D_sp.shape[1])
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-5
    plus = list(args)
    minus = list(args)
    plus[0] = D_sp[0] + epsilon * direction
    minus[0] = D_sp[0] - epsilon * direction
    finite_difference = (
        spin_paired_pbe_paw_energy(*plus) -
        spin_paired_pbe_paw_energy(*minus)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        dH_p @ direction, finite_difference, rtol=2e-5, atol=2e-6)
