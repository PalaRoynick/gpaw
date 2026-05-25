from time import perf_counter

import numpy as np
import pytest
from math import pi, sqrt

from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv


def test_paw_double_array_matmul_transposed_views():
    matmul = pytest.importorskip('_gpaw')._test_paw_double_array_matmul

    rng = np.random.default_rng(472)
    cases = [
        (rng.normal(size=(4, 3)),
         rng.normal(size=(3, 5))),
        (rng.normal(size=(3, 4)).T,
         rng.normal(size=(3, 5))),
        (rng.normal(size=(4, 3)),
         rng.normal(size=(5, 3)).T),
        (rng.normal(size=(3, 4)).T,
         rng.normal(size=(5, 3)).T),
    ]

    for a, b in cases:
        actual = np.empty((a.shape[0], b.shape[1]))
        matmul(a, b, actual)
        np.testing.assert_allclose(actual, a @ b,
                                   rtol=0.0, atol=2.0e-14)


def derivative2_reference(dr_g, a_g):
    c_g = a_g / dr_g
    b_g = np.empty_like(a_g)
    b_g[0] = 0.5 * c_g[1] + c_g[0]
    b_g[1] = 0.5 * c_g[2] - c_g[0]
    b_g[1:-1] = 0.5 * (c_g[2:] - c_g[:-2])
    b_g[-2] = c_g[-1] - 0.5 * c_g[-3]
    b_g[-1] = -c_g[-1] - 0.5 * c_g[-2]
    return b_g


def derivative_reference(dr_g, n_g):
    dndr_g = np.empty_like(n_g)
    dndr_g[0] = (n_g[1] - n_g[0]) / dr_g[0]
    dndr_g[1:-1] = 0.5 * (n_g[2:] - n_g[:-2]) / dr_g[1:-1]
    dndr_g[-1] = (n_g[-1] - n_g[-2]) / dr_g[-1]
    return dndr_g


def project_density_reference(n_sLg, dndr_sLg, Y_nL, rnablaY_nLv, r_g):
    nspins, Lmax, ngrids = n_sLg.shape
    nlebedev = len(Y_nL)
    n_sng = np.einsum('nL,sLg->sng', Y_nL, n_sLg, optimize=True)
    a_sng = np.einsum('nL,sLg->sng', Y_nL, dndr_sLg, optimize=True)
    b_vnsg = np.einsum('nLv,sLg->vnsg', rnablaY_nLv, n_sLg,
                        optimize=True)
    sigma_xng = np.empty((2 * nspins - 1, nlebedev, ngrids))
    sigma_xng[::2] = (b_vnsg ** 2).sum(0).transpose(1, 0, 2)
    if nspins == 2:
        sigma_xng[1] = (b_vnsg[:, :, 0] * b_vnsg[:, :, 1]).sum(0)
    sigma_xng[:, :, 1:] /= r_g[1:] ** 2
    sigma_xng[:, :, 0] = sigma_xng[:, :, 1]
    sigma_xng[::2] += a_sng ** 2
    if nspins == 2:
        sigma_xng[1] += a_sng[0] * a_sng[1]
    return n_sng, a_sng, b_vnsg, sigma_xng


def radial_gradient_correction_reference(dr_g, dv_g, dedsigma_xng, a_sng):
    nspins, nlebedev, ngrids = a_sng.shape
    vv_sng = np.empty_like(a_sng)
    for s in range(nspins):
        for n in range(nlebedev):
            vv_sng[s, n] = derivative2_reference(
                dr_g, -2 * dv_g * dedsigma_xng[2 * s, n] * a_sng[s, n])

    if nspins == 2:
        for n in range(nlebedev):
            vv_sng[0, n] -= derivative2_reference(
                dr_g, dv_g * dedsigma_xng[1, n] * a_sng[1, n])
            vv_sng[1, n] -= derivative2_reference(
                dr_g, dv_g * dedsigma_xng[1, n] * a_sng[0, n])

    vv_sng[:, :, 1:] /= dv_g[1:]
    vv_sng[:, :, 0] = vv_sng[:, :, 1]
    return vv_sng


@pytest.mark.parametrize('nspins', [1, 2])
def test_paw_gga_project_density(nspins):
    paw_gga_project_density = pytest.importorskip(
        'gpaw.cgpaw').paw_gga_project_density

    rng = np.random.default_rng(311)
    Lmax = 5
    nlebedev = 7
    ngrids = 13
    r_g = np.linspace(0.0, 1.4, ngrids)
    n_sLg = rng.normal(size=(nspins, Lmax, ngrids))
    dndr_sLg = rng.normal(size=(nspins, Lmax, ngrids))
    Y_nL = rng.normal(size=(nlebedev, Lmax))
    rnablaY_nLv = rng.normal(size=(nlebedev, Lmax, 3))

    actual_n_sng = np.empty((nspins, nlebedev, ngrids))
    actual_a_sng = np.empty_like(actual_n_sng)
    actual_b_vnsg = np.empty((3, nlebedev, nspins, ngrids))
    actual_sigma_xng = np.empty((2 * nspins - 1, nlebedev, ngrids))

    paw_gga_project_density(n_sLg=n_sLg,
                            dndr_sLg=dndr_sLg,
                            Y_nL=Y_nL,
                            rnablaY_nLv=rnablaY_nLv,
                            r_g=r_g,
                            n_sng=actual_n_sng,
                            a_sng=actual_a_sng,
                            b_vnsg=actual_b_vnsg,
                            sigma_xng=actual_sigma_xng)

    expected = project_density_reference(
        n_sLg, dndr_sLg, Y_nL, rnablaY_nLv, r_g)
    for actual, expected_array in zip((actual_n_sng, actual_a_sng,
                                       actual_b_vnsg, actual_sigma_xng),
                                      expected):
        np.testing.assert_allclose(actual, expected_array,
                                   rtol=2.0e-14, atol=2.0e-14)


@pytest.mark.parametrize('nspins', [1, 2])
def test_paw_gga_radial_gradient_correction(nspins):
    cgpaw = pytest.importorskip('gpaw.cgpaw')
    paw_gga_radial_gradient_correction = pytest.importorskip(
        'gpaw.cgpaw').paw_gga_radial_gradient_correction

    rng = np.random.default_rng(812)
    nlebedev = 7
    ngrids = 13
    dr_g = 0.03 + rng.random(ngrids)
    dv_g = 0.02 + rng.random(ngrids)
    dedsigma_xng = rng.normal(size=(2 * nspins - 1, nlebedev, ngrids))
    a_sng = rng.normal(size=(nspins, nlebedev, ngrids))

    actual_sng = np.empty_like(a_sng)
    paw_gga_radial_gradient_correction(dr_g=dr_g,
                                       dv_g=dv_g,
                                       dedsigma_xng=dedsigma_xng,
                                       a_sng=a_sng,
                                       vv_sng=actual_sng)

    expected_sng = radial_gradient_correction_reference(
        dr_g, dv_g, dedsigma_xng, a_sng)

    assert cgpaw is not None
    np.testing.assert_allclose(actual_sng, expected_sng,
                               rtol=0.0, atol=2.0e-14)


def back_project_gradient_reference(dEdD_sqL,
                                    dedsigma_xng,
                                    b_vnsg,
                                    n_qg,
                                    weight_n,
                                    rnablaY_nLv):
    nspins = dEdD_sqL.shape[0]
    B_vnsg = dedsigma_xng[::2].transpose(1, 0, 2)[None] * b_vnsg
    if nspins == 2:
        B_vnsg += 0.5 * dedsigma_xng[1][None, :, None] * b_vnsg[:, :, ::-1]
    B_vnsq = np.einsum('vnsg,qg->vnsq', B_vnsg, n_qg, optimize=True)
    dEdD_sqL += 8 * pi * np.einsum(
        'n,nLv,vnsq->sqL', weight_n, rnablaY_nLv, B_vnsq, optimize=True)


@pytest.mark.parametrize('nspins', [1, 2])
def test_paw_gga_back_project_gradient(nspins):
    paw_gga_back_project_gradient = pytest.importorskip(
        'gpaw.cgpaw').paw_gga_back_project_gradient

    rng = np.random.default_rng(175)
    nlebedev = 7
    ngrids = 13
    nq = 5
    Lmax = 4

    dEdD0_sqL = rng.normal(size=(nspins, nq, Lmax))
    dedsigma_xng = rng.normal(size=(2 * nspins - 1, nlebedev, ngrids))
    b_vnsg = rng.normal(size=(3, nlebedev, nspins, ngrids))
    n_qg = rng.normal(size=(nq, ngrids))
    weight_n = rng.random(nlebedev)
    rnablaY_nLv = rng.normal(size=(nlebedev, Lmax, 3))

    actual_sqL = dEdD0_sqL.copy()
    paw_gga_back_project_gradient(dEdD_sqL=actual_sqL,
                                  dedsigma_xng=dedsigma_xng,
                                  b_vnsg=b_vnsg,
                                  n_qg=n_qg,
                                  weight_n=weight_n,
                                  rnablaY_nLv=rnablaY_nLv)

    expected_sqL = dEdD0_sqL.copy()
    back_project_gradient_reference(
        expected_sqL, dedsigma_xng, b_vnsg, n_qg, weight_n, rnablaY_nLv)

    np.testing.assert_allclose(actual_sqL, expected_sqL,
                               rtol=2.0e-14, atol=2.0e-14)


def finish_lebedev_reference(dEdD_sqL,
                             e_ng,
                             dedn_sng,
                             dedsigma_xng,
                             a_sng,
                             b_vnsg,
                             n_qg,
                             dr_g,
                             dv_g,
                             weight_n,
                             Y_nL,
                             rnablaY_nLv):
    vv_sng = radial_gradient_correction_reference(
        dr_g, dv_g, dedsigma_xng, a_sng)
    dEdD_sqL += np.einsum(
        'n,g,sng,qg,nL->sqL',
        weight_n, dv_g, dedn_sng + vv_sng, n_qg, Y_nL, optimize=True)
    scaled_dedsigma_xng = dedsigma_xng * dr_g
    back_project_gradient_reference(
        dEdD_sqL, scaled_dedsigma_xng, b_vnsg, n_qg, weight_n,
        rnablaY_nLv)
    return weight_n @ (e_ng @ dv_g)


@pytest.mark.parametrize('nspins', [1, 2])
def test_paw_gga_finish_lebedev(nspins):
    paw_gga_finish_lebedev = pytest.importorskip(
        'gpaw.cgpaw').paw_gga_finish_lebedev

    rng = np.random.default_rng(693)
    nlebedev = 7
    ngrids = 13
    nq = 5
    Lmax = 4

    dEdD0_sqL = rng.normal(size=(nspins, nq, Lmax))
    e_ng = rng.normal(size=(nlebedev, ngrids))
    dedn_sng = rng.normal(size=(nspins, nlebedev, ngrids))
    dedsigma_xng = rng.normal(size=(2 * nspins - 1, nlebedev, ngrids))
    a_sng = rng.normal(size=(nspins, nlebedev, ngrids))
    b_vnsg = rng.normal(size=(3, nlebedev, nspins, ngrids))
    n_qg = rng.normal(size=(nq, ngrids))
    dr_g = 0.03 + rng.random(ngrids)
    dv_g = 0.02 + rng.random(ngrids)
    weight_n = rng.random(nlebedev)
    Y_nL = rng.normal(size=(nlebedev, Lmax))
    rnablaY_nLv = rng.normal(size=(nlebedev, Lmax, 3))

    actual_sqL = dEdD0_sqL.copy()
    actual_E = paw_gga_finish_lebedev(dEdD_sqL=actual_sqL,
                                      e_ng=e_ng,
                                      dedn_sng=dedn_sng,
                                      dedsigma_xng=dedsigma_xng,
                                      a_sng=a_sng,
                                      b_vnsg=b_vnsg,
                                      n_qg=n_qg,
                                      dr_g=dr_g,
                                      dv_g=dv_g,
                                      weight_n=weight_n,
                                      Y_nL=Y_nL,
                                      rnablaY_nLv=rnablaY_nLv)

    expected_sqL = dEdD0_sqL.copy()
    expected_E = finish_lebedev_reference(
        expected_sqL, e_ng, dedn_sng, dedsigma_xng, a_sng, b_vnsg,
        n_qg, dr_g, dv_g, weight_n, Y_nL, rnablaY_nLv)

    np.testing.assert_allclose(actual_sqL, expected_sqL,
                               rtol=3.0e-14, atol=3.0e-14)
    assert actual_E == pytest.approx(expected_E, abs=2.0e-14)


def fake_gga_kernel(e_ng, n_sng, dedn_sng, sigma_xng, dedsigma_xng):
    e_ng[:] = 0.19 * n_sng[0]
    dedn_sng[:] += 0.11 + 0.23 * n_sng
    dedsigma_xng[:] = 0.07 + 0.13 * sigma_xng
    if len(n_sng) == 2:
        e_ng[:] += 0.17 * n_sng[1] + 0.03 * sigma_xng[1]
        dedn_sng[0] += 0.05 * n_sng[1]
        dedn_sng[1] -= 0.04 * n_sng[0]
    e_ng[:] += 0.02 * sigma_xng.sum(axis=0)


def expansion_reference(kernel,
                        D_sLq,
                        n_qg,
                        nc0_sg,
                        B_pqL,
                        dr_g,
                        dv_g,
                        r_g,
                        Y_nL,
                        weight_n,
                        rnablaY_nLv,
                        dEdD_sp,
                        scale):
    nspins, Lmax, nq = D_sLq.shape
    npacked = B_pqL.shape[0]
    n_sLg = np.einsum('sLq,qg->sLg', D_sLq, n_qg, optimize=True)
    n_sLg[:, 0] += nc0_sg
    dndr_sLg = np.empty_like(n_sLg)
    for s in range(nspins):
        for L in range(Lmax):
            dndr_sLg[s, L] = derivative_reference(dr_g, n_sLg[s, L])
    n_sng, a_sng, b_vnsg, sigma_xng = project_density_reference(
        n_sLg, dndr_sLg, Y_nL, rnablaY_nLv, r_g)
    e_ng = np.empty_like(n_sng[0])
    dedn_sng = np.zeros_like(n_sng)
    dedsigma_xng = np.zeros_like(sigma_xng)
    kernel(e_ng, n_sng, dedn_sng, sigma_xng, dedsigma_xng)
    dEdD_sqL = np.zeros((nspins, nq, Lmax))
    E = finish_lebedev_reference(
        dEdD_sqL, e_ng, dedn_sng, dedsigma_xng, a_sng, b_vnsg, n_qg,
        dr_g, dv_g, weight_n, Y_nL, rnablaY_nLv)
    dEdD_sp += scale * np.inner(
        dEdD_sqL.reshape((nspins, -1)),
        B_pqL.reshape((npacked, -1)))
    return E


@pytest.mark.parametrize('nspins', [1, 2])
def test_paw_gga_single_atom_correction(nspins):
    paw_gga_single_atom_correction = pytest.importorskip(
        'gpaw.cgpaw').paw_gga_single_atom_correction

    rng = np.random.default_rng(902)
    npacked = 6
    nq = 5
    Lmax = 4
    ngrids = 13
    nlebedev = 7

    D_sp = rng.normal(size=(nspins, npacked))
    dEdD0_sp = rng.normal(size=(nspins, npacked))
    B_pqL = rng.normal(size=(npacked, nq, Lmax))
    n_qg = rng.normal(size=(nq, ngrids))
    nt_qg = rng.normal(size=(nq, ngrids))
    nc0_sg = rng.normal(size=(nspins, ngrids))
    nct0_sg = rng.normal(size=(nspins, ngrids))
    dr_g = 0.03 + rng.random(ngrids)
    dv_g = 0.02 + rng.random(ngrids)
    r_g = np.linspace(0.0, 1.4, ngrids)
    Y_nL = rng.normal(size=(nlebedev, Lmax))
    weight_n = rng.random(nlebedev)
    rnablaY_nLv = rng.normal(size=(nlebedev, Lmax, 3))
    e_xc0 = 0.123

    actual_dEdD_sp = dEdD0_sp.copy()
    actual_E = paw_gga_single_atom_correction(
        kernel_calculate=fake_gga_kernel,
        D_sp=D_sp,
        dEdD_sp=actual_dEdD_sp,
        B_pqL=B_pqL,
        n_qg=n_qg,
        nt_qg=nt_qg,
        nc0_sg=nc0_sg,
        nct0_sg=nct0_sg,
        dr_g=dr_g,
        dv_g=dv_g,
        r_g=r_g,
        Y_nL=Y_nL,
        weight_n=weight_n,
        rnablaY_nLv=rnablaY_nLv,
        e_xc0=e_xc0,
        addcoredensity=True)

    expected_dEdD_sp = dEdD0_sp.copy()
    D_sLq = np.inner(D_sp, B_pqL.T)
    E = expansion_reference(fake_gga_kernel, D_sLq, n_qg, nc0_sg, B_pqL,
                            dr_g, dv_g, r_g, Y_nL, weight_n, rnablaY_nLv,
                            expected_dEdD_sp, 1.0)
    Et = expansion_reference(fake_gga_kernel, D_sLq, nt_qg, nct0_sg, B_pqL,
                             dr_g, dv_g, r_g, Y_nL, weight_n, rnablaY_nLv,
                             expected_dEdD_sp, -1.0)
    expected_E = E - Et - e_xc0

    np.testing.assert_allclose(actual_dEdD_sp, expected_dEdD_sp,
                               rtol=5.0e-14, atol=5.0e-14)
    assert actual_E == pytest.approx(expected_E, abs=2.0e-12)


def gga_paw_expansion_numpy_reference(kernel, rgd, D_sLq, n_qg, nc0_sg):
    n_sLg = np.dot(D_sLq, n_qg)
    n_sLg[:, 0] += nc0_sg

    dndr_sLg = np.empty_like(n_sLg)
    for n_Lg, dndr_Lg in zip(n_sLg, dndr_sLg):
        for n_g, dndr_g in zip(n_Lg, dndr_Lg):
            rgd.derivative(n_g, dndr_g)

    nspins, Lmax, nq = D_sLq.shape
    Y_nL1 = Y_nL[:, :Lmax]
    rnablaY_nLv1 = rnablaY_nLv[:, :Lmax]

    n_sng = np.einsum('nL,sLg->sng', Y_nL1, n_sLg, optimize=True)
    a_sng = np.einsum('nL,sLg->sng', Y_nL1, dndr_sLg, optimize=True)
    b_vnsg = np.einsum('nLv,sLg->vnsg', rnablaY_nLv1, n_sLg,
                        optimize=True)

    nlebedev, ngrids = Y_nL1.shape[0], n_sLg.shape[-1]
    sigma_xng = np.empty((2 * nspins - 1, nlebedev, ngrids))
    sigma_xng[::2] = (b_vnsg ** 2).sum(0).transpose(1, 0, 2)
    if nspins == 2:
        sigma_xng[1] = (b_vnsg[:, :, 0] * b_vnsg[:, :, 1]).sum(0)
    sigma_xng[:, :, 1:] /= rgd.r_g[1:] ** 2
    sigma_xng[:, :, 0] = sigma_xng[:, :, 1]
    sigma_xng[::2] += a_sng ** 2
    if nspins == 2:
        sigma_xng[1] += a_sng[0] * a_sng[1]

    n_sg = n_sng.reshape(nspins, nlebedev * ngrids)
    sigma_xg = sigma_xng.reshape(2 * nspins - 1, nlebedev * ngrids)
    e_g = np.empty(nlebedev * ngrids)
    dedn_sg = np.zeros((nspins, nlebedev * ngrids))
    dedsigma_xg = np.zeros((2 * nspins - 1, nlebedev * ngrids))
    kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)

    e_ng = e_g.reshape(nlebedev, ngrids)
    dedn_sng = dedn_sg.reshape(nspins, nlebedev, ngrids)
    dedsigma_xng = dedsigma_xg.reshape(2 * nspins - 1, nlebedev, ngrids)
    vv_sng = radial_gradient_correction_reference(
        rgd.dr_g, rgd.dv_g, dedsigma_xng, a_sng)

    dEdD_sqL = np.einsum(
        'n,g,sng,qg,nL->sqL',
        weight_n, rgd.dv_g, dedn_sng + vv_sng, n_qg, Y_nL1,
        optimize=True)
    back_project_gradient_reference(
        dEdD_sqL, dedsigma_xng * rgd.dr_g, b_vnsg, n_qg, weight_n,
        rnablaY_nLv1)
    E = weight_n @ rgd.integrate(e_ng)
    return E, dEdD_sqL


def calculate_paw_correction_numpy_reference(xc, setup, D_sp, dEdD_sp):
    xcc = setup.xc_correction
    rgd = xcc.rgd
    nspins = len(D_sp)

    nc0_sg = rgd.empty(nspins)
    nct0_sg = rgd.empty(nspins)
    nc0_sg[:] = sqrt(4 * pi) / nspins * xcc.nc_g
    nct0_sg[:] = sqrt(4 * pi) / nspins * xcc.nct_g
    if xcc.nc_corehole_g is not None and nspins == 2:
        nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
        nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g

    D_sLq = np.inner(D_sp, xcc.B_pqL.T)
    e, dEdD_sqL = gga_paw_expansion_numpy_reference(
        xc.kernel, rgd, D_sLq, xcc.n_qg, nc0_sg)
    et, dEtdD_sqL = gga_paw_expansion_numpy_reference(
        xc.kernel, rgd, D_sLq, xcc.nt_qg, nct0_sg)

    dEdD_sp += np.inner(
        (dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
        xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))
    return e - et - xcc.e_xc0


def best_time(function, repeats=5):
    times = []
    result = None
    for _ in range(repeats):
        t0 = perf_counter()
        result = function()
        times.append(perf_counter() - t0)
    return min(times), result


@pytest.mark.serial
def test_paw_gga_c_single_atom_speedup_vs_numpy(monkeypatch):
    pytest.importorskip('gpaw.cgpaw')
    from gpaw.setup import create_setup
    from gpaw.xc import XC

    monkeypatch.setenv('GPAW_BATCH_LEBEDEV_PAW', '1')
    xc = XC('PBE')
    setup = create_setup('Si', xc)
    rng = np.random.default_rng(617)
    D_sp = np.ascontiguousarray(
        0.2 + 0.1 * rng.random((2, setup.ni * (setup.ni + 1) // 2)))

    def numpy_call():
        dEdD_sp = np.zeros_like(D_sp)
        e = calculate_paw_correction_numpy_reference(
            xc, setup, D_sp, dEdD_sp)
        return e, dEdD_sp

    def c_call():
        dEdD_sp = np.zeros_like(D_sp)
        e = xc.calculate_paw_correction(setup, D_sp, dEdD_sp)
        return e, dEdD_sp

    numpy_time, numpy_result = best_time(numpy_call)
    c_time, c_result = best_time(c_call)

    np.testing.assert_allclose(c_result[0], numpy_result[0],
                               rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(c_result[1], numpy_result[1],
                               rtol=2.0e-12, atol=2.0e-12)

    speedup = numpy_time / c_time
    assert speedup > 1.25, (
        f'C PAW correction should be faster than NumPy reference; '
        f'numpy={numpy_time:.6f}s c={c_time:.6f}s speedup={speedup:.2f}x')
