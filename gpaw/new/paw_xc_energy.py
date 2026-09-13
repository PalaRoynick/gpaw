"""PAW XC energies expressed as ordinary NumPy programs.

The functions return energy only.  The decorator marks active inputs so that
Meinsum can generate the PAW Hamiltonian contribution by reverse automatic
differentiation; direct calls remain ordinary NumPy evaluation.
"""

import numpy as np

import meinsum


def radial_derivative(values, spacing):
    """Differentiate radial samples, omitting the quadrature origin."""
    values_without_origin = values[..., 1:]
    derivative = np.empty_like(values_without_origin)
    derivative[..., :-1] = 0.5 * (
        values[..., 2:] - values[..., :-2]) / spacing[1:-1]
    derivative[..., -1] = (
        values[..., -1] - values[..., -2]) / spacing[-1]
    return derivative


@meinsum.compile
def spin_paired_pbe_energy(
        density: meinsum.scalar_w_gradient,
        sigma: meinsum.scalar_w_gradient) -> meinsum.scalartype:
    """Spin-paired PBE energy density used by GPAW's native kernel."""
    density = np.maximum(1.0e-10, density)
    rs = (0.238732414637843 / density) ** (1.0 / 3.0)
    sqrt_rs = rs ** 0.5

    exchange = -0.45816529328314287 / rs
    exchange_scale = 0.26053088059892404 * rs / density
    reduced_gradient = sigma * exchange_scale * exchange_scale
    exchange_factor = (
        1.804 - 0.804 /
        (1.0 + 0.2195164512208958 * reduced_gradient / 0.804))
    exchange = exchange * exchange_factor

    q0 = -0.062182 * (1.0 + 0.21370 * rs)
    q1 = (0.062182 * sqrt_rs *
          (7.5957 + sqrt_rs *
           (3.5876 + sqrt_rs *
            (1.6382 + 0.49294 * sqrt_rs))))
    correlation = q0 * np.log(1.0 + 1.0 / q1)

    density2 = density * density
    t2 = 0.10231023756535741 * sigma * rs / density2
    y = -correlation / 0.031091
    exponential = np.exp(y)
    coefficient = 0.066725 / (0.031091 * (exponential - 1.0))
    coefficient_t2 = coefficient * t2
    numerator = 1.0 + coefficient_t2
    denominator = numerator + coefficient_t2 * coefficient_t2
    correction = (
        0.031091 *
        np.log(1.0 + 0.066725 * t2 * numerator /
               (denominator * 0.031091)))
    correlation = correlation + correction
    return density * (exchange + correlation)


@meinsum.compile
def spin_paired_pbe_paw_energy(
        density_matrix_p: meinsum.tensor_w_gradient,
        B_pqL: meinsum.tensortype,
        n_qg: meinsum.tensortype,
        nt_qg: meinsum.tensortype,
        nc_g: meinsum.tensortype,
        nct_g: meinsum.tensortype,
        Y_nL: meinsum.tensortype,
        rnablaY_nLv: meinsum.tensortype,
        dr_g: meinsum.tensortype,
        inverse_r2_g: meinsum.tensortype,
        weight_n: meinsum.tensortype,
        dv_g: meinsum.tensortype,
        e_xc0: meinsum.scalartype) -> meinsum.scalartype:
    """Spin-paired PBE PAW correction as one energy-only NumPy graph."""
    D_Lq = np.einsum('p,pqL->Lq', density_matrix_p, B_pqL)
    inverse_r2_eval_g = inverse_r2_g[1:]
    dv_eval_g = dv_g[1:]

    n_Lg = np.einsum('Lq,qg->Lg', D_Lq, n_qg)
    core_g = np.sqrt(4.0 * np.pi) * nc_g
    n_Lg[0] += core_g
    dndr_Lg = radial_derivative(n_Lg, dr_g)
    density_ng = np.einsum('nL,Lg->ng', Y_nL, n_Lg)
    density_ng = density_ng[..., 1:]
    radial_ng = np.einsum('nL,Lg->ng', Y_nL, dndr_Lg)
    tangent_nvg = np.einsum('nLv,Lg->nvg', rnablaY_nLv, n_Lg)
    tangent2_ng = np.einsum(
        'nvg,nvg->ng', tangent_nvg, tangent_nvg)
    tangent2_ng = tangent2_ng[..., 1:] * inverse_r2_eval_g
    sigma_ng = radial_ng * radial_ng + tangent2_ng
    energy_ng = spin_paired_pbe_energy(density_ng, sigma_ng)
    energy_ae = np.einsum('n,ng,g->', weight_n, energy_ng, dv_eval_g)

    nt_Lg = np.einsum('Lq,qg->Lg', D_Lq, nt_qg)
    coret_g = np.sqrt(4.0 * np.pi) * nct_g
    nt_Lg[0] += coret_g
    dndrt_Lg = radial_derivative(nt_Lg, dr_g)
    densityt_ng = np.einsum('nL,Lg->ng', Y_nL, nt_Lg)
    densityt_ng = densityt_ng[..., 1:]
    radialt_ng = np.einsum('nL,Lg->ng', Y_nL, dndrt_Lg)
    tangentt_nvg = np.einsum('nLv,Lg->nvg', rnablaY_nLv, nt_Lg)
    tangent2t_ng = np.einsum(
        'nvg,nvg->ng', tangentt_nvg, tangentt_nvg)
    tangent2t_ng = tangent2t_ng[..., 1:] * inverse_r2_eval_g
    sigmat_ng = radialt_ng * radialt_ng + tangent2t_ng
    energyt_ng = spin_paired_pbe_energy(densityt_ng, sigmat_ng)
    energy_ps = np.einsum('n,ng,g->', weight_n, energyt_ng, dv_eval_g)
    energy = energy_ae - energy_ps
    correction = energy - e_xc0
    return correction
