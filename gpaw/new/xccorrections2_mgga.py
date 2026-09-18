"""MGGA extension of the eager NumPy/CuPy PAW result-graph experiment.

MGGA is kept separate because GPAW currently has no GPU MGGA point evaluator.
The tensor graph and reverse pass are backend-independent, while callers must
provide a function implementing GPAW's density, sigma, and tau kernel
contract.  TPSS is used by the tests as the representative CPU functional.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np

from gpaw.new.xccorrections2 import (
    PBEXCCorrection,
    _integrate_correction,
    density_L_expansion,
    paired_density_and_grad,
    polarized_density_and_grad,
)
from gpaw.utilities.tensor_autograd import Engine, TensorResult


class MGGAResult(TensorResult):
    """MGGA energy with density, sigma, and tau reverse derivatives."""

    def __init__(self,
                 density: TensorResult,
                 sigma: TensorResult,
                 tau: TensorResult,
                 function):
        xp = density.engine.xp
        spin_axis = (density.indices.index('s')
                     if 's' in density.indices else None)
        sigma_axis = (sigma.indices.index('x')
                      if 'x' in sigma.indices else None)
        density_sx = (xp.ascontiguousarray(density.value[None])
                      if spin_axis is None else
                      xp.ascontiguousarray(xp.moveaxis(
                          density.value, spin_axis, 0)))
        sigma_xr = (xp.ascontiguousarray(sigma.value[None])
                    if sigma_axis is None else
                    xp.ascontiguousarray(xp.moveaxis(
                        sigma.value, sigma_axis, 0)))
        tau_sx = (xp.ascontiguousarray(tau.value[None])
                  if spin_axis is None else
                  xp.ascontiguousarray(xp.moveaxis(
                      tau.value, spin_axis, 0)))
        potential_sx = xp.zeros_like(density_sx)
        sigma_derivative_xr = xp.empty_like(sigma_xr)
        tau_derivative_sx = xp.empty_like(tau_sx)
        energy_r = xp.empty(density_sx.shape[1:], density.value.dtype)
        function(density_sx, potential_sx, energy_r,
                 sigma_xr, sigma_derivative_xr,
                 tau_sx, tau_derivative_sx)
        super().__init__(
            density.engine, energy_r,
            density.indices.replace('s', '', 1))
        self.density = density
        self.sigma = sigma
        self.tau = tau
        self.spin_axis = spin_axis
        self.sigma_axis = sigma_axis
        self.potential = (potential_sx[0] if spin_axis is None else
                          xp.moveaxis(potential_sx, 0, spin_axis))
        self.sigma_derivative = (
            sigma_derivative_xr[0] if sigma_axis is None else
            xp.moveaxis(sigma_derivative_xr, 0, sigma_axis))
        self.tau_derivative = (
            tau_derivative_sx[0] if spin_axis is None else
            xp.moveaxis(tau_derivative_sx, 0, spin_axis))
        self.parents = (density, sigma, tau)

    def _backward(self, gradient):
        """Return contributions for density, sigma, and tau parents."""
        density_gradient = gradient
        tau_gradient = gradient
        if self.spin_axis is not None:
            density_gradient = self.engine.xp.expand_dims(
                density_gradient, self.spin_axis)
            tau_gradient = self.engine.xp.expand_dims(
                tau_gradient, self.spin_axis)
        sigma_gradient = gradient
        if self.sigma_axis is not None:
            sigma_gradient = self.engine.xp.expand_dims(
                sigma_gradient, self.sigma_axis)
        return (
            (self.density, self.potential * density_gradient),
            (self.sigma, self.sigma_derivative * sigma_gradient),
            (self.tau, self.tau_derivative * tau_gradient))


def _paired_tau(engine: Engine,
                density_ap: TensorResult,
                kinetic_anpg,
                core_kinetic_ag) -> TensorResult:
    """Reconstruct the paired kinetic-energy density on points ``h=g[1:]``."""
    # tau_ang[a, n, g] = D_ap[a, p] * tau_anpg[a, n, p, g]
    tau_ang = engine.einsum('ap,anpg->ang', density_ap, kinetic_anpg)
    tau_ang.value += (core_kinetic_ag[:, None, :] /
                      np.sqrt(4.0 * np.pi))
    return tau_ang.slice((..., slice(1, None)), 'anh')


def _polarized_tau(engine: Engine,
                   density_asp: TensorResult,
                   kinetic_anpg,
                   core_kinetic_ag) -> TensorResult:
    """Reconstruct both spin kinetic-energy densities on ``h=g[1:]``."""
    # tau_asng[a, s, n, g] = D_asp[a, s, p] * tau_anpg[a, n, p, g]
    tau_asng = engine.einsum('asp,anpg->asng', density_asp, kinetic_anpg)
    tau_asng.value += (0.5 * core_kinetic_ag[:, None, None, :] /
                       np.sqrt(4.0 * np.pi))
    return tau_asng.slice((..., slice(1, None)), 'asnh')


@dataclass(frozen=True)
class MGGAXCCorrection(PBEXCCorrection):
    """Kinetic-energy-density arrays additionally required by MGGA."""

    ae_kinetic_anpg: object
    pseudo_kinetic_anpg: object
    ae_core_kinetic_ag: object
    pseudo_core_kinetic_ag: object


@dataclass(frozen=True)
class SpinPairedMGGAXCCorrection(MGGAXCCorrection):
    """Prepared atom batch for a spin-paired MGGA PAW correction."""

    def evaluate(self, engine: Engine, evaluate_mgga):
        # D_ap is the packed density matrix for every atom.
        density_ap = engine.requires_gradient(
            self.density, self.gradient, 'ap')
        # D_aLq contains projector-pair weights in angular/radial channels.
        D_aLq = engine.einsum(
            'ap,apqL->aLq', density_ap, self.basis_apqL)
        ae_aLg = density_L_expansion(
            engine, D_aLq, self.ae_radial_aqg, self.nc_ag)
        pseudo_aLg = density_L_expansion(
            engine, D_aLq, self.pseudo_radial_aqg, self.nct_ag)
        ae_density_anh, ae_sigma_anh = paired_density_and_grad(
            engine, ae_aLg, self.angular_anL, self.angular_gradient_anLv,
            self.radial_spacing_ag, self.inverse_radius2_ag)
        pseudo_density_anh, pseudo_sigma_anh = paired_density_and_grad(
            engine, pseudo_aLg, self.angular_anL,
            self.angular_gradient_anLv, self.radial_spacing_ag,
            self.inverse_radius2_ag)
        ae_tau = _paired_tau(
            engine, density_ap, self.ae_kinetic_anpg,
            self.ae_core_kinetic_ag)
        pseudo_tau = _paired_tau(
            engine, density_ap, self.pseudo_kinetic_anpg,
            self.pseudo_core_kinetic_ag)
        energy_a = _integrate_correction(
            engine,
            MGGAResult(
                ae_density_anh, ae_sigma_anh, ae_tau, evaluate_mgga),
            MGGAResult(
                pseudo_density_anh, pseudo_sigma_anh,
                pseudo_tau, evaluate_mgga),
            self.angular_weight_an, self.radial_weight_ag,
            self.reference_energy_a, True)
        energy_a.backward()
        return energy_a.value, self.gradient


@dataclass(frozen=True)
class SpinPolarizedMGGAXCCorrection(MGGAXCCorrection):
    """Prepared atom batch for a spin-polarized MGGA PAW correction."""

    def evaluate(self, engine: Engine, evaluate_mgga):
        # D_asp is the packed density matrix for every atom and spin.
        density_asp = engine.requires_gradient(
            self.density, self.gradient, 'asp')
        # D_asLq contains spin-resolved angular/radial channel weights.
        D_asLq = engine.einsum(
            'asp,apqL->asLq', density_asp, self.basis_apqL)
        ae_asLg = density_L_expansion(
            engine, D_asLq, self.ae_radial_aqg, self.nc_ag)
        pseudo_asLg = density_L_expansion(
            engine, D_asLq, self.pseudo_radial_aqg, self.nct_ag)
        ae_density_asnh, ae_sigma_axnh = polarized_density_and_grad(
            engine, ae_asLg, self.angular_anL, self.angular_gradient_anLv,
            self.radial_spacing_ag, self.inverse_radius2_ag)
        pseudo_density_asnh, pseudo_sigma_axnh = polarized_density_and_grad(
            engine, pseudo_asLg, self.angular_anL,
            self.angular_gradient_anLv, self.radial_spacing_ag,
            self.inverse_radius2_ag)
        ae_tau = _polarized_tau(
            engine, density_asp, self.ae_kinetic_anpg,
            self.ae_core_kinetic_ag)
        pseudo_tau = _polarized_tau(
            engine, density_asp, self.pseudo_kinetic_anpg,
            self.pseudo_core_kinetic_ag)
        energy_a = _integrate_correction(
            engine,
            MGGAResult(
                ae_density_asnh, ae_sigma_axnh, ae_tau, evaluate_mgga),
            MGGAResult(
                pseudo_density_asnh, pseudo_sigma_axnh,
                pseudo_tau, evaluate_mgga),
            self.angular_weight_an, self.radial_weight_ag,
            self.reference_energy_a, True)
        energy_a.backward()
        return energy_a.value, self.gradient


def _evaluate_mgga_cpu(functional,
                       density_sx,
                       potential_sx,
                       energy_x,
                       sigma_xr,
                       sigma_derivative_xr,
                       tau_sx,
                       tau_derivative_sx):
    """Evaluate GPAW's scalar MGGA kernel on one batch of radial points."""
    functional.xc.kernel.calculate(
        energy_x, density_sx, potential_sx, sigma_xr,
        sigma_derivative_xr, tau_sx, tau_derivative_sx)


def _evaluate_mgga_gpu(functional, density_sx, potential_sx, energy_x,
                       sigma_xr, sigma_derivative_xr, tau_sx,
                       tau_derivative_sx):
    """Evaluate the CUDA Libxc MGGA kernel on CuPy arrays."""
    calculate_gpu = getattr(functional.xc.kernel, 'calculate_gpu', None)
    if calculate_gpu is None:
        raise RuntimeError('GPU MGGA kernel is not available')
    calculate_gpu(energy_x, density_sx, potential_sx, sigma_xr,
                  sigma_derivative_xr, tau_sx, tau_derivative_sx)


def evaluate_paw_mgga(functional, batch):
    """Evaluate one batched spin-paired or spin-polarized MGGA correction."""
    xp = functional.xp
    polarized = batch.density_asp.shape[1] == 2
    density = (batch.density_asp if polarized else
               batch.density_asp[:, 0])
    gradient = xp.zeros_like(density)
    arguments = dict(
        density=density,
        basis_apqL=batch.basis_apqL,
        ae_radial_aqg=batch.ae_radial_aqg,
        pseudo_radial_aqg=batch.pseudo_radial_aqg,
        nc_ag=batch.nc_ag,
        nct_ag=batch.nct_ag,
        angular_anL=batch.angular_anL,
        angular_weight_an=batch.angular_weight_an,
        radial_weight_ag=batch.radial_weight_ag,
        reference_energy_a=batch.reference_energy_a,
        gradient=gradient,
        angular_gradient_anLv=batch.angular_gradient_anLv,
        radial_spacing_ag=batch.radial_spacing_ag,
        inverse_radius2_ag=batch.inverse_radius2_ag,
        ae_kinetic_anpg=batch.ae_kinetic_anpg,
        pseudo_kinetic_anpg=batch.pseudo_kinetic_anpg,
        ae_core_kinetic_ag=batch.ae_core_kinetic_ag,
        pseudo_core_kinetic_ag=batch.pseudo_core_kinetic_ag)
    correction = (SpinPolarizedMGGAXCCorrection if polarized else
                  SpinPairedMGGAXCCorrection)(**arguments)
    evaluate = (partial(_evaluate_mgga_cpu, functional)
                if xp is np else
                partial(_evaluate_mgga_gpu, functional))
    energies, gradients = correction.evaluate(Engine(xp), evaluate)
    if gradients.ndim == 2:
        gradients = gradients[:, None, :]
    return energies, gradients
