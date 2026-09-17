"""Spin-paired and spin-polarized LDA/PBE PAW correction programs.

The four correction classes express the physical calculation as readable
tensor operations.  Generic differentiation and batching live in
``gpaw.utilities.tensor_autograd`` and ``gpaw.utilities.tensor_batch``.
GPAW's existing point evaluators supply local XC energies and derivatives.

Array suffixes document tensor indices throughout the module:

``a`` atom, ``s`` spin, ``p`` packed density-matrix element,
``q`` radial basis function, ``L`` spherical harmonic, ``n`` angular point,
``v`` Cartesian direction, ``g`` radial point, and ``x`` a packed spin pair
(aa, ab, bb).

This experimental implementation deliberately excludes MGGA.  Unsupported
functionals and density layouts continue through GPAW's established PAW
correction implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from gpaw.new.c import evaluate_lda_gpu, evaluate_pbe_gpu
from gpaw.utilities.tensor_batch import PAWXCBatch
from gpaw.utilities.tensor_autograd import (
    EinsumResult,
    Engine,
    TensorResult,
)


class LDAResult(TensorResult):
    """LDA energy and the potential needed by its reverse rule."""

    def __init__(self, density: TensorResult, function):
        xp = density.engine.xp
        spin_axis = (density.indices.index('s')
                     if 's' in density.indices else None)
        density_sx = (xp.ascontiguousarray(density.value[None])
                      if spin_axis is None else
                      xp.ascontiguousarray(xp.moveaxis(
                          density.value, spin_axis, 0)))
        potential_sx = xp.zeros_like(density_sx)
        energy_x = xp.empty(density_sx.shape[1:], density.value.dtype)
        function(density_sx, potential_sx, energy_x)
        super().__init__(
            density.engine, energy_x,
            density.indices.replace('s', '', 1))
        self.density = density
        self.spin_axis = spin_axis
        self.potential = (potential_sx[0] if spin_axis is None else
                          xp.moveaxis(potential_sx, 0, spin_axis))
        self.parents = (density,)

    def _backward(self, gradient):
        if self.spin_axis is not None:
            gradient = self.engine.xp.expand_dims(gradient, self.spin_axis)
        return ((self.density, self.potential * gradient),)


class PBEResult(TensorResult):
    """PBE energy and its density and sigma partial derivatives."""

    def __init__(self,
                 density: TensorResult,
                 sigma: TensorResult,
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
        potential_sx = xp.zeros_like(density_sx)
        sigma_derivative_xr = xp.empty_like(sigma_xr)
        energy_r = xp.empty(density_sx.shape[1:], density.value.dtype)
        function(density_sx, potential_sx, energy_r,
                 sigma_xr, sigma_derivative_xr)
        super().__init__(
            density.engine, energy_r,
            density.indices.replace('s', '', 1))
        self.density = density
        self.sigma = sigma
        self.spin_axis = spin_axis
        self.sigma_axis = sigma_axis
        self.potential = (potential_sx[0] if spin_axis is None else
                          xp.moveaxis(potential_sx, 0, spin_axis))
        self.sigma_derivative = (
            sigma_derivative_xr[0] if sigma_axis is None else
            xp.moveaxis(sigma_derivative_xr, 0, sigma_axis))
        self.parents = (density, sigma)

    def _backward(self, gradient):
        density_gradient = gradient
        if self.spin_axis is not None:
            density_gradient = self.engine.xp.expand_dims(
                density_gradient, self.spin_axis)
        sigma_gradient = gradient
        if self.sigma_axis is not None:
            sigma_gradient = self.engine.xp.expand_dims(
                sigma_gradient, self.sigma_axis)
        return (
            (self.density, self.potential * density_gradient),
            (self.sigma, self.sigma_derivative * sigma_gradient))


def density_L_expansion(engine: Engine,
                        D: TensorResult,
                        radial_aqg,
                        core_ag) -> EinsumResult:
    """Expand paired or spin-resolved coefficients into density multipoles."""
    if 's' in D.indices:
        # density_asLg[a, s, L, g] = D_asLq * radial_aqg.
        density_asLg = engine.einsum('asLq,aqg->asLg', D, radial_aqg)
        density_asLg[:, :, 0] += (
            0.5 * np.sqrt(4.0 * np.pi) * core_ag[:, None, :])
        return density_asLg
    # density_aLg[a, L, g] = D_aLq * radial_aqg.
    density_aLg = engine.einsum('aLq,aqg->aLg', D, radial_aqg)
    density_aLg[:, 0] += np.sqrt(4.0 * np.pi) * core_ag
    return density_aLg


def paired_density_and_grad(engine: Engine,
                            density_aLg: TensorResult,
                            angular_anL,
                            angular_gradient_anLv,
                            radial_spacing_ag,
                            inverse_radius2_ag
                            ) -> tuple[TensorResult, TensorResult]:
    """Construct paired density and sigma on the full PAW radial grid."""
    # density_ang is n(r) at every atom, angular point, and radial point.
    density_ang = engine.einsum(
        'anL,aLg->ang', angular_anL, density_aLg)
    derivative_aLg = engine.radial_derivative(
        density_aLg, radial_spacing_ag)
    # radial_ang is the radial component of grad(n).
    radial_ang = engine.einsum(
        'anL,aLg->ang', angular_anL, derivative_aLg)
    tangent_anvg = engine.einsum(
        'anLv,aLg->anvg', angular_gradient_anLv, density_aLg)
    # tangent2_ang is the squared angular component before division by r**2.
    tangent2_ang = engine.sum(tangent_anvg * tangent_anvg, axis=-2)
    inverse_radius2_a1g = inverse_radius2_ag[:, None, :]
    sigma_ang = (radial_ang * radial_ang +
                 tangent2_ang * inverse_radius2_a1g)
    return density_ang, sigma_ang


def polarized_density_and_grad(engine: Engine,
                               density_asLg: TensorResult,
                               angular_anL,
                               angular_gradient_anLv,
                               radial_spacing_ag,
                               inverse_radius2_ag
                               ) -> tuple[TensorResult, TensorResult]:
    """Construct two densities and aa/ab/bb sigma on the full radial grid."""
    # density_asng contains n_s(r) for both spins.
    density_asng = engine.einsum(
        'anL,asLg->asng', angular_anL, density_asLg)
    derivative_asLg = engine.radial_derivative(
        density_asLg, radial_spacing_ag)
    # radial_asng contains the radial gradient component for both spins.
    radial_asng = engine.einsum(
        'anL,asLg->asng', angular_anL, derivative_asLg)
    tangent_asnvg = engine.einsum(
        'anLv,asLg->asnvg', angular_gradient_anLv, density_asLg)

    radial_a_ang = radial_asng.slice(
        (slice(None), 0, slice(None), slice(None)), 'ang')
    radial_b_ang = radial_asng.slice(
        (slice(None), 1, slice(None), slice(None)), 'ang')
    tangent_a_anvg = tangent_asnvg.slice(
        (slice(None), 0, slice(None), slice(None), slice(None)), 'anvg')
    tangent_b_anvg = tangent_asnvg.slice(
        (slice(None), 1, slice(None), slice(None), slice(None)), 'anvg')

    inverse_radius2_a1g = inverse_radius2_ag[:, None, :]

    def sigma(radial_left, radial_right, tangent_left, tangent_right):
        """Form one grad(n_left) dot grad(n_right) field."""
        tangent2_ang = engine.sum(
            tangent_left * tangent_right, axis=-2)
        return (radial_left * radial_right +
                tangent2_ang * inverse_radius2_a1g)

    sigma_aa_ang = sigma(
        radial_a_ang, radial_a_ang, tangent_a_anvg, tangent_a_anvg)
    sigma_ab_ang = sigma(
        radial_a_ang, radial_b_ang, tangent_a_anvg, tangent_b_anvg)
    sigma_bb_ang = sigma(
        radial_b_ang, radial_b_ang, tangent_b_anvg, tangent_b_anvg)
    sigma_axng = engine.stack(
        (sigma_aa_ang, sigma_ab_ang, sigma_bb_ang), 1, 'axng')
    return density_asng, sigma_axng


def _integrate_correction(engine: Engine,
                          ae_energy: TensorResult,
                          pseudo_energy: TensorResult,
                          angular_weight_an,
                          radial_weight_ag,
                          reference_energy_a) -> TensorResult:
    """Integrate all-electron minus pseudo energy and subtract ``e_xc0``."""
    quadrature = (angular_weight_an[:, :, None] *
                  radial_weight_ag[:, None, :])
    energy_a = engine.sum(
        (ae_energy - pseudo_energy) * quadrature, axis=(-2, -1))
    energy_a -= reference_energy_a
    return energy_a


@dataclass(frozen=True)
class PAWXCCorrection:
    """Setup and atom arrays shared by all four PAW XC corrections."""

    # D_ap or D_asp: packed on-site density matrices.
    density: Any
    # B_apqL maps packed matrices to radial/angular coefficients.
    basis_apqL: Any
    # All-electron and pseudo radial basis products n_aqg and nt_aqg.
    ae_radial_aqg: Any
    pseudo_radial_aqg: Any
    # Frozen all-electron and pseudo core densities nc_ag and nct_ag.
    nc_ag: Any
    nct_ag: Any
    # Spherical harmonics Y_anL on the angular quadrature.
    angular_anL: Any
    # Angular and radial quadrature weights weight_an and dv_ag.
    angular_weight_an: Any
    radial_weight_ag: Any
    # Atomic reference correction e_xc0_a.
    reference_energy_a: Any
    # Destination dE/dD_ap or dE/dD_asp.
    gradient: Any

    def evaluate(self, engine: Engine, function) -> tuple[Any, Any]:
        """Return per-atom energies and fill the density-matrix gradient."""
        raise NotImplementedError


@dataclass(frozen=True)
class SpinPairedLDAXCCorrection(PAWXCCorrection):
    """Prepared atom batch for the spin-paired LDA PAW correction."""

    def evaluate(self, engine: Engine, evaluate_lda):
        """Return per-atom LDA energies and fill ``gradient`` with dE/dD."""
        # density_ap is the packed density matrix D_p for every atom.
        density_ap = engine.requires_gradient(
            self.density, self.gradient, 'ap')
        # basis_apqL maps packed p to radial q and angular L channels.
        # D_aLq contains the resulting density-expansion coefficients.
        D_aLq = engine.einsum(
            'ap,apqL->aLq', density_ap, self.basis_apqL)
        # ae_aLg and pseudo_aLg are all-electron/pseudo radial multipoles.
        ae_aLg = density_L_expansion(
            engine, D_aLq, self.ae_radial_aqg, self.nc_ag)
        pseudo_aLg = density_L_expansion(
            engine, D_aLq, self.pseudo_radial_aqg, self.nct_ag)
        # ae_ang and pseudo_ang are physical densities n(r) on the quadrature.
        ae_ang = engine.einsum('anL,aLg->ang', self.angular_anL, ae_aLg)
        pseudo_ang = engine.einsum(
            'anL,aLg->ang', self.angular_anL, pseudo_aLg)
        energy_a = _integrate_correction(
            engine,
            LDAResult(ae_ang, evaluate_lda),
            LDAResult(pseudo_ang, evaluate_lda),
            self.angular_weight_an, self.radial_weight_ag,
            self.reference_energy_a)
        energy_a.backward()
        return energy_a.value, self.gradient


@dataclass(frozen=True)
class SpinPolarizedLDAXCCorrection(PAWXCCorrection):
    """Prepared atom batch for the spin-polarized LDA PAW correction."""

    def evaluate(self, engine: Engine, evaluate_lda):
        """Return per-atom polarized LDA energies and fill dE/dD_sp."""
        # density_asp is the packed density matrix D_sp for each atom/spin.
        density_asp = engine.requires_gradient(
            self.density, self.gradient, 'asp')
        # D_asLq holds spin-resolved density-expansion coefficients.
        D_asLq = engine.einsum(
            'asp,apqL->asLq', density_asp, self.basis_apqL)
        ae_asLg = density_L_expansion(
            engine, D_asLq, self.ae_radial_aqg, self.nc_ag)
        pseudo_asLg = density_L_expansion(
            engine, D_asLq, self.pseudo_radial_aqg, self.nct_ag)
        # ae_asng and pseudo_asng are the two spin densities on the grid.
        ae_asng = engine.einsum(
            'anL,asLg->asng', self.angular_anL, ae_asLg)
        pseudo_asng = engine.einsum(
            'anL,asLg->asng', self.angular_anL, pseudo_asLg)
        energy_a = _integrate_correction(
            engine,
            LDAResult(ae_asng, evaluate_lda),
            LDAResult(pseudo_asng, evaluate_lda),
            self.angular_weight_an, self.radial_weight_ag,
            self.reference_energy_a)
        energy_a.backward()
        return energy_a.value, self.gradient


@dataclass(frozen=True)
class PBEXCCorrection(PAWXCCorrection):
    """Additional radial and angular arrays needed by PBE."""

    # r * grad(Y_anL), with Cartesian component v.
    angular_gradient_anLv: object
    # Radial grid spacing dr_ag used by the finite-difference stencil.
    radial_spacing_ag: object
    # 1 / r_ag**2, with zero at the zero-weight origin.
    inverse_radius2_ag: object


@dataclass(frozen=True)
class SpinPairedPBEXCCorrection(PBEXCCorrection):
    """Prepared atom batch for the spin-paired PBE PAW correction."""

    def evaluate(self, engine: Engine, evaluate_pbe):
        """Return paired PBE energies and fill the packed-matrix gradient."""
        # density_ap is the packed density matrix D_p for every atom.
        density_ap = engine.requires_gradient(
            self.density, self.gradient, 'ap')
        # D_aLq holds its angular/radial density-expansion coefficients.
        D_aLq = engine.einsum(
            'ap,apqL->aLq', density_ap, self.basis_apqL)
        ae_aLg = density_L_expansion(
            engine, D_aLq, self.ae_radial_aqg, self.nc_ag)
        pseudo_aLg = density_L_expansion(
            engine, D_aLq, self.pseudo_radial_aqg, self.nct_ag)
        # Each field pair is (density_ang, sigma_ang) on the full grid.
        ae_density_ang, ae_sigma_ang = paired_density_and_grad(
            engine, ae_aLg, self.angular_anL, self.angular_gradient_anLv,
            self.radial_spacing_ag, self.inverse_radius2_ag)
        pseudo_density_ang, pseudo_sigma_ang = paired_density_and_grad(
            engine, pseudo_aLg, self.angular_anL,
            self.angular_gradient_anLv, self.radial_spacing_ag,
            self.inverse_radius2_ag)
        energy_a = _integrate_correction(
            engine,
            PBEResult(ae_density_ang, ae_sigma_ang, evaluate_pbe),
            PBEResult(
                pseudo_density_ang, pseudo_sigma_ang, evaluate_pbe),
            self.angular_weight_an, self.radial_weight_ag,
            self.reference_energy_a)
        energy_a.backward()
        return energy_a.value, self.gradient


@dataclass(frozen=True)
class SpinPolarizedPBEXCCorrection(PBEXCCorrection):
    """Prepared atom batch for the spin-polarized PBE PAW correction."""

    def evaluate(self, engine: Engine, evaluate_pbe):
        """Return polarized PBE energies and fill dE/dD_asp."""
        # density_asp is the packed density matrix D_sp for each atom/spin.
        density_asp = engine.requires_gradient(
            self.density, self.gradient, 'asp')
        # D_asLq holds spin-resolved density-expansion coefficients.
        D_asLq = engine.einsum(
            'asp,apqL->asLq', density_asp, self.basis_apqL)
        ae_asLg = density_L_expansion(
            engine, D_asLq, self.ae_radial_aqg, self.nc_ag)
        pseudo_asLg = density_L_expansion(
            engine, D_asLq, self.pseudo_radial_aqg, self.nct_ag)
        # Densities are asng; sigma fields are axng with x=(aa, ab, bb).
        ae_density_asng, ae_sigma_axng = polarized_density_and_grad(
            engine, ae_asLg, self.angular_anL, self.angular_gradient_anLv,
            self.radial_spacing_ag, self.inverse_radius2_ag)
        pseudo_density_asng, pseudo_sigma_axng = polarized_density_and_grad(
            engine, pseudo_asLg, self.angular_anL,
            self.angular_gradient_anLv, self.radial_spacing_ag,
            self.inverse_radius2_ag)
        energy_a = _integrate_correction(
            engine,
            PBEResult(ae_density_asng, ae_sigma_axng, evaluate_pbe),
            PBEResult(
                pseudo_density_asng, pseudo_sigma_axng, evaluate_pbe),
            self.angular_weight_an, self.radial_weight_ag,
            self.reference_energy_a)
        energy_a.backward()
        return energy_a.value, self.gradient


def _evaluate_lda_cpu(functional, density_sx, potential_sx, energy_x):
    """Adapt GPAW's CPU LDA evaluator to the result-node convention."""
    functional.xc.kernel.calculate(energy_x, density_sx, potential_sx)


def _evaluate_pbe_cpu(functional,
                      density_sx,
                      potential_sx,
                      energy_x,
                      sigma_xr,
                      sigma_derivative_xr):
    """Adapt GPAW's CPU PBE evaluator to the result-node convention."""
    functional.xc.kernel.calculate(
        energy_x, density_sx, potential_sx,
        sigma_xr, sigma_derivative_xr)


def evaluate_paw_xc(functional, batch: PAWXCBatch) -> tuple[object, object]:
    """Evaluate one prepared LDA or PBE PAW correction batch."""
    assert functional.name in {'LDA', 'PBE'}
    xp = functional.xp
    polarized = batch.density_asp.shape[1] == 2
    density = (batch.density_asp if polarized else
               batch.density_asp[:, 0])
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
        gradient=xp.zeros_like(density))

    if functional.name == 'LDA':
        cls: type[PAWXCCorrection] = (
            SpinPolarizedLDAXCCorrection if polarized else
            SpinPairedLDAXCCorrection)
        evaluate = (partial(_evaluate_lda_cpu, functional)
                    if xp is np else evaluate_lda_gpu)
    else:
        cls = (SpinPolarizedPBEXCCorrection if polarized else
               SpinPairedPBEXCCorrection)
        arguments.update(
            angular_gradient_anLv=batch.angular_gradient_anLv,
            radial_spacing_ag=batch.radial_spacing_ag,
            inverse_radius2_ag=batch.inverse_radius2_ag)
        evaluate = (partial(_evaluate_pbe_cpu, functional)
                    if xp is np else evaluate_pbe_gpu)

    correction = cls(**arguments)
    energies, gradients = correction.evaluate(Engine(xp), evaluate)
    if gradients.ndim == 2:
        gradients = gradients[:, None, :]
    return energies, gradients
