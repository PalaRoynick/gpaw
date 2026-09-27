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

from functools import partial
import numpy as np

from gpaw.new.c import evaluate_lda_gpu, evaluate_pbe_gpu
from gpaw.utilities.tensor_batch import PAWXCBatch
from gpaw.utilities.tensor_autograd import EinsumResult, Engine, TensorResult


class LDAResult(TensorResult):
    """LDA energy and the potential needed by its reverse rule."""

    def __init__(self, density: TensorResult, function):
        xp = density.engine.xp
        density_sx = density.value.reshape(density.value.shape[0], -1)
        potential_sx = xp.zeros_like(density_sx)
        energy_x = xp.empty(density_sx.shape[1:], density.value.dtype)
        function(density_sx, potential_sx, energy_x)
        super().__init__(
            density.engine, energy_x.reshape(density.value.shape[1:]),
            density.indices.replace('s', '', 1))
        self.density = density
        self.potential = potential_sx.reshape(density.value.shape)
        self.parents = (density,)

    def _backward(self, gradient):
        return ((self.density, self.potential * gradient),)


class PBEResult(TensorResult):
    """PBE energy and its density and sigma partial derivatives."""

    def __init__(self,
                 density: TensorResult,
                 sigma: TensorResult,
                 function):
        xp = density.engine.xp
        density_sx = density.value.reshape(len(density.value), -1)
        sigma_xr = sigma.value.reshape(len(sigma.value), -1)
        potential_sx = xp.zeros_like(density_sx)
        sigma_derivative_xr = xp.empty_like(sigma_xr)
        energy_r = xp.empty(density_sx.shape[1:], density.value.dtype)
        function(density_sx, potential_sx, energy_r,
                 sigma_xr, sigma_derivative_xr)
        energy_r = energy_r.reshape(density.value.shape[1:])
        super().__init__(
            density.engine, energy_r,
            density.indices.replace('s', '', 1))
        self.density = density
        self.sigma = sigma
        self.potential = potential_sx.reshape(density.value.shape)
        self.sigma_derivative = sigma_derivative_xr.reshape(sigma.value.shape)
        self.parents = (density, sigma)

    def _backward(self, gradient):
        return ((self.density, self.potential * gradient[None]),
                (self.sigma, self.sigma_derivative * gradient[None]))


def density_L_expansion(engine: Engine,
                        D: TensorResult,
                        radial_qg,
                        core_g,
                        spinpol,
                        xp) -> EinsumResult:
    """Expand paired or spin-resolved coefficients into density multipoles."""
    const = xp.sqrt(xp.pi) if spinpol else xp.sqrt(4.0 * xp.pi)

    density_saLg = engine.einsum('saLq,qg->saLg', D, radial_qg)
    density_saLg[:, :, 0] += const * core_g[None, None, :]
    return density_saLg


def get_density_and_grad(engine: Engine,
                         density_saLg: TensorResult,
                         angular_nL,
                         angular_gradient_nLv,
                         radial_spacing_g,
                         inverse_radius2_g,
                         spinpol: bool
                         ) -> tuple[TensorResult, TensorResult]:
    """Construct two densities and aa/ab/bb sigma on the full radial grid."""
    # density_asng contains n_s(r) for both spins.
    density_sang = engine.einsum('nL,saLg->sang', angular_nL, density_saLg)
    derivative_saLg = engine.radial_derivative(density_saLg, radial_spacing_g)
    # radial_asng contains the radial gradient component for both spins.
    radial_sang = engine.einsum('nL,saLg->sang', angular_nL, derivative_saLg)
    tangent_sanvg = engine.einsum('nLv,saLg->sanvg', angular_gradient_nLv,
                                  density_saLg)

    inverse_radius2_11g = inverse_radius2_g[None, None, :]
    if not spinpol:
        # tangent2_ang is the squared angular component before division by r**2
        tangent2_sang = engine.sum(tangent_sanvg * tangent_sanvg, axis=-2)
        sigma_sang = (radial_sang * radial_sang +
                      tangent2_sang * inverse_radius2_11g)
        return density_sang, sigma_sang

    radial_a_ang = radial_sang.slice(
        (0, slice(None), slice(None), slice(None)), 'ang')
    radial_b_ang = radial_sang.slice(
        (1, slice(None), slice(None), slice(None)), 'ang')
    tangent_a_anvg = tangent_sanvg.slice(
        (0, slice(None), slice(None), slice(None), slice(None)), 'anvg')
    tangent_b_anvg = tangent_sanvg.slice(
        (1, slice(None), slice(None), slice(None), slice(None)), 'anvg')

    def sigma(radial_left, radial_right, tangent_left, tangent_right):
        """Form one grad(n_left) dot grad(n_right) field."""
        tangent2_ang = engine.sum(
            tangent_left * tangent_right, axis=-2)
        return (radial_left * radial_right +
                tangent2_ang * inverse_radius2_11g)

    sigma_aa_ang = sigma(
        radial_a_ang, radial_a_ang, tangent_a_anvg, tangent_a_anvg)
    sigma_ab_ang = sigma(
        radial_a_ang, radial_b_ang, tangent_a_anvg, tangent_b_anvg)
    sigma_bb_ang = sigma(
        radial_b_ang, radial_b_ang, tangent_b_anvg, tangent_b_anvg)
    sigma_xang = engine.stack(
        (sigma_aa_ang, sigma_ab_ang, sigma_bb_ang), 0, 'xang')
    return density_sang, sigma_xang


def _integrate_correction(engine: Engine,
                          ae_energy: TensorResult,
                          pseudo_energy: TensorResult,
                          angular_weight_n,
                          radial_weight_g,
                          reference_energy) -> TensorResult:
    """Integrate all-electron minus pseudo energy and subtract ``e_xc0``."""
    quadrature_ng = (angular_weight_n[:, None] *
                     radial_weight_g[None, :])
    energy_a = engine.sum(
        (ae_energy - pseudo_energy) * quadrature_ng, axis=(-2, -1))
    energy_a -= reference_energy
    return energy_a


class LDAXCCorrection(PAWXCBatch):
    """Prepared atom batch for the spin-polarized LDA PAW correction."""

    def evaluate(self, engine: Engine, evaluate_lda):
        """Return per-atom polarized LDA energies and fill dE/dD_sp."""
        # density_sap is the packed density matrix D_p for each atom/spin.
        density_sap = engine.requires_gradient(
            self.density_sap, self.gradient_sap, 'sap')
        # D_saLq holds spin-resolved density-expansion coefficients.
        D_saLq = engine.einsum(
            'sap,pqL->saLq', density_sap, self.basis_pqL)
        ae_saLg = density_L_expansion(engine, D_saLq,
                                      self.ae_radial_qg, self.nc_g,
                                      self.spinpol, xp=self.xp)
        pseudo_saLg = density_L_expansion(engine, D_saLq,
                                          self.pseudo_radial_qg, self.nct_g,
                                          self.spinpol, xp=self.xp)
        # ae_sang and pseudo_sang are the two spin densities on the grid.
        ae_sang = engine.einsum(
            'nL,saLg->sang', self.angular_nL, ae_saLg)
        pseudo_sang = engine.einsum(
            'nL,saLg->sang', self.angular_nL, pseudo_saLg)
        energy_a = _integrate_correction(
            engine,
            LDAResult(ae_sang, evaluate_lda),
            LDAResult(pseudo_sang, evaluate_lda),
            self.angular_weight_n, self.radial_weight_g,
            self.reference_energy)
        energy_a.backward()
        return energy_a.value, self.gradient_sap


class GGAXCCorrection(PAWXCBatch):
    """Additional radial and angular arrays needed by PBE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        xp = self.xp
        setup = self.setup
        rgd = self.setup.xc_correction.rgd
        self.radial_g = xp.asarray(rgd.r_g)
        self.radial_spacing_g = xp.asarray(rgd.dr_g)
        self.inverse_radius2_g = xp.zeros_like(rgd.r_g)
        self.inverse_radius2_g[self.radial_g != 0] = \
            1. / self.radial_g[self.radial_g != 0]**2
        self.angular_gradient_nLv = xp.asarray(setup.xc_correction.rnablaY_nLv)


class PBEXCCorrection(GGAXCCorrection):
    """Prepared atom batch for the spin-polarized PBE PAW correction."""

    def evaluate(self, engine: Engine, evaluate_pbe):
        """Return polarized PBE energies and fill dE/dD_asp."""
        # density_asp is the packed density matrix D_sp for each atom/spin.
        density_sap = engine.requires_gradient(
            self.density_sap, self.gradient_sap, 'sap')
        # D_asLq holds spin-resolved density-expansion coefficients.
        D_saLq = engine.einsum(
            'sap,pqL->saLq', density_sap, self.basis_pqL)
        ae_saLg = density_L_expansion(
            engine, D_saLq, self.ae_radial_qg, self.nc_g, self.spinpol,
            self.xp)
        pseudo_saLg = density_L_expansion(
            engine, D_saLq, self.pseudo_radial_qg, self.nct_g, self.spinpol,
            self.xp)
        # Densities are asng; sigma fields are axng with x=(aa, ab, bb).
        ae_density_sang, ae_sigma_xang = get_density_and_grad(
            engine, ae_saLg, self.angular_nL, self.angular_gradient_nLv,
            self.radial_spacing_g, self.inverse_radius2_g, self.spinpol)
        pseudo_density_sang, pseudo_sigma_xang = get_density_and_grad(
            engine, pseudo_saLg, self.angular_nL, self.angular_gradient_nLv,
            self.radial_spacing_g, self.inverse_radius2_g, self.spinpol)
        energy_a = _integrate_correction(
            engine,
            PBEResult(ae_density_sang, ae_sigma_xang, evaluate_pbe),
            PBEResult(pseudo_density_sang, pseudo_sigma_xang, evaluate_pbe),
            self.angular_weight_n, self.radial_weight_g,
            self.reference_energy)
        energy_a.backward()
        return energy_a.value, self.gradient_sap


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
    xp = batch.xp

    if functional.name == 'LDA':
        evaluate = (partial(_evaluate_lda_cpu, functional)
                    if xp is np else evaluate_lda_gpu)
    else:
        evaluate = (partial(_evaluate_pbe_cpu, functional)
                    if xp is np else evaluate_pbe_gpu)

    energies, gradients = batch.evaluate(Engine(xp), evaluate)
    return energies, gradients
