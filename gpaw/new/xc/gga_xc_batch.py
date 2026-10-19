"""Spin-paired and spin-polarized PBE PAW correction programs, as well as tools
for GGAs.

See also the LDA XC batch.
"""

from __future__ import annotations

from functools import partial
import numpy as np

from gpaw.new.c import evaluate_pbe_gpu
from gpaw.new.xc.paw_xc_batch import PAWXCBatch
from gpaw.new.xc.lda_xc_batch import density_L_expansion, integrate_correction
from gpaw.utilities.tensor_autograd import Engine, TensorResult
from gpaw.gpu import cupy_is_fake


class PBEResult(TensorResult):
    """PBE energy and its density and sigma partial derivatives."""

    def __init__(self,
                 density: TensorResult,
                 sigma: TensorResult,
                 function):
        xp = density.engine.xp
        density_sx = density.value.reshape((density.value.shape[0], -1))
        sigma_xr = sigma.value.reshape((sigma.value.shape[0], -1))
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


class GGAXCCorrection(PAWXCBatch):
    """Additional radial and angular arrays needed by PBE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.setup.xc_correction is None:
            return
        xp = self.xp
        setup = self.setup
        rgd = self.setup.xc_correction.rgd
        self.radial_g = xp.asarray(rgd.r_g)
        self.radial_spacing_g = xp.asarray(rgd.dr_g)
        self.inverse_radius2_g = xp.zeros_like(self.radial_g)
        self.inverse_radius2_g[self.radial_g != 0] = \
            1. / self.radial_g[self.radial_g != 0]**2

        self.angular_gradient_nLv = xp.asarray(setup.xc_correction.rnablaY_nLv)

    def evaluate(self):
        if self.setup.xc_correction is None:
            return super().evaluate()

        engine = Engine(self.xp)

        """Return polarized PBE energies and fill dE/dD_asp."""
        # density_asp is the packed density matrix D_sp for each atom/spin.
        density_sap = engine.requires_gradient(
            self.density_sap, self.gradient_sap, 'sap')
        # D_asLq holds spin-resolved density-expansion coefficients.
        D_saLq = engine.einsum(
            'sap,pqL->saLq', density_sap, self.basis_pqL)
        ae_saLg = density_L_expansion(
            engine, D_saLq, self.ae_radial_qg, self.nc0_sg,
            self.xp)
        pseudo_saLg = density_L_expansion(
            engine, D_saLq, self.pseudo_radial_qg, self.nct0_sg,
            self.xp)
        # Densities are asng; sigma fields are axng with x=(aa, ab, bb).
        ae_density_sang, ae_sigma_xang = get_density_and_grad(
            engine, ae_saLg, self.angular_nL, self.angular_gradient_nLv,
            self.radial_spacing_g, self.inverse_radius2_g, self.spinpol)
        pseudo_density_sang, pseudo_sigma_xang = get_density_and_grad(
            engine, pseudo_saLg, self.angular_nL, self.angular_gradient_nLv,
            self.radial_spacing_g, self.inverse_radius2_g, self.spinpol)
        energy_a = integrate_correction(
            engine,
            PBEResult(ae_density_sang, ae_sigma_xang, self.evaluate_xc),
            PBEResult(pseudo_density_sang, pseudo_sigma_xang, self.evaluate_xc),
            self.angular_weight_n, self.radial_weight_g,
            self.reference_energy)
        energy_a.backward()
        return energy_a.value, self.gradient_sap


class PBEXCCorrection(GGAXCCorrection):
    """Prepared atom batch for the spin-polarized PBE PAW correction."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_xc = (partial(_evaluate_pbe_cpu, self.functional)
                            if self.xp is np or cupy_is_fake else
                            evaluate_pbe_gpu)


def _evaluate_pbe_cpu(functional,
                      density_sx,
                      potential_sx,
                      energy_x,
                      sigma_xr,
                      sigma_derivative_xr):
    """Adapt GPAW's CPU PBE evaluator to the result-node convention."""
    # In case of cpupy we want to get the _data object (the numpy array)
    density_sx = getattr(density_sx, '_data', density_sx)
    potential_sx = getattr(potential_sx, '_data', potential_sx)
    energy_x = getattr(energy_x, '_data', energy_x)
    sigma_xr = getattr(sigma_xr, '_data', sigma_xr)
    sigma_derivative_xr = getattr(sigma_derivative_xr, '_data',
                                  sigma_derivative_xr)

    functional.xc.kernel.calculate(
        energy_x, density_sx, potential_sx,
        sigma_xr, sigma_derivative_xr)


def get_density_and_grad(engine: Engine,
                         density_saLg: TensorResult,
                         angular_nL,
                         angular_gradient_nLv,
                         radial_spacing_g,
                         inverse_radius2_g,
                         spinpol: bool,
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
