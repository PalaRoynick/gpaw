"""Spin-paired and spin-polarized LDA PAW correction programs.

The correction classes express the physical calculation as readable
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

from gpaw.new.c import evaluate_lda_gpu
from gpaw.new.xc.paw_xc_batch import PAWXCBatch
from gpaw.utilities.tensor_autograd import EinsumResult, Engine, TensorResult


class LDAResult(TensorResult):
    """LDA energy and the potential needed by its reverse rule."""

    def __init__(self, density: TensorResult, function):
        xp = density.engine.xp
        density_sx = density.value.reshape((density.value.shape[0], -1))
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


class LDAXCCorrection(PAWXCBatch):
    """Prepared atom batch for the spin-polarized LDA PAW correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        xp = self.xp
        self.evaluate_xc = (partial(_evaluate_lda_cpu, self.functional)
                            if xp is np else evaluate_lda_gpu)

    def evaluate(self):
        if self.setup.xc_correction is None:
            return super().evaluate()

        engine = Engine(self.xp)

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
        energy_a = integrate_correction(
            engine,
            LDAResult(ae_sang, self.evaluate_xc),
            LDAResult(pseudo_sang, self.evaluate_xc),
            self.angular_weight_n, self.radial_weight_g,
            self.reference_energy)
        energy_a.backward()
        return energy_a.value, self.gradient_sap


def _evaluate_lda_cpu(functional, density_sx, potential_sx, energy_x):
    """Adapt GPAW's CPU LDA evaluator to the result-node convention."""
    functional.xc.kernel.calculate(energy_x, density_sx, potential_sx)


def integrate_correction(engine: Engine,
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
