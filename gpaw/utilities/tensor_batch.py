"""Gather and scatter homogeneous PAW XC tensor batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpaw.gpu import as_np
from gpaw.sphere.lebedev import weight_n


@dataclass(frozen=True)
class PAWXCBatch:
    """One shape-compatible collection of PAW XC tensor inputs."""

    atom_indices: tuple[int, ...]
    density_asp: object
    basis_apqL: object
    ae_radial_aqg: object
    pseudo_radial_aqg: object
    nc_ag: object
    nct_ag: object
    angular_anL: object
    angular_weight_an: object
    radial_weight_ag: object
    reference_energy_a: object
    angular_gradient_anLv: object | None = None
    radial_spacing_ag: object | None = None
    inverse_radius2_ag: object | None = None


def _shape_key(density_sp, correction, gga: bool) -> tuple:
    """Return the array-shape key for one homogeneous atom batch."""
    arrays = [correction.B_pqL,
              correction.n_qg,
              correction.nt_qg,
              correction.nc_g,
              correction.nct_g,
              correction.Y_nL,
              correction.rgd.dv_g]
    if gga:
        arrays.extend([correction.rnablaY_nLv, correction.rgd.dr_g])
    return (density_sp.shape, *(array.shape for array in arrays))


def _stack(xp, values, dtype):
    """Copy equal-shaped per-atom arrays into one contiguous leading batch."""
    return xp.ascontiguousarray(
        xp.stack([xp.asarray(value, dtype=dtype) for value in values]))


def _prepare_batch(xp, records, gga: bool) -> PAWXCBatch:
    """Transfer one shape group into batch-major tensor storage."""
    dtype = np.dtype(records[0][1].dtype)
    atom_indices = tuple(atom for atom, _, _ in records)
    corrections = [setup.xc_correction for _, _, setup in records]

    def arrays(name):
        return _stack(xp, [getattr(correction, name)
                           for correction in corrections], dtype)

    radial_grids = [correction.rgd for correction in corrections]
    arguments = dict(
        atom_indices=atom_indices,
        density_asp=_stack(xp, [D_sp for _, D_sp, _ in records], dtype),
        basis_apqL=arrays('B_pqL'),
        ae_radial_aqg=arrays('n_qg'),
        pseudo_radial_aqg=arrays('nt_qg'),
        nc_ag=arrays('nc_g'),
        nct_ag=arrays('nct_g'),
        angular_anL=arrays('Y_nL'),
        angular_weight_an=_stack(xp, [weight_n] * len(records), dtype),
        radial_weight_ag=_stack(
            xp, [grid.dv_g for grid in radial_grids], dtype),
        reference_energy_a=xp.asarray(
            [correction.e_xc0 for correction in corrections], dtype=dtype))
    if gga:
        radial_ag = _stack(xp, [grid.r_g for grid in radial_grids], dtype)
        inverse_radius2_ag = xp.zeros_like(radial_ag)
        inverse_radius2_ag[:, 1:] = 1.0 / radial_ag[:, 1:]**2
        arguments.update(
            angular_gradient_anLv=arrays('rnablaY_nLv'),
            radial_spacing_ag=_stack(
                xp, [grid.dr_g for grid in radial_grids], dtype),
            inverse_radius2_ag=inverse_radius2_ag)
    return PAWXCBatch(**arguments)


def gather_xc_paw(xp, setups, D_asp, *, gga: bool) \
        -> tuple[PAWXCBatch, ...]:
    """Group local atoms by tensor shape and gather their input arrays."""
    groups = {}
    for atom, D_sp in D_asp.items():
        setup = setups[atom]
        key = _shape_key(D_sp, setup.xc_correction, gga)
        groups.setdefault(key, []).append((atom, D_sp, setup))
    return tuple(_prepare_batch(xp, records, gga)
                 for records in groups.values())


def scatter_xc_paw(batches, results):
    """Scatter batch energies and gradients into atom-indexed mappings."""
    energy_a = {}
    gradient_asp = {}
    for batch, (energies, gradients) in zip(batches, results, strict=True):
        energies = as_np(energies)
        gradients = as_np(gradients)
        for position, atom in enumerate(batch.atom_indices):
            energy_a[atom] = float(energies[position])
            gradient_asp[atom] = gradients[position]
    return energy_a, gradient_asp
