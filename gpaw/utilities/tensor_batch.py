"""Gather and scatter homogeneous PAW XC tensor batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gpaw.gpu import as_np
from gpaw.sphere.lebedev import weight_n
from gpaw.utilities import pack_density, unpack_density

class PAWXCBatch:
    """One shape-compatible collection of PAW XC tensor inputs."""
    def __init__(self, c_a, D_asii, V_aL, setups, gga: bool):
        xp = D_asii.layout.xp
        self.xp = xp
        self.spinpol = D_asii.mydims == (2,)
        order_b = D_asii.layout.order_b
        atom_indices = [D_asii.layout.myindices[c] for c in c_a]
        i_start = min(atom_index[1] for atom_index in atom_indices)
        i_end = max(atom_index[2] for atom_index in atom_indices)
        self.atom_indices = c_a
        self.V_aL = np.asarray([V_aL[c] for c in c_a])

        # Assert contiguous
        i_starts = {atom_index[1] for atom_index in atom_indices}
        i_starts.add(i_end)
        i_ends = {atom_index[2] for atom_index in atom_indices}
        i_ends.add(i_start)
        assert i_starts == i_ends

        setup = setups[c_a[0]]
        self.setup = setup
        D_scii = D_asii.data[:, i_start:i_end].reshape(
            -1, len(c_a), setup.ni, setup.ni)
        self.density_sap = pack_density(D_scii.real, xp=self.xp)
        self.gradient_sap = xp.zeros_like(self.density_sap)
        self.basis_pqL = xp.asarray(setup.xc_correction.B_pqL)
        self.ae_radial_qg = xp.asarray(setup.xc_correction.n_qg)
        self.pseudo_radial_qg = xp.asarray(setup.xc_correction.nt_qg)
        self.nc_g = xp.asarray(setup.xc_correction.nc_g)
        self.nct_g = xp.asarray(setup.xc_correction.nct_g)
        self.angular_nL = xp.asarray(setup.xc_correction.Y_nL)
        self.angular_weight_n = xp.asarray(weight_n)

        rgd = setup.xc_correction.rgd
        self.radial_weight_g = xp.asarray(rgd.dv_g)
        self.reference_energy = xp.asarray(setup.xc_correction.e_xc0)

        if gga:
            self.radial_g = xp.asarray(rgd.r_g)
            self.radial_spacing_g = xp.asarray(rgd.dr_g)
            self.inverse_radius2_g = xp.zeros_like(rgd.r_g)
            self.inverse_radius2_g[rgd.radial_g != 0] = 1. / self.radial_g**2
            self.angular_gradient_nLv = xp.asarray(setup.xc_correction.rnablaY_nLv)

    def unpack(self):
        setup = self.setup
        D_ap = as_np(self.density_sap.sum(axis=0))
        dH_ap = self.xp.array(setup.K_p + setup.M_p +
                              setup.MB_p + 2.0 * D_ap @ setup.M_pp +
                              self.V_aL @ setup.Delta_pL.T)
        ekin_a = D_ap @ setup.K_p + setup.Kc
        e0_a = setup.MB + D_ap @ setup.MB_p
        ecou_a = setup.M + D_ap[:, None] @ (setup.M_p + D_ap @ setup.M_pp)[:, :, None]
        ecou_a = ecou_a.squeeze()
        return ekin_a, e0_a, ecou_a, dH_ap

def pack_xc_batches(D_asii, V_aL, setups, *, functional) \
        -> tuple[PAWXCBatch, ...]:
    """Group local atoms by tensor shape and gather their input arrays."""
    myid_a = {ind: setups.id_a[ind] for ind in D_asii.layout.atomdist.indices}

    atoms_batch = {}
    for a, id in myid_a.items():
        atoms_batch[id] = atoms_batch.get(id, []) + [a]

    assert functional.name in {'LDA', 'PBE'}
    gga = functional.name == 'PBE'
    polarized = True  # D_asii.mydims == (2, )

    from gpaw.new.xccorrections2 import (
        LDAXCCorrection,
        SpinPairedPBEXCCorrection,
        SpinPolarizedPBEXCCorrection)


    if functional.name == 'LDA':
        cls = LDAXCCorrection
    else:
        if polarized:
            cls = SpinPolarizedPBEXCCorrection
        else:
            cls = SpinPairedPBEXCCorrection

    return [cls(a_a, D_asii, V_aL, setups, gga=gga)
            for a_a in atoms_batch.values()]


def unpack_xc_batches(batches, results) -> dict[int, np.ndarray]:
    """Upnack xc batch energies and gradients into atom-indexed mappings."""
    corr_a = {}

    for batch, (energies, gradients) in zip(batches, results, strict=True):
        ekin_a, e0_a, ecou_a, dH_ap = batch.unpack()

        exc_a = as_np(energies)
        gradients += dH_ap
        gradients_saii = unpack_density(gradients, xp=batch.xp)
        gradients = as_np(gradients)
        gradients_saii = as_np(gradients_saii)
        for position, atom in enumerate(batch.atom_indices):
            corr_a[atom] = {
                'e_xc': float(exc_a[position]),
                'e_kin': float(ekin_a[position]),
                'e_0': float(e0_a[position]),
                'e_cou': float(ecou_a[position]),
                'e_soc': float(0),
                'gradient_sp': gradients[:, position],
                'gradient_sii': gradients_saii[:, position],
            }
    return corr_a
