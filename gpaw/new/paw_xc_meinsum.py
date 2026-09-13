"""Small adapter between GPAW's collective PAW boundary and Meinsum."""

from collections import OrderedDict

import numpy as np

import meinsum
from gpaw.new.paw_xc_energy import spin_paired_pbe_paw_energy
from gpaw.new.timer import tracectx
from gpaw.sphere.lebedev import weight_n


def _static_arguments(xcc, xp):
    def array(value):
        return xp.ascontiguousarray(xp.asarray(value, dtype=xp.float32))

    r_g = array(xcc.rgd.r_g)
    inverse_r2_g = xp.zeros_like(r_g)
    inverse_r2_g[1:] = 1.0 / (r_g[1:] * r_g[1:])
    return (array(xcc.B_pqL), array(xcc.n_qg), array(xcc.nt_qg),
            array(xcc.nc_g), array(xcc.nct_g), array(xcc.Y_nL),
            array(xcc.rnablaY_nLv), array(xcc.rgd.dr_g),
            inverse_r2_g, array(weight_n), array(xcc.rgd.dv_g),
            xp.asarray(xcc.e_xc0, dtype=xp.float32))


def calculate_paw_corrections(setups, D_asp, xp):
    """Return PBE energies and AD Hamiltonians for all local atoms."""
    groups = OrderedDict()
    for a, D_sp in D_asp.items():
        xcc = setups[a].xc_correction
        if xcc.nc_corehole_g is not None:
            raise NotImplementedError(
                'Meinsum PAW XC does not yet support core-hole setups')
        group = groups.setdefault(id(xcc), (xcc, []))
        group[1].append((a, D_sp[0]))

    program = meinsum.differentiate_tensor_function(
        spin_paired_pbe_paw_energy)
    e_xc_a = {}
    dH_asp = {}
    for xcc, records in groups.values():
        density_ap = xp.asarray(
            np.stack([D_p for _, D_p in records]), dtype=xp.float32)
        static = _static_arguments(xcc, xp)
        blocks = [(density_ap[index], *static)
                  for index in range(len(records))]
        executor = meinsum.TensorGPUExecutor(program, blocks)
        with tracectx('Meinsum PAW XC', gpu=True):
            energy_a, (dH_ap,) = executor.execute()
        energy_a = xp.asnumpy(energy_a)
        dH_ap = xp.asnumpy(dH_ap)
        for index, (a, _) in enumerate(records):
            e_xc_a[a] = float(energy_a[index])
            dH_asp[a] = dH_ap[index][None]
    return e_xc_a, dH_asp
