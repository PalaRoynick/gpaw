"""Small adapter between GPAW's collective PAW boundary and Meinsum."""

from collections import OrderedDict

import numpy as np

import meinsum
from examples_and_benchmarks.paw_xc_batch import PAWXCInputs
from examples_and_benchmarks.paw_xc_energy import (
    spin_paired_lda_paw_energy,
    spin_paired_pbe_paw_energy,
    spin_polarized_lda_paw_energy,
    spin_polarized_pbe_paw_energy,
)
from gpaw.new.timer import tracectx
from gpaw.sphere.lebedev import weight_n


FUNCTIONS = {
    ('LDA', 1): spin_paired_lda_paw_energy,
    ('LDA', 2): spin_polarized_lda_paw_energy,
    ('PBE', 1): spin_paired_pbe_paw_energy,
    ('PBE', 2): spin_polarized_pbe_paw_energy,
}


def calculate_paw_corrections(setups, D_asp, xp, xc_name):
    """Return LDA/PBE energies and AD Hamiltonians for all local atoms."""
    groups = OrderedDict()
    for a, D_sp in D_asp.items():
        xcc = setups[a].xc_correction
        if xcc.nc_corehole_g is not None:
            raise NotImplementedError(
                'Meinsum PAW XC does not yet support core-hole setups')
        function = FUNCTIONS[xc_name, len(D_sp)]
        group = groups.setdefault((id(xcc), function), (xcc, function, []))
        group[2].append((a, D_sp if len(D_sp) == 2 else D_sp[0]))

    e_xc_a = {}
    dH_asp = {}
    for xcc, function, records in groups.values():
        density_batch = xp.asarray(
            np.stack([D for _, D in records]), dtype=xp.float32)
        inputs = PAWXCInputs.from_gpaw(
            xcc, weight_n, xp, dtype=np.float32)
        blocks = [inputs.arguments(function, density_batch[index])
                  for index in range(len(records))]
        program = meinsum.differentiate_tensor_function(function)
        executor = meinsum.TensorGPUExecutor(program, blocks)
        with tracectx('Meinsum PAW XC', gpu=True):
            energy_a, (dH_ap,) = executor.execute()
        energy_a = xp.asnumpy(energy_a)
        dH_ap = xp.asnumpy(dH_ap)
        for index, (a, _) in enumerate(records):
            e_xc_a[a] = float(energy_a[index])
            gradient = dH_ap[index]
            dH_asp[a] = gradient if gradient.ndim == 2 else gradient[None]
    return e_xc_a, dH_asp
