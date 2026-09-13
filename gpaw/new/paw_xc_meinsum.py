"""Small adapter between GPAW's collective PAW boundary and Meinsum."""

from collections import OrderedDict

import numpy as np

import meinsum
from examples_and_benchmarks.paw_xc_batch import (
    PAWXCInputs,
    collect_paw_xc_batches,
)
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
    problems_by_function = OrderedDict()
    inputs_by_correction = {}
    for a, D_sp in D_asp.items():
        xcc = setups[a].xc_correction
        if xcc.nc_corehole_g is not None:
            raise NotImplementedError(
                'Meinsum PAW XC does not yet support core-hole setups')
        function = FUNCTIONS[xc_name, len(D_sp)]
        inputs = inputs_by_correction.get(id(xcc))
        if inputs is None:
            inputs = PAWXCInputs.from_gpaw(
                xcc, weight_n, xp, dtype=np.float32)
            inputs_by_correction[id(xcc)] = inputs
        density = D_sp if len(D_sp) == 2 else D_sp[0]
        density = xp.asarray(density, dtype=xp.float32)
        problems_by_function.setdefault(function, []).append(
            (a, density, inputs))

    batches = tuple(
        batch
        for function, problems in problems_by_function.items()
        for batch in collect_paw_xc_batches(function, problems))
    if not batches:
        return {}, {}
    executors = tuple(
        batch.gpu_executor(matrix_backend='batched') for batch in batches)

    e_xc_a = {}
    dH_asp = {}
    executor_group = meinsum.TensorGPUExecutorGroup(executors)
    with tracectx('Meinsum PAW XC', gpu=True):
        results = executor_group.execute()
    for batch, (energy_a, (dH_ap,)) in zip(
            batches, results, strict=True):
        energy_a = xp.asnumpy(energy_a)
        dH_ap = xp.asnumpy(dH_ap)
        for index, a in enumerate(batch.atom_indices):
            e_xc_a[a] = float(energy_a[index])
            gradient = dH_ap[index]
            dH_asp[a] = gradient if gradient.ndim == 2 else gradient[None]
    return e_xc_a, dH_asp
