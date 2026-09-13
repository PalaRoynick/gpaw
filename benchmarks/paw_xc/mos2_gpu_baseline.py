"""Profile the float32 GPU baseline for a 5x5 MoS2 monolayer.

Run from an empty output directory.  GPAW's GPUProfiler writes Chrome/
Perfetto trace files alongside ``baseline-summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from functools import wraps
from pathlib import Path
from time import perf_counter

# This must be set before GPAW (and gpaw.new.timer) is imported.
os.environ.setdefault('GPAW_TRACE', '1')

import cupy as cp
import numpy as np
from ase.build import mx2
from ase.units import Ha

import gpaw
from gpaw import GPAW
from gpaw.new.timer import global_timer, tracectx
from gpaw.new.xc import Functional
from gpaw.utilities.timing import GPUProfiler


def install_paw_xc_timers(samples: dict[str, list[float]]) -> None:
    """Profile both the collective boundary and legacy atom calls."""
    original_atom = Functional.calculate_paw_correction

    @wraps(original_atom)
    def timed_atom(self, *args, **kwargs):
        start = perf_counter()
        try:
            with tracectx('PAW XC correction'):
                return original_atom(self, *args, **kwargs)
        finally:
            samples['atom'].append(perf_counter() - start)

    original_collective = Functional.calculate_paw_corrections

    @wraps(original_collective)
    def timed_collective(self, *args, **kwargs):
        start = perf_counter()
        try:
            with tracectx('PAW XC collective'):
                return original_collective(self, *args, **kwargs)
        finally:
            samples['collective'].append(perf_counter() - start)

    Functional.calculate_paw_correction = timed_atom
    Functional.calculate_paw_corrections = timed_collective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('.'))
    parser.add_argument('--ecut', type=float, default=300.0)
    parser.add_argument('--maxiter', type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    samples: dict[str, list[float]] = {'atom': [], 'collective': []}
    install_paw_xc_timers(samples)

    atoms = mx2(formula='MoS2', kind='2H', a=3.184,
                thickness=3.127, size=(5, 5, 1), vacuum=5.0)
    atoms.pbc = (True, True, False)
    txt = args.output / 'mos2-5x5-gpu-float32.txt'
    atoms.calc = GPAW(
        mode={'name': 'pw', 'ecut': args.ecut, 'dtype': np.float32},
        xc='PBE',
        kpts=(1, 1, 1),
        parallel={'gpu': True},
        random=True,
        maxiter=args.maxiter,
        convergence={'density': 3.0, 'eigenstates': 3.0, 'energy': 10.0},
        txt=str(txt))

    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    prefix = args.output / 'gpu-profile'
    with global_timer.context(GPUProfiler(str(prefix))):
        energy_ev = atoms.get_potential_energy()
    cp.cuda.runtime.deviceSynchronize()
    wall_time = perf_counter() - start

    dtype = np.dtype(atoms.calc.wfs.dtype)
    assert dtype == np.dtype(np.float32), dtype
    device = cp.cuda.runtime.getDeviceProperties(0)
    device_name = device['name']
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    summary = {
        'system': 'MoS2 5x5 monolayer',
        'atoms': len(atoms),
        'formula': atoms.get_chemical_formula(),
        'xc': 'PBE',
        'mode': 'pw',
        'ecut_eV': args.ecut,
        'dtype': dtype.name,
        'kpts': [1, 1, 1],
        'gpu': device_name,
        'python': platform.python_version(),
        'gpaw': gpaw.__version__,
        'numpy': np.__version__,
        'cupy': cp.__version__,
        'energy_eV': energy_ev,
        'energy_Ha': energy_ev / Ha,
        'wall_time_s': wall_time,
        'paw_xc_collective_calls': len(samples['collective']),
        'paw_xc_collective_total_s': sum(samples['collective']),
        'paw_xc_collective_mean_ms': (
            1000.0 * sum(samples['collective']) /
            len(samples['collective'])),
        'legacy_paw_xc_atom_calls': len(samples['atom']),
        'legacy_paw_xc_atom_total_s': sum(samples['atom']),
        'trace': str(prefix.with_suffix('.json')),
    }
    path = args.output / 'baseline-summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
