"""Benchmark heterogeneous PAW-XC batches for a 5x5 MoS2 composition."""

from __future__ import annotations

import argparse
from statistics import median

import cupy as cp
import numpy as np

import meinsum
from examples_and_benchmarks.paw_xc_batch import (
    PAWXCInputs,
    collect_paw_xc_batches,
)
from examples_and_benchmarks.paw_xc_energy import spin_paired_pbe_paw_energy
from gpaw.setup import create_setup
from gpaw.sphere.lebedev import weight_n
from gpaw.xc import XC


def make_batches():
    """Create 25 Mo and 50 S problems and group by complete tensor layout."""
    setup_by_symbol = {
        symbol: create_setup(symbol, XC('PBE'))
        for symbol in ['Mo', 'S']}
    inputs_by_symbol = {
        symbol: PAWXCInputs.from_gpaw(
            setup.xc_correction, weight_n, cp, dtype=np.float32)
        for symbol, setup in setup_by_symbol.items()}
    symbols = ['Mo'] * 25 + ['S'] * 50
    scales = np.linspace(0.8, 1.2, len(symbols), dtype=np.float32)
    problems = []
    for atom, (symbol, scale) in enumerate(zip(symbols, scales, strict=True)):
        nii = setup_by_symbol[symbol].xc_correction.nii
        density = cp.zeros(nii, dtype=cp.float32)
        density[0] = scale
        problems.append((atom, density, inputs_by_symbol[symbol]))
    batches = collect_paw_xc_batches(
        spin_paired_pbe_paw_energy, problems)
    assert sorted(len(batch.atom_indices) for batch in batches) == [25, 50]
    return batches


def make_group(batches, matrix_backend, concurrent):
    executors = [batch.gpu_executor(matrix_backend=matrix_backend)
                 for batch in batches]
    return meinsum.TensorGPUExecutorGroup(
        executors, concurrent=concurrent)


def elapsed_ms(group):
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    result = group.execute()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end), result


def benchmark(warmups, trials):
    batches = make_batches()
    groups = {
        'block sequential': make_group(batches, 'block', False),
        'BLAS sequential': make_group(batches, 'batched', False),
        'BLAS streams': make_group(batches, 'batched', True),
    }
    for _ in range(warmups):
        for group in groups.values():
            group.execute()
    cp.cuda.get_current_stream().synchronize()

    samples = {name: [] for name in groups}
    results = {}
    for trial in range(trials):
        items = list(groups.items())
        if trial % 2:
            items.reverse()
        for name, group in items:
            elapsed, results[name] = elapsed_ms(group)
            samples[name].append(elapsed)

    reference = results['BLAS sequential']
    for result in results.values():
        for (energy, (gradient,)), (ref_energy, (ref_gradient,)) in zip(
                result, reference, strict=True):
            cp.testing.assert_allclose(energy, ref_energy, rtol=2e-5, atol=2e-5)
            cp.testing.assert_allclose(
                gradient, ref_gradient, rtol=3e-5, atol=3e-5)
    return batches, groups, {name: median(values)
                             for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--trials', type=int, default=9)
    args = parser.parse_args()
    batches, groups, timings = benchmark(args.warmups, args.trials)
    device = cp.cuda.runtime.getDeviceProperties(0)['name']
    if isinstance(device, bytes):
        device = device.decode()
    print(f'GPU: {device}; composition: Mo25S50; dtype: float32')
    print('shape groups: ' + ', '.join(
        f'{len(batch.atom_indices)} atoms' for batch in batches))
    print('regroup tasks: ' + ', '.join(
        str(executor.regroup_task_count)
        for executor in groups['BLAS streams'].executors))
    for name, elapsed in timings.items():
        print(f'{name:18s} {elapsed:8.3f} ms')
    sequential = timings['BLAS sequential']
    streamed = timings['BLAS streams']
    print(f'stream overlap speedup {sequential / streamed:.3f}x')
    print(f'total vs block speedup '
          f'{timings["block sequential"] / streamed:.3f}x')


if __name__ == '__main__':
    main()
