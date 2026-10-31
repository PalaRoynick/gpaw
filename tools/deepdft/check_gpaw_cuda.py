"""Compare CPU and real CUDA PW/PBE calculations; never accept GPU emulation."""

import argparse
import faulthandler
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc100
faulthandler.enable(all_threads=True)

from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world


def calculate(output, gpu, mode='pw', family='bulk', skip_forces=False, sync_gpu=False):
    backend = 'cuda' if gpu else 'cpu'

    def stage(name, operation):
        print(json.dumps({'backend': backend, 'stage': name, 'status': 'started'}), flush=True)
        result = operation()
        if gpu and sync_gpu:
            from gpaw.gpu import cupy
            cupy.cuda.runtime.deviceSynchronize()
        print(json.dumps({'backend': backend, 'stage': name, 'status': 'completed'}), flush=True)
        return result

    if family == 'bulk':
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    elif family == 'slab':
        atoms = fcc100('Al', size=(1, 1, 2), a=4.05, vacuum=3)
    else:
        atoms = Atoms('Al2', positions=[[0, 0, 0], [0, 0, 2.6]])
        atoms.center(vacuum=3)
    atoms.positions[0, 0] += 0.04
    poisson = {}
    if mode == 'fd':
        if family == 'finite':
            poisson = {'name': 'MomentCorrectionPoissonSolver',
                       'poissonsolver': {'name': 'fast'}, 'moment_corrections': 4}
        elif family == 'slab':
            poisson = {'dipolelayer': 'xy'}
    options = {'mode': PW(300)} if mode == 'pw' else {'mode': 'fd', 'h': 0.3}
    if mode == 'fd':
        if gpu:
            from gpaw.density_learning.poisson import HostStagedPoisson
            options['poissonsolver'] = HostStagedPoisson(poisson)
        elif poisson:
            options['poissonsolver'] = poisson
    atoms.calc = stage('calculator_creation', lambda: GPAW(**options, xc='PBE',
                      kpts=tuple(2 if p else 1 for p in atoms.pbc),
                      occupations=FermiDirac(0.1), symmetry='off',
                      convergence={'energy': 1e-8, 'density': 1e-8,
                                   'eigenstates': 1e-10},
                      parallel={'gpu': gpu}, txt=str(output / f'{gpu=}.log')))
    start = monotonic()
    energy = stage('energy_scf', atoms.get_potential_energy)
    forces = None if skip_forces else stage('forces', atoms.get_forces)
    densities = stage('densities_api', atoms.calc.dft.densities)
    field = stage('pseudo_density', lambda: densities.pseudo_densities(
        grid_refinement=1, add_compensation_charges=False))
    field = stage('density_to_host', lambda: field.to_xp(np))
    field = stage('density_gather', field.gather)
    density = field.data
    elapsed = monotonic() - start
    stage('calculator_close', atoms.calc.log.close)
    return energy, forces, density, elapsed


def run_backend(output, name, args):
    """Use exec-based subprocesses so CPU/GPU calculator lifetimes cannot overlap."""
    directory = output / name
    log = output / (name + '-worker.log')
    command = [sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve()),
               '--backend', name, '--mode', args.mode, '--family', args.family,
               '--output', str(directory)]
    for flag, enabled in [('--skip-forces', args.skip_forces),
                          ('--sync-gpu', args.sync_gpu),
                          ('--require-v100', args.require_v100)]:
        if enabled:
            command.append(flag)
    print(json.dumps({'backend': name, 'stage': 'worker_launch', 'log': str(log)}), flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        failure = {'gpu_validated': False, 'failed_backend': name,
                   'returncode': code, 'signal': -code if code < 0 else None,
                   'worker_log': str(log)}
        (output / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise SystemExit(f'Backend {name} failed with returncode {code}; see {log}')
    report = json.loads((directory / 'validation.json').read_text())
    with np.load(directory / 'result.npz', allow_pickle=False) as data:
        result = (float(data['energy']), data['forces'].copy(),
                  data['density'].copy(), float(data['seconds']))
    return result, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--cpu-only', action='store_true')
    selection.add_argument('--backend', choices=['both', 'cpu', 'cuda'], default='both',
                           help='Single-backend runs are diagnostics, not acceptance')
    p.add_argument('--skip-forces', action='store_true',
                   help='Diagnostic only; never mark GPU acceptance as passed')
    p.add_argument('--sync-gpu', action='store_true',
                   help='Synchronize after every stage to localize asynchronous CUDA errors')
    p.add_argument('--mode', choices=['pw', 'fd'], default='pw')
    p.add_argument('--family', choices=['bulk', 'slab', 'finite'], default='bulk')
    p.add_argument('--require-v100', action='store_true')
    args = p.parse_args()
    if world.size != 1:
        raise SystemExit('This acceptance check requires one MPI rank and one GPU')
    if args.mode == 'pw' and args.family != 'bulk':
        raise SystemExit('Use FD to validate the open-boundary protocol')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    backend = 'cpu' if args.cpu_only else args.backend
    report = {'gpu_validated': False, 'mode': args.mode, 'family': args.family,
              'backend': backend, 'force_check_skipped': args.skip_forces,
              'gpu_stage_synchronization': args.sync_gpu}
    if backend == 'cuda':
        from gpaw.gpu import cupy as cp, cupy_is_fake
        from gpaw.new.c import GPU_ENABLED
        if cupy_is_fake or not GPU_ENABLED:
            raise SystemExit('A real CUDA-enabled GPAW build and allocated GPU are required')
        props = cp.cuda.runtime.getDeviceProperties(0)
        capability = (props['major'], props['minor'])
        if args.require_v100 and capability != (7, 0):
            raise SystemExit(f'Expected Volta capability 7.0, found {capability}')
        report.update(device=str(props['name']), capability=capability,
                      cupy_version=cp.__version__)
    results = {}
    if backend == 'both':
        for name in ['cpu', 'cuda']:
            results[name], child_report = run_backend(output, name, args)
            report[name + '_seconds'] = results[name][3]
            if name == 'cuda':
                for key in ('device', 'capability', 'cupy_version'):
                    report[key] = child_report[key]
    else:
        results[backend] = calculate(output, backend == 'cuda', args.mode, args.family,
                                     args.skip_forces, args.sync_gpu)
        energy, forces, density, seconds = results[backend]
        np.savez(output / 'result.npz', energy=energy,
                 forces=np.empty((0, 3)) if forces is None else forces,
                 density=density, seconds=seconds)
        report[backend + '_seconds'] = seconds
    if backend == 'both':
        cpu, gpu = results['cpu'], results['cuda']
        np.testing.assert_allclose(gpu[0], cpu[0], rtol=0, atol=1e-5)
        if not args.skip_forces:
            np.testing.assert_allclose(gpu[1], cpu[1], rtol=0, atol=2e-5)
            report['max_force_error_ev_per_angstrom'] = float(np.max(np.abs(gpu[1] - cpu[1])))
        relative_l1 = float(np.abs(gpu[2] - cpu[2]).sum() / np.abs(cpu[2]).sum())
        assert relative_l1 < 1e-6, relative_l1
        report.update(gpu_validated=not args.skip_forces,
                      energy_error_ev=float(gpu[0] - cpu[0]),
                      density_relative_l1=relative_l1)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
