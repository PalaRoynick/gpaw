"""Compare CPU and real CUDA PW/PBE calculations; never accept GPU emulation."""

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc100
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world


def calculate(output, gpu, mode='pw', family='bulk'):
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
    atoms.calc = GPAW(**options, xc='PBE',
                      kpts=tuple(2 if p else 1 for p in atoms.pbc),
                      occupations=FermiDirac(0.1), symmetry='off',
                      convergence={'energy': 1e-8, 'density': 1e-8,
                                   'eigenstates': 1e-10},
                      parallel={'gpu': gpu}, txt=str(output / f'{gpu=}.log'))
    start = monotonic()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    density = atoms.calc.dft.densities().pseudo_densities(
        grid_refinement=1, add_compensation_charges=False).to_xp(np).gather().data
    return energy, forces, density, monotonic() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cpu-only', action='store_true')
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
    report = {'gpu_validated': False, 'mode': args.mode, 'family': args.family}
    if not args.cpu_only:
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
    cpu = calculate(output, False, args.mode, args.family)
    report['cpu_seconds'] = cpu[3]
    if not args.cpu_only:
        gpu = calculate(output, True, args.mode, args.family)
        np.testing.assert_allclose(gpu[0], cpu[0], rtol=0, atol=1e-5)
        np.testing.assert_allclose(gpu[1], cpu[1], rtol=0, atol=2e-5)
        relative_l1 = float(np.abs(gpu[2] - cpu[2]).sum() / np.abs(cpu[2]).sum())
        assert relative_l1 < 1e-6, relative_l1
        report.update(gpu_validated=True, gpu_seconds=gpu[3],
                      energy_error_ev=float(gpu[0] - cpu[0]),
                      max_force_error_ev_per_angstrom=float(np.max(np.abs(gpu[1] - cpu[1]))),
                      density_relative_l1=relative_l1)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
