from os import getcwd
from pathlib import Path

from gpaw.benchmark import benchmark_main, get_benchmarks

platforms = [('xeon24el8_test', 24, 24, 0, '10m'),
             ('a100', 8, 1, 1, '1h'),
             ('a100', 128, 4, 4, '1h'),
             ('sm3090el8', 8, 1, 1, '1h'),
             ('epyc96', 96, 96, 0, '2h'),
             ('epyc96', 96 * 2, 96 * 2, 0, '4h')]

si_defect_gpu_benchmarks = [
    'Si383_defect_gpu',
    'Si511_defect_gpu',
    'Si799_defect_gpu',
    'Si1151_defect_gpu',
    'Si1535_defect_gpu',
    'Si2047_defect_gpu',
]

si_defect_weak_scaling = [
    ('Si383_defect_gpu', 3),
    ('Si511_defect_gpu', 4),
    ('Si799_defect_gpu', 6),
    ('Si1151_defect_gpu', 9),
    ('Si1535_defect_gpu', 12),
    ('Si2047_defect_gpu', 16),
]

si_defect_strong_scaling = [
    ('Si2047_defect_gpu', 1),
    ('Si2047_defect_gpu', 2),
    ('Si2047_defect_gpu', 4),
    ('Si2047_defect_gpu', 8),
    ('Si2047_defect_gpu', 16),
]


def submit_a100_gpu_scaling(run, path, name, benchmark, ngpus, time='4h'):
    ncores = 32 * ngpus
    nprocs = ngpus
    folder = f'a100-{name}-{benchmark}-{ngpus}gpu'
    (path / folder).mkdir(exist_ok=True)
    run(function=benchmark_main, args=(benchmark,),
        nodename='a100', cores=ncores, processes=nprocs,
        gpus=ngpus, tmax=time,
        name=f'{benchmark}-a100-{name}-{ngpus}gpu',
        folder=folder)


def workflow():
    from myqueue.workflow import run
    path = Path(getcwd())
    for partition, ncores, nprocs, ngpus, time in platforms:
        for benchmark in get_benchmarks(
                cores=nprocs, memory='10000G', gpus=ngpus):
            if benchmark in si_defect_gpu_benchmarks:
                continue
            (path / f'{partition}-{nprocs}').mkdir(exist_ok=True)
            run(function=benchmark_main, args=(benchmark,),
                nodename=partition, cores=ncores, processes=nprocs,
                gpus=ngpus, tmax=time,
                name=f'{benchmark}-{partition}-{nprocs}',
                folder=f'{partition}-{nprocs}')
            if ngpus == 0:
                run(function=benchmark_main, args=(f'{benchmark}#old',),
                    nodename=partition, cores=ncores, processes=nprocs,
                    gpus=ngpus, tmax=time,
                    name=f'{benchmark}-{partition}-{nprocs}-old',
                    folder=f'{partition}-{nprocs}')
    for benchmark, ngpus in si_defect_weak_scaling:
        submit_a100_gpu_scaling(run, path, 'weak', benchmark, ngpus)
    for benchmark, ngpus in si_defect_strong_scaling:
        submit_a100_gpu_scaling(run, path, 'strong', benchmark, ngpus)
