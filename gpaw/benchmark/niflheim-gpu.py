from os import getcwd
from pathlib import Path

from gpaw.benchmark import benchmark_main

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
    for benchmark, ngpus in si_defect_weak_scaling:
        submit_a100_gpu_scaling(run, path, 'weak', benchmark, ngpus)
    for benchmark, ngpus in si_defect_strong_scaling:
        submit_a100_gpu_scaling(run, path, 'strong', benchmark, ngpus)
