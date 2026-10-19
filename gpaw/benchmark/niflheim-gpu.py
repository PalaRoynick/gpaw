from os import getcwd
from pathlib import Path

from gpaw.benchmark import benchmark_main

si_defect_weak_scaling = [
    ('Si127_defect_gpu', 1),
    ('Si255_defect_gpu', 2),
    ('Si511_defect_gpu', 4),
    ('Si1023_defect_gpu', 8),
    ('Si1535_defect_gpu', 12),
    ('Si2047_defect_gpu', 16),
]

si_defect_strong_scaling = [
    ('Si2047_defect_gpu', 1),
    ('Si2047_defect_gpu', 2),
    ('Si2047_defect_gpu', 4),
    ('Si2047_defect_gpu', 8),
    ('Si2047_defect_gpu', 12),
    ('Si2047_defect_gpu', 16),
]


def submit_a100_gpu_scaling(run, path, name, benchmark, total_gpus, time='4h'):
    ncores = 32 * total_gpus
    nprocs = total_gpus
    gpus_per_node = min(total_gpus, 4)
    folder = f'a100-{name}-{benchmark}-{total_gpus}gpu'
    task_name = f'{benchmark}-a100-{name}-{total_gpus}gpu'
    (path / folder).mkdir(exist_ok=True)
    kwargs = {
        'function': benchmark_main,
        'args': (benchmark,),
        'nodename': 'a100',
        'cores': ncores,
        'processes': nprocs,
        'gpus': gpus_per_node,
        'tmax': time,
        'name': task_name,
        'folder': folder,
    }
    info = {'total_gpus': total_gpus,
            **{key: value for key, value in kwargs.items()
               if key != 'function'}}
    print('Submitting', name, benchmark, info)
    try:
        run(**kwargs)
    except Exception as err:
        print('Failed to submit', name, benchmark, info)
        print(repr(err))
        return False
    return True


def workflow():
    from myqueue.workflow import run
    path = Path(getcwd())
    failures = []
    for benchmark, total_gpus in si_defect_weak_scaling:
        ok = submit_a100_gpu_scaling(
            run, path, 'weak', benchmark, total_gpus)
        if not ok:
            failures.append(('weak', benchmark, total_gpus))
    for benchmark, total_gpus in si_defect_strong_scaling:
        ok = submit_a100_gpu_scaling(
            run, path, 'strong', benchmark, total_gpus)
        if not ok:
            failures.append(('strong', benchmark, total_gpus))
    if failures:
        raise RuntimeError(f'Failed to submit {failures}')
