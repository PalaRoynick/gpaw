from gpaw.benchmark.performance_index import REFERENCES, PARAMS, work


def workflow(skip: list[str] | None = None) -> None:
    """MyQueue workflow."""
    from myqueue.workflow import run
    GPU_PARAMS = PARAMS.copy()
    GPU_PARAMS['parallel'] = dict(gpu=True)

    for name, (_, _, cores, _) in REFERENCES.items():
        if skip and name in skip:
            continue
        tmax = '2h'
        nodename = 'a100'
        processes = 1
        cores = 32
        gpus = 1

        if name in {'Au309', 'C676', 'C1588'}:
            cores *= 4
            processes *= 4
            gpus *= 4

        run(function=work,
            args=[name],
            kwargs={'params': GPU_PARAMS},
            cores=cores,
            processes=processes,
            gpus=gpus,
            tmax=tmax,
            nodename=nodename,
            name=name,
            creates=[f'{name}.json'])
