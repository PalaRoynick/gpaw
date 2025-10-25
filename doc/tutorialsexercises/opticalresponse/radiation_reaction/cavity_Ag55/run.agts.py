from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=48, tmax='30m'):
        with run(script='td.py', cores=48, tmax='4h'):
            with run(script='spec.py') as spec:
                run(script='plot_spec.py')
        with run(script='tdcavity.py', cores=48, tmax='4h'):
            with run(script='spec.py') as spec:
                run(script='plot_spec.py')
