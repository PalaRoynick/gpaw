from myqueue.workflow import run


def workflow():
    run(script='eads.py', tmax='1h', cores=8)
    run(script='convergence.py', tmax='1h', cores=8)
