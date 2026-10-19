from myqueue.workflow import run


def workflow():
    with run(script='ethanol_in_water.py', cores=4):
        run(script='check.py')
    with run(script='build-slab.py'):
        run(script='plot-overview.py')
        with run(script='run-uncorrected.py', cores=8, tmax='1h'):
            run(script='plot-potential.py')
        with run(script='run-corrected.py', cores=8, tmax='1h'):
            run(script='plot-layer.py')
            run(script='plot-corrected.py')
