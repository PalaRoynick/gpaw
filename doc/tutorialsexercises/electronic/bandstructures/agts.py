from myqueue.workflow import run


def workflow():
    run(script='bandstructure.py')
    run(script='soc.py', tmax='1h')
    with (run(script='hse06.py'),
          run(script='hse06_sc.py', cores=24)):
        run(script='plot_hse06.py')
