from myqueue.workflow import run


def workflow():
    run(script='bilayer.py', args=[True], cores=4, tmax='1h',
        nodename='a100G4')
    run(script='bilayer.py', args=[False], cores=96, tmax='1h')
