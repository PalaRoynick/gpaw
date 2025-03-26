from myqueue.workflow import run


def workflow():
    run(script='mos2.py', args=[True], cores=2, tmax='1h', nodename='a100G2')
    run(script='mos2.py', args=[False], cores=96, tmax='1h')
