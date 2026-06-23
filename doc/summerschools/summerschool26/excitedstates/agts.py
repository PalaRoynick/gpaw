from myqueue.workflow import run


def workflow():
    with run(script='solution1.py', tmax='1h', cores=1):
        with run(script='solution1b.py', tmax='1h', cores=1):
            with run(script='solution3.py', tmax='15h', cores=8):
                run(script='solution4.py', tmax='15h', cores=8)
    with run(script='solution2.py', tmax='1h', cores=1):
        pass
