from myqueue.workflow import run


def workflow():
    r2 = run(script='part1/n2_on_metal.py', tmax='2h')
    r3 = run(script='part3/neb.py', tmax='1h', cores=40, deps=[r2])
    r4 = run(script='part_extra/relax.py', tmax='1h', cores=24, deps=[r3])
    r5 = run(script='part_extra/vibrations_initial.py',
             tmax='1h', cores=24, deps=[r4])
    r6 = run(script='part_extra/vibrations_final.py',
             tmax='1h', cores=24, deps=[r4])
    run(script='part_extra/analysis.py', deps=[r5, r6])

    r7 = run(script='part_extra/generate_transition_state_xyz.py',
             tmax='12h', cores=96, deps=[r3])
    run(script='part_extra/vibrations_transition.py',
        tmax='12h', cores=24, deps=[r7])
