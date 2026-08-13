from myqueue.workflow import run


def workflow():
    with run(script='xyz.py'):
        run(script='part1_ferromagnetism/relax.py')
        fm = run(script='part1_ferromagnetism/gs_fm.py')
        afm = run(script='part1_ferromagnetism/gs_afm.py')
        run(script='part1_ferromagnetism/Tc_MF.py', deps=[fm, afm])
        run(script='part1_ferromagnetism/anisotropy.py', deps=[fm])
        run(script='part1_ferromagnetism/anisotropy_plot.py', deps=[fm])
        run(script='part1_ferromagnetism/Tc_MC.py')

        with run(script='part2_noncollinear/relax.py')
            run(script='part2_noncollinear/anisotropy.py')
            run(script='part2_noncollinear/gs_afm_fm.py')
            run(script='part2_noncollinear/noncollinear_gs.py', cores=4)
