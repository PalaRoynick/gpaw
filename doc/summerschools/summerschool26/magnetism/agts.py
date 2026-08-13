from myqueue.workflow import run


def workflow():
    with run(script='xyz.py'):
        run(script='part1_ferromagnetism/relax.py')
        fm = run(script='part1_ferromagnetism/gs_fm.py')
        afm = run(script='part1_ferromagnetism/gs_afm.py')
        run(script='part1_ferromagnetism/Tc_MF.py', deps=[fm, afm])
        run(script='part1_ferromagnetism/anisotropy.py', deps=[fm])
        run(script='part1_ferromagnetism/anisotropy_plot.py', deps=[fm])
        run(script='get_Tc_fit.py')

        with run(script='part2_noncollinear/VI2_gs_teacher.py'):
            run(script='part2_noncollinear/VI2_anisotropy.py')
            run(script='part2_noncollinear/VI2_afm_teacher.py')
            run(script='part2_noncollinear/VI2_noncol.py', cores=4)
