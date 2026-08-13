from myqueue.workflow import run


def workflow():
    with run(script='xyz.py'):
        run(script='relax.py')
        fm = run(script='gs_fm.py')
        afm = run(script='gs_afm.py')
        run(script='CrI3_plot.py', deps=[fm])
        run(script='Tc_MF.py', deps=[fm, afm])
        run(script='CrI3_anisotropy_teacher.py', deps=[fm])
        run(script='get_Tc_fit.py')

        with run(script='VI2_gs_teacher.py'):
            run(script='VI2_anisotropy.py')
            run(script='VI2_afm_teacher.py')
            run(script='VI2_noncol.py', cores=4)
