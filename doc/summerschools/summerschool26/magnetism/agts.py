from myqueue.workflow import run


def workflow():
    with run(script='xyz.py'):
        run(script='CrI3_gs_teacher.py')
        fm = run(script='CrI3_fm.py')
        afm = run(script='CrI3_afm.py')
        run(script='CrI3_plot.py', deps=[fm])
        run(script='get_Tc_mf_teacher.py', deps=[fm, afm])
        run(script='CrI3_anisotropy_teacher.py', deps=[fm])
        run(script='get_Tc_fit.py')

        with run(script='VI2_gs_teacher.py'):
            run(script='VI2_anisotropy.py')
            run(script='VI2_afm_teacher.py')
            run(script='VI2_noncol.py', cores=4)
