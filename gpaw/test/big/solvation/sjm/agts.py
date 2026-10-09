def workflow():
    from myqueue.workflow import run

    from gpaw.test.big.solvation.sjm import extension as ext
    from gpaw.test.big.solvation.sjm import modes
    run(script='run_test_sjm.py', cores=4, tmax='15h')
    for test in [ext.test_change_target_potential,
                 ext.test_switch_to_constant_charge,
                 ext.test_change_before_first_calculation,
                 ext.test_change_drops_a_staged_step,
                 ext.test_sjm_zero_charge_forces_match_solvation,
                 ext.test_zero_vacuum_does_not_matter,
                 ext.test_grand_output_toggle,
                 ext.test_unknown_keyword,
                 ext.test_nothing_to_report_before_an_scf_loop,
                 ext.test_settings_live_on_the_potentiostat,
                 ext.test_tol_set_at_constant_charge_survives_the_switch,
                 ext.test_parts_of_your_own_are_the_ones_that_run,
                 ext.test_always_adjust_without_target_potential]:
        run(function=test, cores=4)
    for flag in [False, True]:
        run(function=ext.test_step_reaches_the_next_geometry, args=[flag],
            name=f'test_step_reaches_the_next_geometry-{flag}', cores=4)
        run(function=ext.test_region_follows_atoms, args=[flag],
            name=f'test_region_follows_atoms-{flag}', cores=4)
    # Old GPAW has no pw mode, and its SJM fails in parallel (issue #1381):
    for gpaw_new, mode in [(False, 'fd'), (True, 'fd'), (True, 'pw')]:
        run(function=modes.test_h, args=[gpaw_new, mode],
            name=f'test_h-{gpaw_new}-{mode}', cores=4)
    for gpaw_new in [False, True]:
        run(function=modes.test_sjm, args=[gpaw_new, 'fd'],
            name=f'test_sjm-{gpaw_new}-fd', cores=4 if gpaw_new else 1)
