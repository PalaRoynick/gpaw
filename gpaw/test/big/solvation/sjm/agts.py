def workflow():
    from myqueue.workflow import run

    from gpaw.test.big.solvation.sjm import extension as ext
    run(script='run_test_sjm.py', cores=4, tmax='15h')
    for test in [ext.test_change_target_potential,
                 ext.test_switch_to_constant_charge,
                 ext.test_change_before_first_calculation,
                 ext.test_change_drops_a_staged_step,
                 ext.test_sjm_zero_charge_forces_match_solvation]:
        run(function=test, cores=4)
    for flag in [False, True]:
        run(function=ext.test_step_reaches_the_next_geometry, args=[flag],
            name=f'test_step_reaches_the_next_geometry-{flag}', cores=4)
        run(function=ext.test_region_follows_atoms, args=[flag],
            name=f'test_region_follows_atoms-{flag}', cores=4)
