def workflow():
    from myqueue.workflow import run

    from gpaw.test.big.solvation import forces, forces_new, new_vs_old
    for test in [forces.test_solvation_forces,
                 forces_new.test_pw_solvation_forces,
                 forces_new.test_pw_fd_forces_agree,
                 forces_new.test_fd_solvation_forces,
                 new_vs_old.test_fd_energy_new_vs_old,
                 new_vs_old.test_interaction_energy_consistency]:
        run(function=test, cores=4)
