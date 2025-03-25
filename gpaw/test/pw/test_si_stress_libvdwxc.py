import numpy as np
import pytest
from ase.build import bulk

from gpaw.utilities import compiled_with_libvdwxc
from gpaw import GPAW, PW, Mixer
from gpaw.mpi import world
from gpaw.xc.libvdwxc import vdw_df, libvdwxc_has_spin, libvdwxc_has_stress


skip_reason = "libvdwxc version does not implement stress"
if compiled_with_libvdwxc():
    try:
        skip_cond = not libvdwxc_has_stress()
    except SystemError:
        skip_cond = True
else:
    skip_cond = True


@pytest.mark.stress
@pytest.mark.skipif(skip_cond, reason=skip_reason)
def test_pw_si_stress_libvdwxc(in_tmp_dir, gpaw_new):
    assert libvdwxc_has_spin() and libvdwxc_has_stress()
    si = bulk('Si')
    si.calc = GPAW(mode=PW(200),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=vdw_df(),
                   kpts=(1, 1, 2),  # Run (1, 1, 2) to avoid gamma pt code
                   convergence={'energy': 1e-8},
                   parallel={'domain': min(2, world.size)},
                   txt='si_stress.txt')

    si.set_cell(np.dot(si.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    si.get_potential_energy()

    if not gpaw_new:
        # Trigger nasty bug (fixed in !486):
        si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1

    s_analytical = si.get_stress()
    s_ref = [-0.17538969, -0.08992427, -0.14401403,
             -0.05984851, -0.00058295, 0.04351124]
    # s_ref = si.calc.calculate_numerical_stress(si, 1e-5)
    print(s_analytical)
    s_err = s_analytical - s_ref
    assert np.all(abs(s_err) < 1e-4)
