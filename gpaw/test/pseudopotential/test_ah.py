import pytest
from ase.build import bulk

from gpaw import GPAW, PW


@pytest.mark.stress
@pytest.mark.parametrize('xc', ['LDA', 'PBE'])
def test_pseudopotential_ah(in_tmp_dir, xc):
    si = bulk('Si', 'diamond', a=5.5, cubic=not True)
    si.calc = GPAW(mode=PW(200),
                   setups='ah',
                   xc=xc,
                   kpts=(2, 2, 2))
    f = si.get_forces()
    assert f == pytest.approx(0.0)
    s = si.get_stress()
    assert s[3:] == pytest.approx(0.0)
    refs = {'LDA': 0.299, 'PBE': 0.2957}
    assert s[:3] == pytest.approx(refs[xc], abs=0.001)
    si.calc.write('Si.gpw', 'all')
    GPAW('Si.gpw')
