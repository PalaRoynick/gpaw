import pytest
from ase import Atoms

from gpaw.spinorbit import soc_eigenstates
from gpaw import GPAW


@pytest.mark.soc
def test_spinorbit_Kr():
    a = Atoms('Kr')
    a.center(vacuum=3.0)

    calc = GPAW(mode='pw', xc='LDA')

    a.calc = calc
    a.get_potential_energy()

    e_n = calc.get_eigenvalues()
    e_m = soc_eigenstates(calc).eigenvalues()[0]
    print(e_n)
    print(e_m)

    e1 = -22.08477236
    e2 = -8.27112345
    e2a = -8.72318224
    e2b = -8.04509405

    assert e_n == pytest.approx([e1, e2, e2, e2], abs=1.0e-3)
    assert e_m == pytest.approx(
        [e1, e1, e2a, e2a, e2b, e2b, e2b, e2b], abs=1.0e-3)


if __name__ == '__main__':
    test_spinorbit_Kr()
