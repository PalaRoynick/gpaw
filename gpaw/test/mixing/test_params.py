import pytest
from ase.build import molecule

from gpaw import GPAW
from gpaw.dft import MSR1, NoMixing, Pulay
from gpaw.mixer import Mixer as OldMixer


@pytest.mark.parametrize('mixer',
                         [NoMixing(),
                          Pulay(beta=0.5),
                          MSR1(beta=0.5),
                          OldMixer(),
                          {'backend': 'pulay', 'beta': 0.5},
                          {'beta': 0.5},
                          {'name': 'multi'}])
@pytest.mark.parametrize('magmoms',
                         [None,
                          [1, 1],
                          [(0, 0, 1), (0, 0, 1)]])
def test_params(mixer, magmoms):
    atoms = molecule('H2')
    atoms.center(vacuum=2)
    calc = GPAW(mixer=mixer,
                xc='LDA',
                mode='pw')
    atoms.calc = calc
    atoms.get_potential_energy()
