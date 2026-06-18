import pytest

from ase.build import molecule

from gpaw.dft import NoMixing, Pulay, MSR1
from gpaw.mixer import Mixer as OldMixer
from gpaw import GPAW


@pytest.mark.parametrize('mixer',
                         [NoMixing(),
                          Pulay(beta=0.5),
                          MSR1(beta=0.5),
                          OldMixer(),
                          {'backend': 'pulay', 'beta': 0.5},
                          {'beta': 0.5}])
def test_params(mixer):
    atoms = molecule('H2')
    atoms.center(vacuum=2)
    calc = GPAW(mixer=mixer,
                mode='lcao')
    atoms.calc = calc
    atoms.get_potential_energy()
