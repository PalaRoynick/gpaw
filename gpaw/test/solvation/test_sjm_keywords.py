"""The keys of SJM's potentiostat dict: what they are, and what they are not.

The keys are shared out between the potentiostat and its default parts.
These tests pin down that set, so that growing or losing a key is a decision
rather than an accident, and so that a key the legacy calculator has but
this one does not is an error instead of a silently ignored setting.  None
of them needs a calculator.
"""
import numpy as np
import pytest

from gpaw.extensions.electrochemistry import (
    SJM, DirectOptimizer, JelliumCharging, PotentialOptimizer, PotentialProbe,
    Potentiostat)
from gpaw.extensions.electrochemistry.sjm import changeable
from gpaw.solvation import GaussianDerivativeDipole, Solvation


# A solvent that is not there: no cavity, and a dielectric of one.  The
# AGTS tests in gpaw/test/big/solvation/sjm/ use it too.
class NoCavity:
    depends_on_el_density = False

    def set_grid_descriptor(self, gd):
        pass

    def allocate(self):
        pass

    def update_atoms(self, atoms, log):
        pass

    def update(self, atoms, density):
        pass

    def communicate_vol_surf(self, comm):
        pass

    def summary(self, log):
        pass

    def todict(self):
        return {}


class Vacuum:
    def set_grid_descriptor(self, gd):
        self.eps_gradeps = [gd.zeros() for _ in range(4)]
        self.eps_gradeps[0][:] = 1.0

    def allocate(self):
        pass

    def todict(self):
        return {}


#: Every key, at a non-default value where that is meaningful.
keys = {
    'target_potential': 3.0,
    'excess_electrons': 1.0,
    'tol': 0.01,
    'max_iters': 100,
    'max_potential_stepsize': 1.0,
    'max_reruns': 5,
    'grand_output': False,
    'jelliumregion': {'top': -2., 'bottom': -4., 'thickness': None,
                      'fix_bottom': True},
    'always_adjust': True,
    'slope': 5.0,
    'mixer': 1.0,
    'slope_regression_depth': 3,
}

#: Keys the legacy ``sj`` dict accepts and this implementation does not.
#: Passing one of these and having it quietly ignored would be the worst
#: outcome: the calculation would run, and mean something other than what
#: was asked for.
legacy_only = {
    'dirichlet': True,
    'pot_ref': 'CIP',
    'cip': {'filter': 10},
    'fdt': {'dt': 0.5},
    'max_step': 1.0,
    'previous_electrons': [],
    'previous_potentials': [],
}

solvent = dict(cavity=NoCavity(), dielectric=Vacuum(), interactions=[])


class MyOptimizer(PotentialOptimizer):
    pass


class MyProbe(PotentialProbe):
    pass


def sjm(**potentiostat) -> SJM:
    """An SJM object with no solvent, built without a calculator."""
    return SJM(solvation=solvent, potentiostat=potentiostat)


def test_key_set_has_not_drifted():
    assert set(changeable) == set(keys), (
        'SJM grew or lost a key; update keys here and check whether the new '
        'one needs a test of its own.')


def test_every_key_reaches_its_part():
    # ... given to SJM, and through change() before a calculation:
    changed = sjm()
    changed.change(potentiostat=keys)
    for obj in [sjm(**keys), changed]:
        pot = obj.potentiostat
        live = {**pot.todict(), **pot.charging.todict(),
                **pot.optimizer.todict()}
        assert {key: live[key] for key in keys} == keys


def test_todict_builds_the_same_potentiostat_again():
    pot = sjm(**keys).potentiostat
    assert Potentiostat.from_dict(pot.todict()).todict() == pot.todict()
    # The written dict holds parts, which take their keys as well:
    pot = Potentiostat.from_dict({'optimizer': {'slope': -20.0}, 'mixer': 0.5})
    assert (pot.optimizer.slope, pot.optimizer.mixer) == (-20.0, 0.5)


def test_nothing_to_read_yet():
    obj = sjm()
    assert np.isnan(obj.get_electrode_potential())
    with pytest.raises(RuntimeError, match='No results'):
        obj.write_sjm_traces()


def test_unknown_key_is_rejected():
    with pytest.raises(TypeError):
        sjm(excess_electron=-0.01)  # note the missing 's'
    with pytest.raises(ValueError, match='Unexpected potentiostat key'):
        sjm().change(potentiostat={'excess_electron': -0.01})
    # The parts are set when the SJM object is created:
    with pytest.raises(ValueError, match='Unexpected potentiostat key'):
        sjm().change(potentiostat={'optimizer': DirectOptimizer()})


@pytest.mark.parametrize('key', sorted(legacy_only))
def test_legacy_only_key_is_rejected(key):
    value = legacy_only[key]
    with pytest.raises(TypeError, match='unexpected keyword'):
        sjm(**{key: value})
    with pytest.raises(ValueError, match='Unexpected potentiostat key'):
        sjm().change(potentiostat={key: value})


def test_the_legacy_only_list_is_complete():
    """Every key the legacy calculator has that this one lacks is above.

    CIP-DFT in particular is legacy-only, as ``sjm.rst`` says.
    """
    from gpaw.solvation.sjm import OldSJM

    legacy = set(OldSJM._sj_default_parameters)
    assert legacy - set(changeable) == set(legacy_only)
    assert {'cip', 'pot_ref'} <= set(legacy_only)


def test_parts_of_your_own():
    opt, probe = MyOptimizer(), MyProbe()
    pot = sjm(optimizer=opt, potentialprobe=probe).potentiostat
    assert pot.optimizer is opt and pot.potentialprobe is probe
    assert isinstance(pot.charging, JelliumCharging)
    # They are written by name, but cannot be read back:
    assert pot.todict()['optimizer'] == {'name': 'MyOptimizer'}
    with pytest.raises(ValueError, match='Unknown SJM part MyOptimizer'):
        Potentiostat.from_dict(pot.todict())

    # A potentiostat of your own is taken as it is:
    class MyPotentiostat(Potentiostat):
        pass

    pot = MyPotentiostat()
    assert SJM(solvation=solvent, potentiostat=pot).potentiostat is pot


@pytest.mark.parametrize('part, key', [
    ('charging', 'excess_electrons'), ('charging', 'jelliumregion'),
    ('optimizer', 'target_potential'), ('optimizer', 'slope'),
    ('optimizer', 'mixer'), ('optimizer', 'slope_regression_depth'),
    ('optimizer', 'tol'), ('optimizer', 'max_potential_stepsize'),
    ('optimizer', 'max_iters'), ('optimizer', 'max_reruns')])
def test_a_part_and_the_keys_of_its_default_conflict(part, key):
    with pytest.raises(ValueError, match=f'{key} configure the default'):
        sjm(**{part: object(), key: keys[key]})


def test_change_refuses_a_setting_the_optimizer_lacks():
    obj = sjm(optimizer=MyOptimizer())
    with pytest.raises(ValueError, match='MyOptimizer has no mixer'):
        obj.change(potentiostat={'mixer': 0.5})
    obj.change(potentiostat={'slope': -20.0})  # every optimizer has one
    assert obj.potentiostat.optimizer.slope == -20.0


def test_change_of_target():
    obj = sjm()
    pot = obj.potentiostat
    # A step staged by always_adjust aims at the old target:
    pot.pending_excess_electrons = 0.3
    obj.change(potentiostat={'target_potential': None})
    assert pot.pending_excess_electrons is None
    # The first guess of the charge needs a slope that lowers the potential
    # as electrons are added:
    pot.electrode_potential = 2.7
    for slope in [0.0, 3.0]:
        obj.change(potentiostat={'slope': slope, 'target_potential': 4.5})
        assert pot.excess_electrons == 0.0
    obj.change(potentiostat={'slope': -30.0, 'target_potential': 4.2})
    assert pot.excess_electrons == pytest.approx(-0.05)


def test_solvation_and_its_dipole_layer():
    # A dict builds a Solvation, which gets the SJM dipole layer:
    solvation = sjm().solvation
    assert isinstance(solvation, Solvation)
    assert solvation.dipolelayer.zero_vacuum
    # ... while a layer of your own is kept:
    layer = GaussianDerivativeDipole()
    obj = SJM(solvation=Solvation(**solvent, dipolelayer=layer))
    assert obj.solvation.dipolelayer is layer
    # ... and no solvation at all gives water, kept from behind the slab:
    solvation = SJM().solvation
    assert solvation.dielectric.epsinf == 78.36
    u = solvation.cavity.effective_potential
    assert type(u).__name__ == 'SJMPower12Potential' and u.unsolv_backside
    assert u.atomic_radii == {'H': 1.09} and u.u0 == 0.18
