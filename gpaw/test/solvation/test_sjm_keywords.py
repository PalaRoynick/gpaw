"""The SJM keyword set: what it is, and what it deliberately is not.

``_KEYWORDS`` is derived from ``inspect.signature(SJM.__init__)``, so it
tracks the constructor by itself.  These tests pin down what that set is
allowed to contain, so that growing or losing a keyword is a decision rather
than an accident, and so that a keyword the legacy calculator has but this
one does not is an error instead of a silently ignored setting.
"""
import pytest

from gpaw.extensions.electrochemistry import SJM
from gpaw.extensions.electrochemistry.sjm import _KEYWORDS
from gpaw.test.solvation.test_modes import NoCavity, Vacuum

#: Every keyword, at a non-default value where that is meaningful.
KEYWORDS = {
    'target_potential': 3.0,
    'excess_electrons': 1.0,
    'tol': 0.01,
    'max_iters': 100,
    'max_potential_stepsize': 1.0,
    'max_reruns': 5,
    'grand_output': False,
    'jelliumregion': {'top': -2., 'bottom': -4.},
    'always_adjust': True,
    'slope': 5.0,
    'mixer': 1.0,
    'slope_regression_depth': 4,
}

#: Keywords the legacy ``sj`` dict accepts and this implementation does not.
#: Passing one of these and having it quietly ignored would be the worst
#: outcome: the calculation would run, and mean something other than what
#: was asked for.
LEGACY_ONLY = {
    'dirichlet': True,
    'pot_ref': 'CIP',
    'cip': {'filter': 10},
    'fdt': {'dt': 0.5},
    'max_step': 1.0,
    'previous_electrons': [],
    'previous_potentials': [],
}


def sjm(**kwargs) -> SJM:
    """An SJM object with no solvent, built without a calculator."""
    return SJM(cavity=NoCavity(), dielectric=Vacuum(), interactions=[],
               **kwargs)


def test_keyword_set_has_not_drifted():
    assert set(_KEYWORDS) == set(KEYWORDS), (
        'SJM grew or lost a keyword; update KEYWORDS here and check whether '
        'the new one needs a test of its own.')


def test_every_keyword_is_accepted_and_read_back():
    obj = sjm(**KEYWORDS)
    for key, value in KEYWORDS.items():
        assert getattr(obj, key) == value, key

    # ... and the same keywords go through change() before a calculation:
    obj.change(**KEYWORDS)
    for key, value in KEYWORDS.items():
        assert getattr(obj, key) == value, key


def test_keywords_cannot_be_assigned():
    obj = sjm(tol=0.01)
    with pytest.raises(AttributeError, match='change'):
        obj.tol = 0.02
    assert obj.tol == 0.01


def test_no_calculation_to_read_yet():
    obj = sjm()
    with pytest.raises(RuntimeError, match='No calculation'):
        obj.get_electrode_potential()
    with pytest.raises(RuntimeError, match='No calculation'):
        obj.write_sjm_traces()


def test_unknown_keyword_is_rejected():
    with pytest.raises(TypeError):
        sjm(excess_electron=-0.01)  # note the missing 's'

    obj = sjm(target_potential=3.0)
    with pytest.raises(ValueError, match='Unexpected SJM keyword'):
        obj.change(excess_electron=-0.01)


@pytest.mark.parametrize('keyword', sorted(LEGACY_ONLY))
def test_legacy_only_keyword_is_rejected(keyword):
    value = LEGACY_ONLY[keyword]
    with pytest.raises(TypeError, match='unexpected keyword'):
        sjm(**{keyword: value})

    obj = sjm(target_potential=3.0)
    with pytest.raises(ValueError, match='Unexpected SJM keyword'):
        obj.change(**{keyword: value})


def test_the_legacy_only_list_is_complete():
    """Every keyword the legacy calculator has that this one lacks is above.

    CIP-DFT in particular is legacy-only, as ``sjm.rst`` says.
    """
    from gpaw.solvation.sjm import OldSJM

    legacy = set(OldSJM._sj_default_parameters)
    assert legacy - set(_KEYWORDS) == set(LEGACY_ONLY)
    assert {'cip', 'pot_ref'} <= set(LEGACY_ONLY)
