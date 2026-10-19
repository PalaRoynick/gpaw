import numpy as np

from gpaw.response.g0w0 import compare_gw_cache_inputs, compare_inputs


def test_compare_inputs():
    A = {'A': [[1.0 + 0.9e-14, 10], np.array([1.0 + 0.9e-14, 10])],
         'C': [1, 2, 3]}
    B = {'A': [[1.0, 10], np.array([1.0, 10])], 'C': [1, 2, 3]}
    assert compare_inputs(A, B)


def test_compare_gw_cache_inputs_fully_coupled_extrapolation():
    legacy = {'ecut_e': [100.0, 150.0, 200.0]}
    boolean = {**legacy, 'extrapolate': True}
    components = {
        **legacy,
        'extrapolate': {'G': True, 'chi0': True, 'epsilon': True}}

    assert compare_gw_cache_inputs(legacy, boolean)
    assert compare_gw_cache_inputs(boolean, components)
    assert compare_gw_cache_inputs(components, legacy)

    epsilon_only = {
        **legacy,
        'extrapolate': {'G': False, 'chi0': False, 'epsilon': True}}
    assert not compare_gw_cache_inputs(legacy, epsilon_only)
