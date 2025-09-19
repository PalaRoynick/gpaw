"""Tests of the GPU debug system"""

from gpaw import GPAW_GPU_DEBUG
import gpaw.gpu.debug


def test_env_variable():
    """Tests that GPAW_GPU_DEBUG envvar enables the debug system globally.
    """
    enabled = gpaw.gpu.debug.is_debug_system_enabled()

    if GPAW_GPU_DEBUG:
        assert enabled
    else:
        assert not enabled


def test_toggle():
    """"""
    gpaw.gpu.debug.toggle_debug_system(True)
    assert gpaw.gpu.debug.is_debug_system_enabled()

    gpaw.gpu.debug.toggle_debug_system(False)
    assert not gpaw.gpu.debug.is_debug_system_enabled()


def test_context_manager():
    """Test the `with GPUDebug():` feature"""
    gpaw.gpu.debug.toggle_debug_system(False)

    with gpaw.gpu.debug.GPUDebug():
        assert gpaw.gpu.debug.is_debug_system_enabled()
