"""Utils for using the C++ GPU debug system from Python.
"""

from gpaw.gpu import cupy_is_fake
from typing import TYPE_CHECKING
import traceback

try:
    # Hack to make type checker always see the dummy functions
    if TYPE_CHECKING:
        raise ImportError

    from gpaw.cgpaw import (is_debug_system_enabled,
                            configure_debug_stream_delay_microseconds,
                            ensure_object_not_in_use,
                            ensure_buffer_not_in_use)

    # We will wrap toggle_debug_system() in a Python function that also
    # activates Cupy's memory hooks
    from gpaw.cgpaw import toggle_debug_system as _toggle_debug_system

except ImportError:
    def _toggle_debug_system(enabled: bool) -> None:
        pass

    def is_debug_system_enabled() -> bool:
        return False

    def configure_debug_stream_delay_microseconds(microsecs: float) -> None:
        pass

    def ensure_object_not_in_use(obj) -> bool:
        return True

    def ensure_buffer_not_in_use(buffer: int) -> bool:
        return True


if not cupy_is_fake:
    from gpaw.gpu import cupy as cp

    class CupyDebugHook(cp.cuda.memory_hook.MemoryHook):
        """"""

        name = 'GPAWDebugHook'

        def __init__(self, take_tracebacks: bool = True):
            self.take_tracebacks = take_tracebacks
            self.alloc_traceback = {}

        def activate_hooks(self):
            """"""
            self.__enter__()

        def disable_hooks(self):
            """"""
            self.__exit__()

        def alloc_postprocess(self, **kwargs):
            """"""
            pass

        def malloc_postprocess(self, **kwargs):
            """"""
            if self.take_tracebacks:
                mem_ptr = kwargs['mem_ptr']
                self.alloc_traceback[mem_ptr] = ''.join(traceback.format_stack())

        def free_preprocess(self, **kwargs):
            """"""
            mem_ptr = kwargs['mem_ptr']  # Integer
            if not ensure_buffer_not_in_use(mem_ptr):
                print("!!! Cupy is freeing pointer that is still in use!")

                if self.take_tracebacks:
                    print("Malloc traceback:\n", self.alloc_traceback[mem_ptr])
                    print("\nFree traceback:\n",
                          ''.join(traceback.format_stack()))
                    print("\n", flush=True)
                raise RuntimeWarning("Data race!!")

        def free_postprocess(self, **kwargs):
            """"""
            mem_ptr = kwargs['mem_ptr']  # Integer
            if not ensure_buffer_not_in_use(mem_ptr):
                print("!!! Cupy freed a buffer that is still in use!")

                if self.take_tracebacks:
                    print("Malloc traceback:\n", self.alloc_traceback[mem_ptr])
                    print("\nFree traceback:\n",
                          ''.join(traceback.format_stack()))
                    print("\n", flush=True)
                raise RuntimeWarning("Data race!!")

            if self.take_tracebacks:
                self.alloc_traceback.pop(mem_ptr)

if not cupy_is_fake:
    g_cupy_debug_hooks = CupyDebugHook()
else:
    g_cupy_debug_hooks = None


def toggle_debug_system(enabled: bool):
    """Toggles the global debug system on/off"""
    _toggle_debug_system(enabled)

    global g_cupy_debug_hooks
    if g_cupy_debug_hooks is not None:
        if enabled:
            g_cupy_debug_hooks.activate_hooks()
        else:
            g_cupy_debug_hooks.disable_hooks()


class GPUDebug:
    """Context manager for the debug system"""

    def __init__(self):
        # Combine with Cupy's debug hooks
        if not cupy_is_fake:
            self.cupy_memory_hook = CupyDebugHook(take_tracebacks=True)
        else:
            self.cupy_memory_hook = None

    def __enter__(self):
        self.should_disable_system = False
        if not is_debug_system_enabled():
            toggle_debug_system(True)
            self.should_disable_system = True

        if self.cupy_memory_hook is not None:
            self.cupy_memory_hook.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.cupy_memory_hook is not None:
            self.cupy_memory_hook.__exit__(exc_type, exc_val, exc_tb)

        # Disable the debug system if it was activated by us
        if self.should_disable_system:
            toggle_debug_system(False)

        return False
