"""Utils for using the C++ GPU debug system from Python.
"""

from gpaw.gpu import cupy_is_fake, cupy as cp
from typing import TYPE_CHECKING, Optional
import traceback
import numpy as np

try:
    # Hack to make type checker always see the dummy functions
    if TYPE_CHECKING:
        raise ImportError

    from gpaw.cgpaw import (is_debug_system_enabled,
                            configure_events_take_pyobject_ownership,
                            configure_debug_stream_delay_microseconds,
                            ensure_object_not_in_use,
                            ensure_buffer_not_in_use,
                            get_overlapping_events_as_string)

    # We will wrap toggle_debug_system() in a Python function that also
    # activates Cupy's memory hooks
    from gpaw.cgpaw import toggle_debug_system as _toggle_debug_system

except ImportError:
    def _toggle_debug_system(enabled: bool) -> None:
        pass

    def is_debug_system_enabled() -> bool:
        return False

    def configure_events_take_pyobject_ownership(enabled: bool) -> None:
        pass

    def configure_debug_stream_delay_microseconds(microsecs: float) -> None:
        pass

    def ensure_object_not_in_use(obj) -> bool:
        return True

    def ensure_buffer_not_in_use(buffer: int) -> bool:
        return True

    def get_overlapping_events_as_string(start: int, nbytes: int) -> list[str]:
        return []


def debug_array_overlap_check(arr: np.ndarray | cp.ndarray,
                              assert_success: bool = True,
                              traceback_start=0,
                              traceback_end=10  # None means no limit
                              ) -> None:
    """Checks the GPU debug system if the memory block occupied by the input
    array is currently in use by any event. Prints debug info on failure,
    including stack trace.
    if `assert_success` is True, will raise AssertionError on failure.
    """

    if not is_debug_system_enabled():
        return

    if isinstance(arr, cp.ndarray):
        start = arr.data.ptr
        nbytes = arr.nbytes
    elif isinstance(arr, np.ndarray):
        start = arr.ctypes.data
        nbytes = arr.nbytes
    else:
        raise ValueError("Input must be a Numpy or Cupy array")

    debug_memory_block_check(start,
                             nbytes,
                             assert_success=assert_success,
                             traceback_start=traceback_start,
                             traceback_end=traceback_end)


def debug_memory_block_check(block_start: int,
                             block_nbytes: int,
                             assert_success: bool = True,
                             traceback_start=0,
                             traceback_end=10  # None means no limit
                             ) -> bool:
    """Checks the GPU debug system if the input memory block is currently in
    use by any event.
    Prints debug info on failure, including stack trace.
    if `assert_success` is True, will raise AssertionError on failure.
    """

    overlaps: list = get_overlapping_events_as_string(block_start,
                                                      block_nbytes)

    if overlaps:
        print(f"\n\nData race detected for memory block at {block_start}",
              f" (size {block_nbytes}).")
        print("This block is in async use by the following events:")

        for s in overlaps:
            print(s)

        print("\nTraceback to where Python is trying to use the memory:")
        # stack[0] is the very first function call,
        # stack[-1] the deepest (this function)
        trace = traceback.extract_stack()[:-1]  # drop this function from stack
        trace = trace[-traceback_end:]

        for line in traceback.format_list(trace):
            print(line, end="")
        print("\n", flush=True)

        if assert_success:
            raise AssertionError("DebugSystem: Bad memory overlap detected")

    return not overlaps


if not cupy_is_fake:
    from gpaw.gpu import cupy as cp

    class CupyDebugHook(cp.cuda.memory_hook.MemoryHook):  # type: ignore
        """"""

        name = 'GPAWDebugHook'

        def __init__(self, take_tracebacks: bool = True):
            self.take_tracebacks = take_tracebacks
            self.alloc_traceback: dict = {}

        def activate_hooks(self, take_tracebacks: bool = True):
            """"""
            self.take_tracebacks = take_tracebacks
            self.__enter__()

        def disable_hooks(self):
            """"""
            self.__exit__()

        def alloc_postprocess(self, **kwargs):
            """"""
            pass

        def malloc_postprocess(self, **kwargs):
            """"""

            mem_ptr = kwargs['mem_ptr']  # integer
            nbytes = kwargs['mem_size']

            debug_memory_block_check(mem_ptr, nbytes, assert_success=False)

            if self.take_tracebacks:
                trace = traceback.format_stack()[:-1]
                trace = trace[-10:]
                self.alloc_traceback[mem_ptr] = (
                    ''.join(trace)
                )

        def free_preprocess(self, **kwargs):
            """"""
            mem_ptr = kwargs['mem_ptr']
            nbytes = kwargs['mem_size']
            ok = debug_memory_block_check(mem_ptr,
                                          nbytes,
                                          assert_success=False)

            if not ok:
                if mem_ptr in self.alloc_traceback:
                    print("Cupy malloc trace for this pointer:")
                    print(self.alloc_traceback[mem_ptr])
                    print("\n", flush=True)
                else:
                    # Should not happen since GPAW doesn't alloc GPU buffers
                    # on its own (except for in old FD code maybe).
                    # If we still get here, maybe there is some Cupy slicing
                    # that this tracker doesn't notice??
                    print("\nThis pointer was NOT allocated by Cupy")

        def free_postprocess(self, **kwargs):
            """"""
            mem_ptr = kwargs['mem_ptr']
            nbytes = kwargs['mem_size']

            debug_memory_block_check(mem_ptr, nbytes, assert_success=False)

            if self.take_tracebacks:
                self.alloc_traceback.pop(mem_ptr)

g_cupy_debug_hooks: Optional[CupyDebugHook]

if not cupy_is_fake:
    g_cupy_debug_hooks = CupyDebugHook()
else:
    g_cupy_debug_hooks = None


def toggle_debug_system(enabled: bool):
    """Toggles the global debug system on/off"""
    _toggle_debug_system(enabled)

    if g_cupy_debug_hooks is not None:
        if enabled:
            g_cupy_debug_hooks.activate_hooks(take_tracebacks=True)
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
