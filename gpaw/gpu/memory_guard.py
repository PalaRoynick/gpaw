from contextlib import contextmanager
import os
from gpaw.gpu import cupy
import numpy as np
import traceback

GPAW_NO_ALLREDUCE = bool(int(os.environ.get('GPAW_NO_ALLREDUCE', 0)))

print('GPAW_NO_ALLREDUCE', GPAW_NO_ALLREDUCE)
"""
# This will reproduce the memory corruption at the end of buffers
# probably meaning that one kernel writes 256 bytes before its giver pointer.
# probably the all reduce kernel.

from gpaw import GPAW
from ase.build import bulk
from gpaw.mpi import world, GPUMemoryGuard
import cupy
cupy.cuda.set_allocator(debug_malloc)

atoms = bulk('Si')
atoms.rattle(0.1, seed=32)
atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 320}, kpts=(4,4,4), xc='PBE',
                  convergence={'density': 1e-6},
                  parallel=dict(gpu=True, kpt=2))
from ase.optimize import BFGS
relax = BFGS(atoms)
with GPUMemoryGuard(2048):
    relax.run()
"""

from collections import defaultdict
from cupy.cuda.memory import _malloc


def get_debug_malloc(pool, front_bytes, back_bytes, print_malloc):
    pool_ptr = defaultdict(int)
 
    def debug_malloc(size):
        total_size = front_bytes + back_bytes + size
        if pool:
            pointer = pool.malloc(total_size)
        else:
            pointer = _malloc(total_size)
        fill = cupy.ndarray((total_size,),
                            memptr=pointer,
                            dtype=np.uint8)
        pool_ptr[pointer.ptr] += 1
        if print_malloc:
            print('malloc', size, pointer.ptr)
            print(traceback.format_stack())
            print(pool_ptr)
            print('end malloc')
        fill[:] = 85
        # synchronize not really necessary but just in case for MPI stuff
        cupy.cuda.runtime.deviceSynchronize()
        return pointer + front_bytes
    return debug_malloc


class GuardContext:
    def check_array_mem(self, a):
        if _guard is not None:
            _guard.check_array_mem(a)


guard = GuardContext()
_guard = None

@contextmanager
def GPUMemoryGuard(front_bytes=1024, back_bytes=1024, pooled=True, print_malloc=False, abort=True):
    
    from cupy.cuda.memory import MemoryPointer, MemoryPool
    from cupy.cuda.memory import get_allocator

    pool = MemoryPool() if pooled else None
    old_allocator = get_allocator()
    cupy.cuda.set_allocator(get_debug_malloc(pool, front_bytes, back_bytes, print_malloc))

    class Guard:
        def check_array_mem(self, a):
            if not isinstance(a, cupy.ndarray):
                return
            mem = a.data.mem
            total_size = mem.size
            cupy.cuda.runtime.deviceSynchronize()
            fill = cupy.ndarray((total_size,),
                                memptr=MemoryPointer(mem, 0),
                                dtype=np.uint8)

            fault = False
            if cupy.all(fill[:front_bytes] == 85):
                # print('front of buffer correct', a.data.mem.ptr, a.data.mem.size)
                pass
            else:
                print('front of buffer corrupted', fill[:front_bytes],
                      a.data.mem.ptr, a.data.mem.size)
                from gpaw.mpi import world
                f = open('start_corruption_%02d.txt' % world.rank, 'a')
                s = fill[:extra_bytes].get().tobytes().hex()
                print(repr(s), file=f)
                stack_trace = ''.join(traceback.format_stack())
                print('^', stack_trace, file=f)
                f.close()
                fault = True

            if cupy.all(fill[-back_bytes:] == 85):
                # print('end of buffer correct', a.data.mem.ptr, a.data.mem.size)
                pass
            else:
                print('end of buffer corrupted', a.data.mem.ptr, a.data.mem.size)
                i = cupy.argmax(fill[-back_bytes:] != 85)
                print('corruption start at', i, fill[-back_bytes:][i:])
                #cupy.cuda.runtime.deviceSynchronize()
                #print('corruption start at after sync', i, fill[-back_bytes:][i:])
                from gpaw.mpi import world
                f = open('end_corruption_%02d.txt' % world.rank, 'a')
                s = fill[-back_bytes:][i:].get().tobytes().hex()
                print('corruption start at', i, 'which means', back_bytes - i,
                      'bytes from the end of the buffer', file=f)
                print(repr(s), file=f)
                stack_trace = ''.join(traceback.format_stack())
                print('^', stack_trace, file=f)
                f.close()
                fault = True

            if fault:
                if abort:
                    raise RuntimeError('Memory corruption detected.')
                # Refill to not to trigger again
                fill[-back_bytes:] = 85
                fill[:front_bytes] = 85

    _guard = Guard()
    globals()['_guard'] = _guard
    if pooled:
        from cupy.cuda.memory_hook import MemoryHook
        from cupy.cuda.memory import UnownedMemory, MemoryPointer
        class PrefillPooledMemory(MemoryHook):
            name = 'PrefillPooledMemory'
            def free_preprocess(self, *, device_id, mem_size, mem_ptr, pmem_id):
                mem = UnownedMemory(mem_ptr, mem_size, self, device_id)
                ptr = MemoryPointer(mem, 0)
                fill = cupy.ndarray((mem_size,),
                                    memptr=ptr,
                                    dtype=np.uint8)
                fill[:] = 85
        with PrefillPooledMemory():
            yield _guard
    else:
        yield _guard
    globals()['_guard'] = lambda a: None
    cupy.cuda.set_allocator(old_allocator)
