"""Process facts for interpreting CPU usage; no inferred core utilization."""

import os
import socket
import sys
from pathlib import Path


def loaded_module_paths():
    import gpaw

    extension = sys.modules.get('_gpaw')
    return {
        'python_executable': sys.executable,
        'python_prefix': sys.prefix,
        'gpaw_file': gpaw.__file__,
        'extension_file': getattr(extension, '__file__', None),
    }


def validate_cuda_environment():
    if not os.environ.get('GPAW_CUDA_STATE'):
        return
    paths = loaded_module_paths()
    prefix = Path(sys.prefix).resolve()
    for key in ('gpaw_file', 'extension_file'):
        location = paths[key]
        if location is None or not Path(location).resolve().is_relative_to(prefix):
            raise RuntimeError(
                f'CUDA environment is shadowed: {key}={location}; expected a wheel '
                f'inside {prefix}. Leave all GPAW source checkouts, source the CUDA '
                'build activate.sh, and rerun from an external work directory.'
            )


def runtime_snapshot():
    result = {
        **loaded_module_paths(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "allowed_logical_cpus": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None
        ),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "OMP_DYNAMIC", "MKL_DYNAMIC")
        },
    }
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        result["thread_pools"] = None
        result["thread_pool_note"] = "Optional threadpoolctl is not installed"
    else:
        # Configured pool capacity is not the number of actively running threads.
        result["thread_pools"] = threadpool_info()
    return result
