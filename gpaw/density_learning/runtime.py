"""Process facts for interpreting CPU usage; no inferred core utilization."""

import os
import socket


def runtime_snapshot():
    result = {
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
