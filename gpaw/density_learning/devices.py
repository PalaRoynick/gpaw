"""Explicit backend policies; never change basis or boundary conditions."""


def cuda_status():
    from gpaw.new.c import GPU_ENABLED
    from gpaw.gpu import cupy_is_fake

    return bool(GPU_ENABLED and not cupy_is_fake)


def select_reference_device(requested, mode, ranks, experimental_fd=False):
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Reference device must be auto, cpu or cuda')
    if requested == 'cpu':
        return 'cpu', 'CPU explicitly requested'
    reason = None
    if ranks != 1:
        reason = 'GPU reference profile requires one MPI rank'
    elif mode == 'fd' and not experimental_fd:
        reason = 'FD CUDA requires --experimental-fd-gpu and backend validation'
    elif not cuda_status():
        reason = 'Real CUDA-enabled GPAW and an allocated GPU are unavailable'
    if reason:
        if requested == 'cuda':
            raise RuntimeError(reason)
        return 'cpu', reason
    return 'cuda', 'Single-rank CUDA; host-staged Poisson for FD' if mode == 'fd' else 'Single-rank CUDA PW'
