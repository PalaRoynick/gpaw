"""CPU + NVIDIA CUDA template; projection and Bloch assembly remain on CPU."""
import os
from pathlib import Path

mpi = True
compiler = 'mpicxx'
libraries = ['blas', 'lapack', 'xc', 'fftw3', 'cudart', 'cublas']
fftw = True
scalapack = False
extra_compile_args = ['-O3']
gpu = True
gpu_target = 'cuda'
gpu_compiler = str(
    Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/nvcc')
cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
include_dirs = [str(cuda / 'include')]
library_dirs = [str(cuda / 'lib64')]
runtime_library_dirs = [str(cuda / 'lib64')]
# Examples: V100=70, A100=80, RTX 30=86, H100=90. Match your GPU/toolkit.
arch = os.environ.get('GPAW_CUDA_ARCH', '80')
gpu_compile_args = ['-O3', '-gencode', f'arch=compute_{arch},code=sm_{arch}',
                    '-gencode', f'arch=compute_{arch},code=compute_{arch}']
# Host-staged MPI is safer unless the installed MPI was built for CUDA buffers.
undef_macros = ['GPAW_GPU_AWARE_MPI']

ccbin = os.environ.get('GPAW_CUDA_CCBIN')
if ccbin:
    gpu_compile_args += ['-ccbin', ccbin]
