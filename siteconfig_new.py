libraries = []
library_dirs = []
runtime_library_dirs = []
include_dirs = []
extra_link_args = []

no_defaults_args = False  # set to True if you want to set absolutely everything manually

use_cpp = True
openmp = False
gpu = True
gpu_target = 'cuda'

mpi_compiler = 'mpic++'  # Optional, will use this to extract mpi paths
mpi = True

compiler = 'mpic++'
#compiler = 'nvcc'
gpu_compiler_override = 'nvcc'
extra_compile_args = ['-Wno-deprecated-declarations']

gpu_extra_compile_args = ['-arch=sm_80', '-Wno-deprecated-gpu-targets']

libraries += ['xc', 'blas']

fftw = True
libraries += ['fftw3']
scalapack = True
libraries += ['scalapack-openmpi']

magma = True
libraries += ['magma']
library_dirs += ['/usr/local/magma/lib']
runtime_library_dirs += ['/usr/local/magma/lib']
include_dirs += ['/usr/local/magma/include']

# cublas needs some of these for whatever reason, even if compiling with nvcc
library_dirs += ['/usr/local/cuda/lib64']
runtime_library_dirs += ['/usr/local/cuda/lib64']
include_dirs += ['/usr/local/cuda/include']


# example undef:
#undef_macros += ['GPAW_GPU_AWARE_MPI']
