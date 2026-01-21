import subprocess
import sys

libraries = []
library_dirs = []
runtime_library_dirs = []
include_dirs = []
extra_link_args = []

no_defaults_args = False  # set to True if you want to set absolutely everything manually

use_cpp = True
use_openmp = True
with_gpu = True  # gpu = True still works but is deprecated
gpu_target = 'cuda'

with_mpi = True  # mpi = True still works but is deprecated
mpi_compiler = 'mpic++'  # Optional, will use this to extract mpi paths

compiler = 'nvcc'
extra_compile_args = ['-forward-unknown-opts',
                      '-arch=sm_80',
                      '-Wno-deprecated-gpu-targets',
                      '-Wno-deprecated-declarations',
                      '-Wno-unknown-pragmas',
                      '-Wno-implicit-fallthrough',  # in libxc.c, seems harmless and done on purpose
                      # Suppress warning from Python.h, GPAW leaves some PyTypeObject fields uninitialized (which looks OK)
                      '-Wno-missing-field-initializers'
                      ]

libraries += ['xc', 'blas']

fftw = True
libraries += ['fftw3']
scalapack = True
libraries += ['scalapack-openmpi']

if use_openmp:
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

magma = True
libraries += ['magma']
library_dirs += ['/usr/local/magma/lib']
runtime_library_dirs += ['/usr/local/magma/lib']
include_dirs += ['/usr/local/magma/include']

# cublas needs some of these for whatever reason
library_dirs += ['/usr/local/cuda/lib64']
runtime_library_dirs += ['/usr/local/cuda/lib64']
include_dirs += ['/usr/local/cuda/include']


# example undef:
#undef_macros += ['GPAW_GPU_AWARE_MPI']
