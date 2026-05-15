import os
GPAW_ENV_ROOT = os.environ["ROIHU_GPAW_ENV_ROOT"]

mpi = True
compiler = 'mpicxx'
libraries = []
library_dirs = []
include_dirs = []
extra_compile_args = [
    '-O3',
    '-march=znver5',
    '-mtune=znver5',
    '-fopenmp',
    ]
extra_link_args = ['-fopenmp']

# blas
libraries += ['openblas']

# scalapack
scalapack = True
libraries += ['scalapack']

# elpa
elpa = True
libraries += ['elpa_openmp']

# fftw
fftw = True
libraries += ['fftw3']
#libraries += ['fftw3_omp']

# libxc
libraries += ['xc']

# libvdwxc
libvdwxc = True
libraries += ['vdwxc']

define_macros += [('GPAW_ASYNC', 1)]

# Path to libraries
include_dirs += [f'{GPAW_ENV_ROOT}/include']
library_dirs += [f'{GPAW_ENV_ROOT}/lib']
extra_link_args += [f'-Wl,-rpath,{GPAW_ENV_ROOT}/lib']
