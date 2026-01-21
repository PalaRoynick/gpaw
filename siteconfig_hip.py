fftw = True
scalapack = True
libraries = ['xc', 'blas', 'fftw3', 'scalapack-openmpi']

extra_compile_args = []
library_dirs = []
runtime_library_dirs = []
include_dirs = []
gpu_include_dirs = []

gpu = True
gpu_target = 'hip-amd'
gpu_compiler = 'hipcc'
gpu_compile_args = ['-O3',
                    '-g',
                    '--offload-arch=gfx90a',
                    ]
libraries += ['amdhip64', 'hipblas']

library_dirs += ['/opt/rocm/lib']
runtime_library_dirs += ['/opt/rocm/lib']
include_dirs += ['/opt/rocm/include']

libraries += ['roctx']
gpu_compile_args += ['-DGPAW_ROCTX']

# magma = True
# libraries += ['magma']

# library_dirs += ['/usr/local/magma/lib']
# runtime_library_dirs += ['/usr/local/magma/lib']
# include_dirs += ['/usr/local/magma/include']
# gpu_include_dirs += ['/usr/local/magma/include']

# #Seems to confuse cpupy
# undef_macros += ['GPAW_GPU_AWARE_MPI']
