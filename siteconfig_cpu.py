fftw = True
scalapack = True
libraries = ['xc', 'blas', 'fftw3', 'scalapack-openmpi']

# define_macros += [('GPAW_GPU_ARRAY_DEBUG', 1)]

# extra_compile_args = []
# library_dirs = []
# runtime_library_dirs = []
# include_dirs = []
# gpu_include_dirs = []

# gpu = True
# gpu_target = 'cuda'
# gpu_compiler = 'nvcc'
# gpu_compile_args = ['-O3',
#                     '-g',
#                     '-Wno-deprecated-gpu-targets',
#                     '-Wno-deprecated-declarations',
# 		    '-arch=sm_80',
#                     ]
# libraries += ['cudart', 'cublas']

# # CUDA spam...
# extra_compile_args += ['-Wno-deprecated-declarations']

# library_dirs += ['/usr/local/cuda/lib64']
# runtime_library_dirs += ['/usr/local/cuda/lib64']
# include_dirs += ['/usr/local/cuda/include']

# magma = True
# libraries += ['magma']

# library_dirs += ['/usr/local/magma/lib']
# runtime_library_dirs += ['/usr/local/magma/lib']
# include_dirs += ['/usr/local/magma/include']
# gpu_include_dirs += ['/usr/local/magma/include']

# #Seems to confuse cpupy
# undef_macros += ['GPAW_GPU_AWARE_MPI']
