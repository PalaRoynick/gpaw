"""Portable Debian/Ubuntu CPU template. Adjust paths to your installation."""
mpi = True
compiler = 'mpicxx'
libraries = ['blas', 'lapack', 'xc', 'fftw3']
fftw = True
scalapack = False
extra_compile_args = ['-O3']
