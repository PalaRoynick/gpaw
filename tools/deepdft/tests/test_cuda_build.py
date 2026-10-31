"""CUDA preparation must preserve the CPU checkout and target Volta explicitly."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location(
    'prepare_cuda_build', Path(__file__).resolve().parents[1] / 'prepare_cuda_build.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_isolated_cuda_snapshot(tmp_path, monkeypatch):
    original = subprocess.check_output

    def output(command, **kwargs):
        if command == ['mpicxx', '--showme:incdirs']:
            return '/private/mpi/include\n'
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, 'check_output', output)
    root, base, cuda, state = [tmp_path / name for name in ('repo', 'base', 'cuda', 'state')]
    for p in (root, base, state, cuda / 'include', cuda / 'lib64', cuda / 'bin'):
        p.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / 'source.py').write_text('value = 1\n')
    subprocess.run(['git', '-C', str(root), 'add', 'source.py'], check=True)
    (root / 'source.py').write_text('value = 2\n')
    (root / '.gitignore').write_text('*.so\n')
    (root / '_gpaw.so').write_bytes(b'cpu-extension')
    for f in ('include/cuda_runtime.h', 'include/cublas_v2.h', 'lib64/libcudart.so', 'lib64/libcublas.so'):
        (cuda / f).touch()
    (cuda / 'bin/nvcc').write_text('#!/bin/sh\necho "release 12.6, V12.6"\n')
    (cuda / 'bin/nvcc').chmod(0o755)
    (base / 'siteconfig.py').write_text('mpi = True\n')
    module.prepare(root, state, base, cuda, '/gcc/bin/g++')
    assert (root / '_gpaw.so').read_bytes() == b'cpu-extension'
    assert not (state / 'source/_gpaw.so').exists()
    assert (state / 'source/source.py').read_text() == 'value = 2\n'
    config = {k: [] for k in ('include_dirs', 'library_dirs', 'runtime_library_dirs',
                             'libraries', 'undef_macros')}
    exec((state / 'siteconfig.py').read_text(), config)
    assert config['gpu'] and config['gpu_target'] == 'cuda'
    assert '/private/mpi/include' in config['gpu_include_dirs']
    assert 'arch=compute_70,code=sm_70' in config['gpu_compile_args']
    assert 'GPAW_GPU_AWARE_MPI' in config['undef_macros']
    assert not any('stubs' in p for p in config['runtime_library_dirs'])
    with pytest.raises(FileExistsError):
        module.prepare(root, state, base, cuda, '/gcc/bin/g++')


def test_hpc_sdk_split_math_libraries(tmp_path):
    cuda = tmp_path / 'sdk/25.7/cuda/12.9'
    math = tmp_path / 'sdk/25.7/math_libs/12.9'
    for p in [cuda / 'bin', cuda / 'include', cuda / 'lib64',
              math / 'include', math / 'lib64']:
        p.mkdir(parents=True)
    (cuda / 'bin/nvcc').write_text('#!/bin/sh\necho "release 12.9, V12.9"\n')
    (cuda / 'bin/nvcc').chmod(0o755)
    (cuda / 'include/cuda_runtime.h').touch()
    (cuda / 'lib64/libcudart.so').touch()
    (math / 'include/cublas_v2.h').touch()
    (math / 'lib64/libcublas.so').touch()
    includes, libraries = module.cuda_layout(cuda)
    assert includes == [cuda / 'include', math / 'include']
    assert libraries == [cuda / 'lib64', math / 'lib64']
    (math / 'lib64/libcublas.so').unlink()
    with pytest.raises(ValueError, match='Missing CUDA development files: libcublas.so') as error:
        module.cuda_layout(cuda)
    assert str(math / 'lib64') in str(error.value)


def test_hpc_sdk_cuda_root_without_nvcc(tmp_path):
    cuda = tmp_path / 'sdk/25.7/cuda'
    compiler = cuda.parent / 'compilers/bin/nvcc'
    compiler.parent.mkdir(parents=True)
    compiler.write_text('#!/bin/sh\necho "release 12.9, V12.9"\n')
    compiler.chmod(0o755)
    for folder in ['include', 'lib64']:
        (cuda / folder).mkdir(parents=True)
    for f in ['include/cuda_runtime.h', 'include/cublas_v2.h',
              'lib64/libcudart.so', 'lib64/libcublas.so']:
        (cuda / f).touch()
    assert module.cuda_compiler(cuda) == compiler
    assert module.cuda_layout(cuda) == ([cuda / 'include'], [cuda / 'lib64'])
