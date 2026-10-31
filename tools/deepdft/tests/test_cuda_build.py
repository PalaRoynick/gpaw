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
    for f in ('include/cuda_runtime.h', 'lib64/libcudart.so', 'lib64/libcublas.so'):
        (cuda / f).touch()
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
