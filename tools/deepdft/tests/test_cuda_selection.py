"""Select site CUDA without accidentally downloading a different toolkit."""

import os
from pathlib import Path
import subprocess

import pytest


TOOLS = Path(__file__).resolve().parents[1]


def toolkit(tmp_path, version):
    root = tmp_path / 'cuda'
    (root / 'bin').mkdir(parents=True)
    nvcc = root / 'bin/nvcc'
    nvcc.write_text(f'#!/bin/sh\necho "Cuda compilation tools, release {version}, V{version}.0"\n')
    nvcc.chmod(0o755)
    (root / 'include').mkdir()
    (root / 'include/cuda_runtime.h').touch()
    (root / 'lib64').mkdir()
    (root / 'lib64/libcudart.so').touch()
    (root / 'lib64/libcublas.so').touch()
    return root


@pytest.mark.parametrize('version', ['12.6', '12.9', '13.0', '11.8'])
def test_version_gate(tmp_path, version):
    root = toolkit(tmp_path, version)
    result = subprocess.run(
        ['bash', '-c', 'source "$1"; deepdft_check_cuda_version "$2"',
         'test', str(TOOLS / 'select_cuda.sh'), str(root)], capture_output=True, text=True)
    assert (result.returncode == 0) == (version in {'12.6', '12.9'})


def test_site_nvcc_symlink_resolves_toolkit(tmp_path):
    root = toolkit(tmp_path, '12.6')
    module_bin = tmp_path / 'module-bin'
    module_bin.mkdir()
    (module_bin / 'nvcc').symlink_to(root / 'bin/nvcc')
    environment = {**os.environ, 'PATH': str(module_bin) + ':/usr/bin:/bin'}
    result = subprocess.check_output(
        ['bash', '-c', 'source "$1"; deepdft_select_cuda',
         'test', str(TOOLS / 'select_cuda.sh')], env=environment, text=True)
    assert result.strip() == str(root)


def test_unsupported_site_cuda_fails_before_state_or_download(tmp_path):
    root = toolkit(tmp_path, '13.0')
    state = tmp_path / 'unused-state'
    result = subprocess.run(
        ['bash', str(TOOLS / 'build_gpaw_cuda.sh'), '--cuda-root', str(root),
         '--state-dir', str(state)], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'cannot target sm_70' in result.stderr
    assert not state.exists()


def test_hpc_sdk_wrapper_selects_sibling_cuda(tmp_path):
    root = toolkit(tmp_path, '12.9')
    sdk = tmp_path / 'sdk/25.7'
    destination = sdk / 'cuda/12.9'
    destination.parent.mkdir(parents=True)
    root.rename(destination)
    wrapper = sdk / 'compilers/bin/nvcc'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('#!/bin/sh\necho "Cuda compilation tools, release 12.9, V12.9.86"\n')
    wrapper.chmod(0o755)
    environment = {**os.environ, 'PATH': str(wrapper.parent) + ':/usr/bin:/bin'}
    result = subprocess.check_output(
        ['bash', '-c', 'source "$1"; deepdft_select_cuda',
         'test', str(TOOLS / 'select_cuda.sh')], env=environment, text=True)
    assert result.strip() == str(destination)
