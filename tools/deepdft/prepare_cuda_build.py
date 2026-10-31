"""Prepare an isolated source snapshot and V100 CUDA configuration."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess


def prepare(root, state, base, cuda, host):
    source = state / 'source'
    if source.exists():
        raise FileExistsError(f'Use a fresh state directory: {source}')
    includes = [p for p in (cuda / 'include', cuda / 'targets/x86_64-linux/include')
                if (p / 'cuda_runtime.h').is_file()]
    libraries = [p for p in (cuda / 'lib64', cuda / 'lib',
                             cuda / 'targets/x86_64-linux/lib')
                 if (p / 'libcudart.so').exists() and (p / 'libcublas.so').exists()]
    if not includes or not libraries:
        raise ValueError('CUDA runtime/cuBLAS development headers and libraries are required')
    files = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-co', '--exclude-standard', '-z']
    ).decode().split('\0')
    hashes = {}
    for name in sorted(set(files) - {''}):
        path = root / name
        # Submodule sources and generated/binary files are not build inputs.
        if name.startswith('third_party/') or not path.is_file():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'External symlink requires review: {name}')
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (state / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    mpi_includes = shlex.split(subprocess.check_output(
        ['mpicxx', '--showme:incdirs'], text=True).strip())
    config = (base / 'siteconfig.py').read_text()
    config += f'''
# V100: CUDA 12.6, native Volta code plus forward-compatible PTX.
gpu = True
gpu_target = 'cuda'
gpu_compiler = {str(cuda / 'bin/nvcc')!r}
gpu_compile_args = ['-O3', '-ccbin', {host!r},
                    '-gencode', 'arch=compute_70,code=sm_70',
                    '-gencode', 'arch=compute_70,code=compute_70']
gpu_include_dirs = {list(map(str, includes)) + mpi_includes!r}
include_dirs += gpu_include_dirs
library_dirs += {list(map(str, libraries))!r}
runtime_library_dirs += {list(map(str, libraries))!r}
libraries += ['cudart', 'cublas']
# Do not assume a site CUDA-aware MPI transport.
undef_macros += ['GPAW_GPU_AWARE_MPI']
'''
    (state / 'siteconfig.py').write_text(config)
    return libraries


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('state', type=Path)
    p.add_argument('base', type=Path)
    p.add_argument('cuda', type=Path)
    p.add_argument('host')
    args = p.parse_args()
    prepare(args.root, args.state, args.base, args.cuda, args.host)
