"""Prepare an isolated source snapshot and V100 CUDA configuration."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess


def cuda_compiler(cuda):
    cuda = cuda.resolve()
    sdk = cuda.parent.parent if cuda.parent.name == 'cuda' else cuda.parent
    for path in (cuda / 'bin/nvcc', sdk / 'compilers/bin/nvcc'):
        if path.is_file():
            return path
    raise ValueError(f'No nvcc found for CUDA root: {cuda}')


def cuda_layout(cuda):
    """Find runtime and cuBLAS independently, including HPC SDK math_libs."""
    cuda = cuda.resolve()
    version_text = subprocess.check_output([str(cuda_compiler(cuda)), '--version'], text=True)
    import re
    match = re.search(r'release ([0-9]+\.[0-9]+),', version_text)
    version = match[1] if match else ''
    sdk = cuda.parent.parent if cuda.parent.name == 'cuda' else cuda.parent
    roots = [cuda, sdk / 'math_libs' / version, sdk / 'math_libs']
    include_candidates = list(dict.fromkeys(
        p for root in roots for p in (root / 'include', root / 'targets/x86_64-linux/include')
    ))
    library_candidates = list(dict.fromkeys(
        p for root in roots for p in (root / 'lib64', root / 'lib',
                                     root / 'targets/x86_64-linux/lib')
    ))
    includes, libraries, missing = [], [], []
    for filename, candidates, selected in [
        ('cuda_runtime.h', include_candidates, includes),
        ('cublas_v2.h', include_candidates, includes),
        ('libcudart.so', library_candidates, libraries),
        ('libcublas.so', library_candidates, libraries),
    ]:
        found = next((p for p in candidates if (p / filename).is_file()), None)
        if found is None:
            missing.append(filename)
        elif found not in selected:
            selected.append(found)
    if missing:
        raise ValueError(
            'Missing CUDA development files: ' + ', '.join(missing)
            + '\nSelected toolkit: ' + str(cuda)
            + '\nSearched includes: ' + ', '.join(map(str, include_candidates))
            + '\nSearched libraries: ' + ', '.join(map(str, library_candidates))
            + '\nSelect the CUDA development root, not compilers/. '
              'Versioned runtime libraries alone do not supply linker development symlinks.'
        )
    return includes, libraries


def prepare(root, state, base, cuda, host):
    source = state / 'source'
    if source.exists():
        raise FileExistsError(f'Use a fresh state directory: {source}')
    includes, libraries = cuda_layout(cuda)
    (state / 'cuda-library-paths.json').write_text(json.dumps(list(map(str, libraries))) + '\n')
    print('CUDA include directories:', ', '.join(map(str, includes)), flush=True)
    print('CUDA library directories:', ', '.join(map(str, libraries)), flush=True)
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
# V100: CUDA 12.x, native Volta code plus forward-compatible PTX.
gpu = True
gpu_target = 'cuda'
gpu_compiler = {str(cuda_compiler(cuda))!r}
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
