"""Add optional PaiNN training to an existing GPAW venv without rebuilding GPAW."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

from update_python_adapters import update


def command(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['v100', 'cpu'], default='v100')
    parser.add_argument('--torch-runtime', choices=['cu118', 'cu126'], default='cu126')
    parser.add_argument('--no-install', action='store_true',
                        help='Configure and check dependencies already present')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prefix = Path(sys.prefix).resolve()
    site = Path(sysconfig.get_path('purelib')).resolve()
    if prefix == Path(sys.base_prefix).resolve() or not site.is_relative_to(prefix):
        parser.error('Run with the destination GPAW venv Python, not system Python')
    if not (site / 'gpaw/__init__.py').is_file():
        parser.error(f'No installed GPAW wheel at {site}')
    suffix = args.torch_runtime if args.profile == 'v100' else 'cpu'
    install = [sys.executable, '-m', 'pip', 'install', f'torch==2.7.1+{suffix}',
               '--index-url', f'https://download.pytorch.org/whl/{suffix}']
    print('Destination Python:', sys.executable, flush=True)
    print('PyTorch install:', ' '.join(install), flush=True)
    print('DeepDFT source:', root / 'third_party/DeepDFT', flush=True)
    if args.dry_run:
        return
    dependency = root / 'third_party/DeepDFT'
    command('git', '-C', root, 'submodule', 'update', '--init', '--recursive', '--', 'third_party/DeepDFT')
    pinned = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD:third_party/DeepDFT'], text=True).strip()
    actual = subprocess.check_output(['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pinned:
        raise RuntimeError('DeepDFT is not at the revision pinned by this checkout')
    status = subprocess.check_output(['git', '-C', str(dependency), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if status:
        raise RuntimeError('DeepDFT submodule has local edits; review them before configuration')
    native = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in site.glob('_gpaw*.so')}
    if not args.no_install:
        command(*install)
    update(root / 'gpaw/density_learning', site, prefix / '.gpaw-adapter-backups')
    config = site / 'gpaw/density_learning/_deepdft_source.json'
    config.write_text(json.dumps({'path': str(dependency), 'revision': actual}, indent=2) + '\n')
    # New process and external cwd avoid stale imports and checkout shadowing.
    program = r'''
import json, sys
sys.dont_write_bytecode = True
import torch
from ase import Atoms
from gpaw.density_learning.painn import PaiNNDensity, ProbeGraph, implementation_fingerprints
profile = sys.argv[1]
if profile == 'v100':
    expected = {'cu118': '11.8', 'cu126': '12.6'}[sys.argv[2]]
    assert torch.version.cuda == expected, torch.version.cuda
    assert torch.cuda.is_available(), 'Allocate a GPU for the V100 readiness check'
    assert torch.cuda.get_device_capability() == (7, 0), 'Expected V100 / Volta'
    assert 'sm_70' in torch.cuda.get_arch_list(), 'PyTorch lacks sm_70'
device = 'cuda' if profile == 'v100' else 'cpu'
atoms = Atoms('Al2', positions=[[2, 2, 2], [2, 2, 4.6]], cell=[8, 8, 8])
model = PaiNNDensity(cutoff=3.0, interactions=2, hidden=16).to(device)
graph = ProbeGraph(atoms, 3.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
points = [[2., 2., 2.5], [2., 2., 3.]]
before = [p.detach().clone() for p in model.parameters()]
loss = model(graph.batch(points, device)).square().mean()
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
print(json.dumps({'torch': torch.__version__, 'device': device,
                  'loss': float(loss.detach().cpu()), 'source': implementation_fingerprints()}))
'''
    command(sys.executable, '-c', program, args.profile, args.torch_runtime, cwd=prefix)
    command(sys.executable, '-m', 'pip', 'check', cwd=prefix)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == checksum
               for p, checksum in native.items()), 'Native GPAW extension changed'
    print('Training check passed; native GPAW extensions preserved.', flush=True)


if __name__ == '__main__':
    main()
