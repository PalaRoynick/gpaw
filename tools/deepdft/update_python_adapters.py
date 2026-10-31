"""Copy Python adapters and optional FD force fixes into this GPAW wheel.

Run with the destination environment's Python. No GPAW imports, package installs,
CUDA compilation, or native-extension replacement are performed.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update(source, site, backup_root, dry_run=False, fd_force_fix=False):
    source, site = source.resolve(), site.resolve()
    target = site / 'gpaw/density_learning'
    if not (site / 'gpaw/__init__.py').is_file():
        raise ValueError(f'No installed GPAW package at {site}; select its venv Python')
    if source == target.resolve():
        raise ValueError('Source and installed adapter directory must be different')
    if not target.resolve().is_relative_to(site):
        raise ValueError('Installed adapter directory points outside site-packages')
    files = sorted(source.glob('*.py'))
    if not files or not (source / '__init__.py').is_file():
        raise ValueError(f'No adapter package at {source}')
    jobs = [(p, target / p.name, p.name) for p in files]
    if fd_force_fix:
        for name in ('lfc.py', 'new/fd/pot_calc.py'):
            jobs.append((source.parent / name, site / 'gpaw' / name, 'core/' + name))
    changed = []
    for path, dest, key in jobs:
        ast.parse(path.read_text(), filename=str(path))
        if dest.is_symlink() or not dest.resolve().is_relative_to(site):
            raise ValueError(f'Refusing an external destination or symlink: {dest}')
        if not dest.exists() or digest(dest) != digest(path):
            changed.append((path, dest, key))
    native = {str(p): digest(p) for p in site.glob('_gpaw*.so')}
    print(f'Python: {sys.executable}\nSource: {source}\nTarget: {target}', flush=True)
    print('Changed Python files: ' + (', '.join(key for _, _, key in changed) or 'none'), flush=True)
    if dry_run or not changed:
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')
    backup = Path(tempfile.mkdtemp(prefix=stamp, dir=backup_root))
    manifest = {'source': str(source), 'target': str(target),
                'native_sha256': native, 'created': [], 'replaced': [],
                'source_sha256': {key: digest(p) for p, _, key in jobs}}
    # Back up every existing file before writing any replacement.
    for path, dest, key in changed:
        if dest.exists():
            saved = backup / key
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, saved)
            manifest['replaced'].append(key)
        else:
            manifest['created'].append(key)
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    target.mkdir(parents=True, exist_ok=True)
    for path, dest, key in changed:
        dest.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(prefix='.adapter-', dir=dest.parent)
        try:
            with os.fdopen(handle, 'wb') as stream:
                stream.write(path.read_bytes())
            os.chmod(temp, 0o644)
            os.replace(temp, dest)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        # Invalidate only this adapter's bytecode, including same-second updates.
        for cache in (dest.parent / '__pycache__').glob(path.stem + '.*.pyc'):
            cache.unlink()
    assert all(digest(Path(p)) == checksum for p, checksum in native.items()), \
        'Native extension changed unexpectedly'
    print(f'Updated {len(changed)} Python files. Backup: {backup}', flush=True)
    print('Native extensions preserved. Start a new Python process for the updated code.', flush=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'gpaw/density_learning',
                        help='Source adapter directory in the updated checkout')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-fd-force-fix', action='store_true',
                        help='Also update the two whitelisted Python core files for host-staged FD forces')
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    site = Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(prefix):
        parser.error('site-packages is outside this interpreter prefix')
    update(args.source, site, prefix / '.gpaw-adapter-backups', args.dry_run, args.include_fd_force_fix)


if __name__ == '__main__':
    main()
