"""Copy density-learning Python adapters into this interpreter's GPAW wheel.

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


def update(source, site, backup_root, dry_run=False):
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
    changed = []
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
        dest = target / path.name
        if dest.is_symlink():
            raise ValueError(f'Refusing to replace a symlink: {dest}')
        if not dest.exists() or digest(dest) != digest(path):
            changed.append(path)
    native = {str(p): digest(p) for p in site.glob('_gpaw*.so')}
    print(f'Python: {sys.executable}\nSource: {source}\nTarget: {target}', flush=True)
    print('Changed Python files: ' + (', '.join(p.name for p in changed) or 'none'), flush=True)
    if dry_run or not changed:
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')
    backup = Path(tempfile.mkdtemp(prefix=stamp, dir=backup_root))
    manifest = {'source': str(source), 'target': str(target),
                'native_sha256': native, 'created': [], 'replaced': [],
                'source_sha256': {p.name: digest(p) for p in files}}
    # Back up every existing file before writing any replacement.
    for path in changed:
        dest = target / path.name
        if dest.exists():
            shutil.copy2(dest, backup / path.name)
            manifest['replaced'].append(path.name)
        else:
            manifest['created'].append(path.name)
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    target.mkdir(parents=True, exist_ok=True)
    for path in changed:
        handle, temp = tempfile.mkstemp(prefix='.adapter-', dir=target)
        try:
            with os.fdopen(handle, 'wb') as stream:
                stream.write(path.read_bytes())
            os.chmod(temp, 0o644)
            os.replace(temp, target / path.name)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        # Invalidate only this adapter's bytecode, including same-second updates.
        for cache in (target / '__pycache__').glob(path.stem + '.*.pyc'):
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
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    site = Path(sysconfig.get_path('purelib')).resolve()
    if not site.is_relative_to(prefix):
        parser.error('site-packages is outside this interpreter prefix')
    update(args.source, site, prefix / '.gpaw-adapter-backups', args.dry_run)


if __name__ == '__main__':
    main()
