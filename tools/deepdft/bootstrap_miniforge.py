"""Install pinned official Miniforge into a user-selected directory.

The installer is verified against the official release SHA-256 before it is
executed in batch mode. No sudo or shell initialization is performed.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request


URL = ('https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/'
       'Miniforge3-25.3.0-3-Linux-x86_64.sh')
SHA256 = '1b57f8cb991982063f79b56176881093abb1dc76d73fda32102afde60585b5a1'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    marker = prefix / '.gpaw-deepdft-miniforge'
    if prefix.exists():
        if marker.is_file() and (prefix / 'bin/conda').is_file():
            print(f'Reusing managed Miniforge: {prefix}')
            return
        raise RuntimeError('Miniforge path already exists; choose a new state directory')
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        installer = Path(temporary) / 'miniforge.sh'
        with urllib.request.urlopen(URL, timeout=120) as source:
            with installer.open('wb') as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
        if digest(installer) != SHA256:
            raise RuntimeError('Miniforge checksum mismatch; installer was not executed')
        environment = os.environ.copy()
        environment['CONDA_REGISTER_ENVS'] = 'false'
        subprocess.run(['bash', str(installer), '-b', '-p', str(prefix)],
                       env=environment, check=True)
    marker.touch()
    print(f'Installed verified Miniforge: {prefix}')


if __name__ == '__main__':
    main()
