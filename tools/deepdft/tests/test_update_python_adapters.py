"""Adapter updates preserve native code, back up files and invalidate caches."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'update_python_adapters', Path(__file__).resolve().parents[1] / 'update_python_adapters.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_update_and_dry_run(tmp_path):
    source, site, backup = [tmp_path / p for p in ('source', 'site', 'backups')]
    target = site / 'gpaw/density_learning'
    source.mkdir()
    target.mkdir(parents=True)
    (site / 'gpaw/__init__.py').touch()
    native = site / '_gpaw.test.so'
    native.write_bytes(b'unchanged compiled CUDA extension')
    (source / '__init__.py').write_text('')
    (target / '__init__.py').write_text('')
    (source / 'reference.py').write_text('value = 2\n')
    (target / 'reference.py').write_text('value = 1\n')
    (source / 'devices.py').write_text('value = 3\n')
    cache = target / '__pycache__/reference.cpython-312.pyc'
    cache.parent.mkdir()
    cache.write_bytes(b'stale bytecode')
    module.update(source, site, backup, dry_run=True)
    assert not backup.exists()
    assert (target / 'reference.py').read_text() == 'value = 1\n'
    saved = module.update(source, site, backup)
    assert (saved / 'reference.py').read_text() == 'value = 1\n'
    assert (target / 'reference.py').read_text() == 'value = 2\n'
    assert (target / 'devices.py').read_text() == 'value = 3\n'
    assert native.read_bytes() == b'unchanged compiled CUDA extension'
    assert not cache.exists()
    assert json.loads((saved / 'manifest.json').read_text())['created'] == ['devices.py']
    assert module.update(source, site, backup) is None


def test_invalid_source_is_not_partially_installed(tmp_path):
    source, site = tmp_path / 'source', tmp_path / 'site'
    source.mkdir()
    (source / '__init__.py').touch()
    (source / 'broken.py').write_text('def broken(\n')
    (site / 'gpaw').mkdir(parents=True)
    (site / 'gpaw/__init__.py').touch()
    with pytest.raises(SyntaxError):
        module.update(source, site, tmp_path / 'backups')
    assert not (site / 'gpaw/density_learning').exists()



def test_opt_in_fd_core_python_update(tmp_path):
    source, site = tmp_path / 'gpaw/density_learning', tmp_path / 'site'
    source.mkdir(parents=True)
    (source / '__init__.py').touch()
    target = site / 'gpaw'
    target.mkdir(parents=True)
    (target / '__init__.py').touch()
    for name in ('lfc.py', 'new/fd/pot_calc.py'):
        src, dest = source.parent / name, target / name
        src.parent.mkdir(parents=True, exist_ok=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.write_text('value = 2\n')
        dest.write_text('value = 1\n')
    module.update(source, site, tmp_path / 'backup')
    assert (target / 'lfc.py').read_text() == 'value = 1\n'
    saved = module.update(source, site, tmp_path / 'backup', fd_force_fix=True)
    assert (target / 'lfc.py').read_text() == 'value = 2\n'
    assert (saved / 'core/lfc.py').read_text() == 'value = 1\n'
    assert (saved / 'core/new/fd/pot_calc.py').read_text() == 'value = 1\n'
