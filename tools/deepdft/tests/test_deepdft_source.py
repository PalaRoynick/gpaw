"""Resolve the pinned source independently of an installed GPAW wheel."""

import json
from pathlib import Path

import pytest

from gpaw.density_learning import dependency


def test_registered_source_and_environment_override(tmp_path, monkeypatch):
    installed = tmp_path / 'site/gpaw/density_learning/dependency.py'
    installed.parent.mkdir(parents=True)
    monkeypatch.setattr(dependency, '__file__', str(installed))
    monkeypatch.delenv('GPAW_DEEPDFT_SOURCE', raising=False)
    sources = [tmp_path / name for name in ('registered', 'override')]
    for source in sources:
        source.mkdir()
        for name in ('densitymodel.py', 'layer.py'):
            (source / name).touch()
    installed.with_name('_deepdft_source.json').write_text(json.dumps({'path': str(sources[0])}))
    assert dependency.source_directory() == sources[0]
    monkeypatch.setenv('GPAW_DEEPDFT_SOURCE', str(sources[1]))
    assert dependency.source_directory() == sources[1]
    (sources[1] / 'layer.py').unlink()
    with pytest.raises(RuntimeError, match='DeepDFT source not found'):
        dependency.source_directory()
