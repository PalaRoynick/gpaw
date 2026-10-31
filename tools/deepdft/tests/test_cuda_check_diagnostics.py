"""Incomplete or single-backend diagnostics must never certify GPU accuracy."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('backend,skip,validated', [
    ('both', False, True), ('both', True, False), ('cuda', False, False),
])
def test_diagnostic_acceptance(tmp_path, monkeypatch, backend, skip, validated):
    spec = importlib.util.spec_from_file_location(
        'check_cuda', Path(__file__).resolve().parents[1] / 'check_gpaw_cuda.py')
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    import gpaw.gpu
    import gpaw.new.c

    monkeypatch.setattr(gpaw.gpu, 'cupy_is_fake', False)
    monkeypatch.setattr(gpaw.new.c, 'GPU_ENABLED', True)
    properties = {'major': 7, 'minor': 0, 'name': b'Test device'}
    fake = SimpleNamespace(__version__='test-only', cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDeviceProperties=lambda _: properties)))
    monkeypatch.setattr(gpaw.gpu, 'cupy', fake)
    monkeypatch.setattr(check, 'calculate', lambda *args: (
        -1.0, None if skip else np.zeros((1, 3)), np.ones((2, 2, 2)), 1.0))
    monkeypatch.setattr(check, 'run_backend', lambda *args: (
        (-1.0, None if skip else np.zeros((1, 3)), np.ones((2, 2, 2)), 1.0),
        {'device': 'Test device', 'capability': [7, 0], 'cupy_version': 'test-only'}))
    output = tmp_path / 'report'
    argv = ['check', '--output', str(output), '--backend', backend]
    if skip:
        argv.append('--skip-forces')
    monkeypatch.setattr(sys, 'argv', argv)
    check.main()
    report = json.loads((output / 'validation.json').read_text())
    assert report['gpu_validated'] is validated
    assert report['force_check_skipped'] is skip



def test_failed_worker_cannot_produce_acceptance(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'check_cuda', Path(__file__).resolve().parents[1] / 'check_gpaw_cuda.py')
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    # Simulate a native worker exit using a tiny standalone subprocess.
    worker = tmp_path / 'failed-worker.py'
    worker.write_text('raise SystemExit(139)\n')
    check.__file__ = str(worker)
    args = SimpleNamespace(mode='fd', family='bulk', skip_forces=False,
                           sync_gpu=True, require_v100=False)
    with pytest.raises(SystemExit, match='Backend cuda failed'):
        check.run_backend(tmp_path, 'cuda', args)
    assert not (tmp_path / 'validation.json').exists()
    failure = json.loads((tmp_path / 'failure.json').read_text())
    assert failure['gpu_validated'] is False
    assert failure['returncode'] == 139
