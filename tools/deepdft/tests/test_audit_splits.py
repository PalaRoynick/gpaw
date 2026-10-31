"""Detect legitimately replaced references without accepting a stale split."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def test_audit_changed_reference(tmp_path):
    cube = tmp_path / 'al.cube'
    cube.write_bytes(b'original reference')
    initial = hashlib.sha256(cube.read_bytes()).hexdigest()
    cube.with_suffix('.json').write_text(json.dumps({'cube_sha256': initial}))
    split = tmp_path / 'splits.json'
    split.write_text(json.dumps({'partitions': {'train': [
        {'sample_id': 'al', 'cube': str(cube), 'cube_sha256': initial}]}}))
    script = Path(__file__).resolve().parents[1] / 'audit_splits.py'
    result = subprocess.run([sys.executable, str(script), str(split)], capture_output=True, text=True)
    assert result.returncode == 0
    cube.write_bytes(b'updated reference')
    current = hashlib.sha256(cube.read_bytes()).hexdigest()
    cube.with_suffix('.json').write_text(json.dumps({'cube_sha256': current}))
    result = subprocess.run([sys.executable, str(script), str(split)], capture_output=True, text=True)
    assert result.returncode == 1
    record = json.loads(result.stdout.splitlines()[0])
    assert record['issues'] == ['split_hash_changed']
    assert record['actual_sha256'] == current
    assert record['split_sha256'] == initial
    assert json.loads(split.read_text())['partitions']['train'][0]['cube_sha256'] == initial
