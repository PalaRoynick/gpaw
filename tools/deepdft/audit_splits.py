"""Read-only verification of split snapshots against cube files and sidecars."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(path):
    split = json.loads(path.read_text())
    failures = 0
    directories = set()
    for partition, records in split['partitions'].items():
        for record in records:
            cube = Path(record['cube']).expanduser().resolve()
            directories.add(str(cube.parent))
            result = {'partition': partition, 'sample_id': record.get('sample_id'),
                      'cube': str(cube), 'split_sha256': record['cube_sha256'], 'issues': []}
            try:
                metadata = json.loads(cube.with_suffix('.json').read_text())
                with cube.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                result.update(actual_sha256=actual, sidecar_sha256=metadata['cube_sha256'])
                if actual != metadata['cube_sha256']:
                    result['issues'].append('cube_sidecar_hash_mismatch')
                if actual != record['cube_sha256']:
                    result['issues'].append('split_hash_changed')
            except (OSError, ValueError, KeyError) as error:
                result['issues'].append(str(error))
            failures += bool(result['issues'])
            print(json.dumps(result), flush=True)
    print(json.dumps({'failed_records': failures, 'reference_directories': sorted(directories)}))
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('splits', type=Path)
    raise SystemExit(audit(parser.parse_args().splits))
