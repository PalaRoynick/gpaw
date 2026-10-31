"""Locate external DeepDFT without putting a GPAW checkout on sys.path."""

import json
import os
from pathlib import Path


def source_directory():
    configured = Path(__file__).with_name('_deepdft_source.json')
    if os.environ.get('GPAW_DEEPDFT_SOURCE'):
        path = Path(os.environ['GPAW_DEEPDFT_SOURCE']).expanduser().resolve()
    elif configured.is_file():
        path = Path(json.loads(configured.read_text())['path'])
    else:
        path = Path(__file__).resolve().parents[2] / 'third_party/DeepDFT'
    path = path.resolve()
    if not all((path / name).is_file() for name in ('densitymodel.py', 'layer.py')):
        raise RuntimeError(
            f'DeepDFT source not found at {path}. Run tools/deepdft/enable_training.py '
            'with the destination environment Python, or set GPAW_DEEPDFT_SOURCE '
            'to the initialized pinned submodule directory.'
        )
    return path
