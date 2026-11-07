"""Extract plotted vector polylines from the user-supplied Chaput PDF.

This extracts Figure 2 data, not original author numerical files. The fixed
page coordinates are specific to the published PRB PDF, checked visually.
Requires PyMuPDF, separately from the benchmark's runtime dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path

import fitz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdf', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    page = fitz.open(a.pdf)[9]
    drawings = page.get_drawings()
    x0, x1, y0, y1 = 65.357, 278.802, 234.481, 65.791
    rows = []
    for component, first in [('total', 70), ('V', 73), ('P', 79)]:
        for repeat in (2, 3, 4):
            path = drawings[first + repeat - 2]
            if len(path['items']) < 1000:
                raise ValueError(
                    'Unexpected PDF layout: vector path not found')
            for item in path['items']:
                if item[0] != 'l':
                    raise ValueError('Expected Figure 2 polyline')
                point = item[1]
                rows.append((component, repeat, (point.x - x0) / (x1 - x0),
                             (y0 - point.y) / (y0 - y1) * .5))
            point = path['items'][-1][-1]
            rows.append((component, repeat, (point.x - x0) / (x1 - x0),
                         (y0 - point.y) / (y0 - y1) * .5))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w') as f:
        f.write('component,conventional_repeat,path_fraction,coupling_eV\n')
        for key, n, x, y in rows:
            f.write(f'{key},{n},{x:.8f},{y:.8f}\n')
    a.output.with_suffix('.json').write_text(json.dumps(dict(
        doi='10.1103/PhysRevB.100.174304', figure=2, pdf_page=10,
        source_sha256=hashlib.sha256(a.pdf.read_bytes()).hexdigest(),
        method=('PDF vector polyline extraction, line center; '
                'not author raw data'),
        axis_coordinates_pt=[x0, x1, y0, y1],
        nominal_half_linewidth_eV=.0011), indent=2) + '\n')


if __name__ == '__main__':
    main()
