"""Inspect a legacy Al calculation without modifying its cache files."""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.utils.filecache import MultiFileJSONCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    fd = MultiFileJSONCache(str(a.directory / 'elph'))
    sc = MultiFileJSONCache(str(a.directory / 'supercell'))
    info, projection = fd['info'], sc['info']
    eq = fd['eq']
    result = dict(displacements=info, projection=projection,
                  potential_shape=eq['Vt_sG'].shape,
                  atom_count=len(eq['dH_all_asp']),
                  same_supercell=info['supercell'] == projection['supercell'],
                  missing_projection_entries=[
                      str(i) for i in range(3 * info['natom'])
                      if str(i) not in sc],
                  basis=sc['basis'],
                  duplicate_mass_factor_Al=float(1 / np.sqrt(26.981538)))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
