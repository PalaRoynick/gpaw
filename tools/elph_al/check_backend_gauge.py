"""Compare Old CPU and New CPU/GPU PAW energy gauges and cached fields.

A one-atom Al PW(150 eV), 12^3-grid test at k=(1/4,1/4,1/4).
--gpu requires a real allocated CUDA device and a CUDA-enabled build.
"""
import argparse
import json
from pathlib import Path

from ase.build import bulk
from gpaw import PW, FermiDirac
from gpaw.old.calculator import GPAW as Old
from gpaw.new.ase_interface import GPAW as New
from gpaw.elph import DisplacementRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.gpu:
        import cupy as cp
        assert cp.cuda.runtime.getDeviceCount() >= 1

    records = []
    for name, engine in [('old', Old), ('new', New)]:
        atoms = bulk('Al', 'fcc', a=4.04)
        extra = ({'parallel': {'gpu': True}}
                 if args.gpu and name == 'new' else {})
        calc = engine(mode=PW(150), gpts=(12, 12, 12),
                      kpts=[[.25, .25, .25]], xc='PBE', symmetry='off',
                      occupations=FermiDirac(.1),
                      convergence={'energy': 1e-8, 'density': 1e-8},
                      txt=str(args.output / (name + '.txt')), **extra)
        atoms.calc = calc
        result = DisplacementRunner(
            atoms, calc, name=str(args.output / name))(atoms)
        records.append(dict(energy=atoms.get_potential_energy(),
                            eigenvalues=calc.get_eigenvalues(),
                            V=result['Vt_sG'], D=result['dH_all_asp'][0],
                            forces=result['forces']))

    old, new = records
    out = {
        'new_backend_on_GPU': args.gpu,
        'old_energy_eV': old['energy'],
        'new_energy_eV': new['energy'],
        'max_eigenvalue_difference_eV':
            float(abs(old['eigenvalues'] - new['eigenvalues']).max()),
        'mean_potential_difference_Hartree':
            float((old['V'] - new['V']).mean()),
        'max_potential_difference_Hartree':
            float(abs(old['V'] - new['V']).max()),
        'max_onsite_difference_Hartree':
            float(abs(old['D'] - new['D']).max()),
        'max_force_difference_eV_per_A':
            float(abs(old['forces'] - new['forces']).max())}
    text = json.dumps(out, indent=2) + '\n'
    print(text, end='')
    (args.output / 'backend-check.json').write_text(text)
    assert out['max_eigenvalue_difference_eV'] < 1e-5
    assert out['max_potential_difference_Hartree'] < 1e-6
    assert out['max_onsite_difference_Hartree'] < 1e-6
    assert out['max_force_difference_eV_per_A'] < 1e-6


if __name__ == '__main__':
    main()
