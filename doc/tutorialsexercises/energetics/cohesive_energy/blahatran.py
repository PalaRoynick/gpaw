"""Collection of bulk systems."""
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.eos import EquationOfState
from ase.formula import Formula
from ase.units import Ha, kJ
from gpaw import GPAW
from gpaw.spherical_harmonics import names


def read_csv(path: Path) -> dict[str, dict[str, float]]:
    header, *lines = path.read_text().splitlines()
    xcs = header.split(',')[2:]
    data = {}
    for line in lines:
        name, x, *values = line.split(',')
        data[name] = {}
        for xc, value in zip(xcs, values):
            data[name][xc] = float(value)
    return data


if 0:
    for line in Path('lattice_constants.csv').read_text().splitlines()[1:]:
        x = line.split(',')
        print(f'   {x[0]!r}: ({x[1]!r}, {x[13]}),')

data = {
    'C': ('diamond', 3.575),
    'Si': ('diamond', 5.471),
    'Ge': ('diamond', 5.764),
    'Sn': ('diamond', 6.657),
    'SiC': ('zincblende', 4.385),
    'BN': ('zincblende', 3.628),
    'BP': ('zincblende', 4.55),
    'AlN': ('zincblende', 4.408),
    'AlP': ('zincblende', 5.513),
    'AlAs': ('zincblende', 5.735),
    'GaN': ('zincblende', 4.551),
    'GaP': ('zincblende', 5.508),
    'GaAs': ('zincblende', 5.749),
    'InP': ('zincblende', 5.961),
    'InAs': ('zincblende', 6.187),
    'InSb': ('zincblende', 6.636),
    'LiH': ('rocksalt', 4.008),
    'LiF': ('rocksalt', 4.07),
    'LiCl': ('rocksalt', 5.152),
    'NaF': ('rocksalt', 4.706),
    'NaCl': ('rocksalt', 5.7),
    'MgO': ('rocksalt', 4.259),
    'Li': ('bcc', 3.437),
    'Na': ('bcc', 4.2),
    'Al': ('fcc', 4.041),
    'K': ('bcc', 5.283),
    'Ca': ('fcc', 5.528),
    'Rb': ('bcc', 5.671),
    'Sr': ('fcc', 6.025),
    'Cs': ('bcc', 6.162),
    'Ba': ('bcc', 5.02),
    'V': ('bcc', 2.998),
    'Ni': ('fcc', 3.518),
    'Cu': ('fcc', 3.632),
    'Nb': ('bcc', 3.312),
    'Mo': ('bcc', 3.162),
    'Rh': ('fcc', 3.832),
    'Pd': ('fcc', 3.943),
    'Ag': ('fcc', 4.148),
    'Ta': ('bcc', 3.32),
    'W': ('bcc', 3.185),
    'Ir': ('fcc', 3.874),
    'Pt': ('fcc', 3.971),
    'Au': ('fcc', 4.161)}


all_atoms = set()
for formula in data:
    all_atoms.update(Formula(formula).count())


def project(calc):
    setup = calc.setups[0]
    labels = [f'{n}{"spd"[l]}({names[l][m]})'
              for n, l in zip(setup.n_j, setup.l_j)
              for m in range(2 * l + 1)]

    lines = ['#,eig. [eV],occ,character,eig. [eV],occ,character']
    for n in range(len(calc.wfs.kpt_qs[0][0].eps_n)):
        line = str(n)
        for spin in [0, 1]:
            kpt = calc.wfs.kpt_qs[0][spin]
            P_i = kpt.P_ani[0][n]
            i = abs(P_i).argmax()
            eig = kpt.eps_n[n] * Ha
            occ = kpt.f_n[n]
            line += f',{eig:.3f},{occ:.1f},{labels[i]}'
        lines.append(line)
    with open(f'{setup.symbol}-atom3.csv', 'w') as fd:
        print('\n'.join(lines), file=fd)


def atom(symbol, L=12.0, d=0.1):
    atoms = Atoms(symbol, cell=[L, L + d, L - d])
    atoms.center()
    atoms.positions[:, 2] += 0.1234
    atoms.calc = GPAW(
        xc='PBE',
        mode={'name': 'pw', 'ecut': 1000.0},
        nbands=-2,
        mixer={'backend': 'no-mixing'},
        occupations={'name': 'fixed-uniform'},
        hund=True,
        eigensolver={'name': 'etdm-fdpw', 'converge_unocc': not True},
        symmetry='off',
        convergence={'density': 1e-3, 'eigenstates': 1e-3},
        txt=f'{symbol}-atom.txt')
    e = atoms.get_potential_energy()
    atoms.calc.write(f'{symbol}-atom.gpw')
    project(atoms.calc)
    return e


def atom3(symbol, L=12.0, d=0.1):
    atoms = Atoms(symbol, cell=[L, L + d, L - d])
    atoms.center()
    atoms.positions[:, 2] += 0.1234
    atoms.calc = GPAW(
        xc='PBE',
        mode={'name': 'pw', 'ecut': 1000.0},
        nbands=-2,
        mixer={'backend': 'no-mixing'},
        occupations={'name': 'fixed-uniform'},
        hund=True,
        eigensolver={'name': 'etdm-fdpw', 'converge_unocc': not True},
        symmetry='off',
        # convergence={'density': 1e-3, 'eigenstates': 1e-3},
        txt=f'{symbol}-atom3.txt')
    e = atoms.get_potential_energy()
    project(atoms.calc)
    return e


def atom2(symbol, L=12.0, d=0.1):
    atoms = Atoms(symbol, cell=[L, L + d, L - d])
    atoms.center()
    atoms.positions[:, 2] += 0.1234
    atoms.calc = GPAW(
        xc='PBE',
        mode={'name': 'pw', 'ecut': 1000.0},
        occupations={'name': 'fermi-dirac', 'width': 0.01},
        hund=True,
        symmetry='off',
        # convergence={'density': 1e-3, 'eigenstates': 1e-3},
        txt=f'{symbol}-atom2.txt')
    e = atoms.get_potential_energy()
    # atoms.calc.write(f'{symbol}-atom.gpw')
    # project(atoms.calc)
    return e


def solid(name):
    x, a_pbe = data[name]
    atoms = bulk(name, crystalstructure=x, a=a_pbe)
    L = 40.0  # Å
    k = int(L / atoms.get_volume()**0.33) + 1
    atoms.calc = GPAW(
        xc='PBE',
        mode={'name': 'pw', 'ecut': 1000.0},
        kpts=(k, k, k),
        occupations={'name': 'marzari-vanderbilt', 'width': 0.1},
        txt=f'{name}-bulk.txt')
    cell = atoms.get_cell()
    volumes = []
    energies = []
    for strain in np.linspace(-0.06, 0.06, 7):
        atoms.set_cell(cell * (1 + strain)**(1 / 3), scale_atoms=True)
        vol = atoms.cell.volume
        e = atoms.get_potential_energy()
        volumes.append(vol)
        energies.append(e)

    eos = EquationOfState(volumes, energies, 'birchmurnaghan')
    v0, e0, B = eos.fit()
    return {'a': a_pbe * (v0 / volumes[3])**(1 / 3),
            'e0': e0,
            'symbols': list(atoms.symbols),
            'B': B,
            'a_pbe': a_pbe}


def workflow():
    from myqueue.workflow import run
    for symbol in all_atoms:
        run(function=atom, args=[symbol], name=f'{symbol}-atom')
        run(function=atom2, args=[symbol], name=f'{symbol}-atom2')
        run(function=atom3, args=[symbol], name=f'{symbol}-atom3')
    for name in data:
        run(function=solid, args=[name], name=f'{name}-bulk',
            cores=24, tmax='50m')


def check_occ():
    for symbol in all_atoms:
        calc = GPAW(f'{symbol}-atom.gpw')
        f_sn = [[calc.get_occupation_numbers(spin=s) for s in [0, 1]]]
        e_sn = [[calc.get_eigenvalues(spin=s) for s in [0, 1]]]
        f_m = np.array(f_sn).reshape((-1,))
        e_m = np.array(e_sn).reshape((-1,))
        f = f_m[np.argsort(e_m)]
        if not (f[1:] <= f[:-1]).all():
            print(symbol)
            print(e_sn)
            print(f_sn)


def summary():
    reference_energies = {}
    for path in Path().glob('*-atom.result'):
        e = json.loads(path.read_text())
        reference_energies[path.name.split('-')[0]] = e
    LC = read_csv(Path('lattice_constants.csv'))
    CE = read_csv(Path('cohesive_energies.csv'))
    BM = read_csv(Path('bulk_moduli.csv'))
    print('name,',
          'aG [Ang], aG-aW [%], aE-aW [%],',
          'eG [eV], eG-eW [%], eE-eW [%],',
          'BG [eV/Ang^3], BG-BW [%]')
    for path in Path().glob('*-bulk.result'):
        result = json.loads(path.read_text())
        name = path.name.split('-')[0]
        ec = (sum(reference_energies[symbol]
                  for symbol in result['symbols']) -
              result['e0']) / len(result['symbols'])
        ec_pbe = CE[name]['PBE']
        ec_exp = CE[name]['Expt.']
        dec = 100 * (ec / ec_pbe - 1)
        dec_exp = 100 * (ec_exp / ec_pbe - 1)
        a = result['a']
        a_pbe = LC[name]['PBE']
        a_exp = LC[name]['Expt.']
        da = 100 * (a / a_pbe - 1)
        da_exp = 100 * (a_exp / a_pbe - 1)
        B = result['B']
        B_pbe = BM[name]['PBE'] * kJ * 1e-24
        dB = 100 * (B / B_pbe - 1)
        print(
            f'{name:4},',
            f'{a:.3f}, {da:6.2f}, {da_exp:6.2f},',
            f'{ec:.3f}, {dec:6.2f}, {dec_exp:6.2f},',
            f'{B:.3f}, {dB:6.2f}')


if __name__ == '__main__':
    #summary()
    check_occ()
