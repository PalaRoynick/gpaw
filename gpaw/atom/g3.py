from contextlib import chdir
from functools import cache
from pathlib import Path

import numpy as np
from gpaw import setup_paths, KohnShamConvergenceError
from gpaw.atom.generator2 import generate
from gpaw.dft import DFT, PW, MonkhorstPack, Parameters
from gpaw.mpi import world, broadcast
from gpaw.utilities.acwf import reference_structure, acwf_structures
from gpaw.atom.tools import eos, ecuts as ecuts_check

PARAMS = {
    # 1-2:
    'H.1': '1s,s,p 0.9',
    'He.2': '1s,s,p 1.5',
    # 3-10:
    'Li.1': '2s,s,2p,D 3.0',
    'Li.3': '1s,2s,2p,p,D 1.5',
    'Be.2': '2s,s,2p 1.9',
    'Be.4': '1s,2s,2p,p,d 1.4',
    'B.3': '2s,s,2p,p,d 1.5',
    'C.4': '2s,s,2p,p,d 1.2',
    'N.5': '2s,5.6s,2p,5.6p,d 1.4,1.4,1.0 0.9',
    'O.6': '2s,s,2p,p,d,F 1.2',
    'F.7': '2s,s,2p,p,d 1.2,1.4',
    'Ne.8': '2s,s,2p,p,d 1.8',
    # 11-18:
    'Na.1': '3s,s,3p 2.6',
    'Na.9': '2s,3s,2p,3p,d,F 2.3',
    'Mg.2': '3s,s,3p,D 2.6',
    'Mg.10': '2s,3s,2p,3p,d,F 2.0,1.8',
    'Al.3': '3s,s,3p,p,d,F 2.1',
    'Si.4': '3s,s,3p,p,d,F 1.9',
    'P.5': '3s,s,3p,p,d,F 1.7',
    'S.6': '3s,s,3p,p,d,F 1.6',
    'Cl.7': '3s,s,3p,p,d,F 1.5',
    'Ar.8': '3s,s,3p,p,d,F 1.5',
    # 19-36:
    'K.1': '4s,s,4p,D 3.9',
    'K.9': '3s,4s,3p,4p,d,d,F 2.1',
    'Ca.2': '4s,s,4p 3.1',
    'Ca.10': '3s,4s,3p,4p,3d,d,F 2.1',
    'Sc.3': '4s,s,4p,p,3d,d,F 3.2',
    'Sc.11': '3s,4s,3p,4p,3d,d,F 2.3',
    'Ti.4': '4s,s,4p,p,3d,d 2.7',
    'Ti.12': '3s,4s,3p,4p,3d,d,F 2.2,2.2,2.3',
    'V.5': '4s,s,4p,p,3d,d 2.6',
    'V.13': '3s,4s,3p,4p,3d,d,F 2.1,2.1,2.3',
    'Cr.6': '4s,s,4p,p,3d,d 2.5',
    'Cr.14': '3s,4s,3p,4p,3d,d,F 2.1,2.1,2.3',
    'Mn.7': '4s,s,4p,p,3d,d 2.4',
    'Mn.15': '3s,4s,3p,4p,3d,d,F 2.0,2.0,2.2',
    'Fe.8': '4s,s,4p,p,3d,d 2.2',
    'Fe.16': '3s,4s,3p,4p,3d,d,F 2.1',
    'Co.9': '4s,s,4p,p,3d,d 2.2',
    'Co.17': '3s,4s,3p,4p,3d,d,F 2.1',
    'Ni.10': '4s,s,4p,p,3d,d 2.1',
    'Ni.18': '3s,4s,3p,4p,3d,d,F 2.0',
    'Cu.11': '4s,s,4p,p,3d,d 2.1',
    'Cu.19': '3s,4s,3p,4p,3d,d,F 1.9',
    'Zn.12': '4s,s,4p,p,3d 2.1',
    'Zn.20': '3s,4s,3p,4p,3d,d,F 1.9',
    'Ga.3': '4s,s,4p,p,d,F 2.2',
    'Ga.13': '4s,s,4p,p,3d,d,F 2.2',
    'Ge.4': '4s,s,4p,p,d,F 2.1',
    'Ge.14': '4s,s,4p,p,3d,d,F 2.1',
    'As.5': '4s,s,4p,p,d,F 2.0',
    'Se.6': '4s,s,4p,p,d,F 2.1',
    'Br.7': '4s,s,4p,p,d,F 2.1',
    'Kr.8': '4s,s,4p,p,d,F 2.1',
    # 37-54:
    'Rb.1': '5s,s,5p,D 3.9',
    'Rb.9': '4s,5s,4p,5p,d,d,F 2.5',
    'Sr.2': '5s,s,5p,D 3.9',
    'Sr.10': '4s,5s,4p,5p,4d,d,F 2.5',
    'Y.3': '5s,s,5p,p,4d,d,F 3.8',
    'Y.11': '4s,5s,4p,5p,4d,d,F 2.5',
    'Zr.4': '5s,s,5p,p,4d,d,F 3.5',
    'Zr.12': '4s,5s,4p,5p,4d,d,F 2.5',
    'Nb.5': '5s,s,5p,p,4d,d,F 3.4',
    'Nb.13': '4s,5s,4p,5p,4d,d,F 2.4,2.4,2.5',
    'Mo.6': '5s,s,5p,p,4d,d,F 3.4',
    'Mo.14': '4s,5s,4p,5p,4d,d,F 2.3',
    'Tc.7': '5s,s,5p,p,4d,d,F 3.3',
    'Tc.15': '4s,5s,4p,5p,4d,d,F 2.3',
    'Ru.8': '5s,s,5p,p,4d,d,F 3.3',
    'Ru.16': '4s,5s,4p,5p,4d,d,F 2.5',
    'Rh.9': '5s,s,5p,p,4d,d,F 3.3',
    'Rh.17': '4s,5s,4p,5p,4d,d,F 2.5',
    'Pd.10': '5s,s,5p,p,4d,d,F 3.3',
    'Pd.18': '4s,5s,4p,5p,4d,d,F 2.5',
    'Ag.11': '5s,s,5p,p,4d,d,F 3.3',
    'Ag.19': '4s,5s,4p,5p,4d,d,F 2.5',
    'Cd.12': '5s,s,5p,p,4d,d,f 3.3',
    'Cd.20': '4s,5s,4p,5p,4d,d,F 2.5',
    'In.13': '5s,s,5p,p,4d,d,F 2.6',
    'Sn.14': '5s,s,5p,p,4d,d,F 2.5',
    'Sb.15': '5s,s,5p,p,4d,d,F 2.5',
    'Te.6': '5s,6s,5p,p,d,d,F 2.5',
    'I.7': '5s,s,5p,p,d,F 2.4',
    'Xe.8': '5s,s,5p,p,d,F 2.3',
    # 55-56:
    'Cs.1': '6s,s,6p,5d,F 4.6',
    'Cs.9': '5s,6s,5p,6p,5d,0.5d,F 3.2',
    'Ba.2': '6s,1.0s,6p,1.0p,5d 4.9 4.1',
    'Ba.10': '5s,6s,5p,6p,5d,d,F 2.2',
    # 57-71:
    'La.11': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Ce.12': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Ce.12b': '5s,6s,5p,6p,5d,d,4f,f,G 2.6',
    'Pr.13': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Nd.14': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Pm.15': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Sm.16': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Eu.17': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Gd.18': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Tb.19': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Dy.20': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Ho.21': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Er.22': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Tm.23': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Yb.24': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Lu.25': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    # 72-86:
    'Hf.4': '6s,s,6p,p,5d,d,F 3.3',
    'Hf.12': '5s,6s,5p,6p,5d,d,F 2.4',
    'Ta.5': '6s,s,6p,p,5d,d,F 3.3',
    'Ta.13': '5s,6s,5p,6p,5d,d,F 2.4',
    'W.6': '6s,s,6p,p,5d,d,F 3.1',
    'W.14': '5s,6s,5p,6p,5d,d,F 2.4',
    'Re.7': '6s,s,6p,p,5d,d,F 3.0',
    'Re.15': '5s,6s,5p,6p,5d,d,F 2.4',
    'Os.8': '6s,s,6p,p,5d,d,F 2.9',
    'Os.16': '5s,6s,5p,6p,5d,d,F 2.4',
    'Ir.9': '6s,s,6p,p,5d,d,F 2.9',
    'Ir.17': '5s,6s,5p,6p,5d,d,F 2.4',
    'Pt.10': '6s,s,6p,p,5d,d,F 2.9',
    'Pt.18': '5s,6s,5p,6p,5d,d,F 2.3',
    'Au.11': '6s,s,6p,p,5d,d,F 2.9',
    'Au.11b': '6s,s,6p,p,5d,d 2.9 2.7',
    'Au.19': '5s,6s,5p,6p,5d,d,F 2.3',
    'Hg.12': '6s,s,6p,p,5d,d 2.5',
    'Hg.20': '5s,6s,5p,6p,5d,d,F 2.3',
    'Tl.13': '6s,s,6p,p,5d,d,F 2.8',
    'Pb.14': '6s,s,6p,p,5d,d,F 2.6',
    'Bi.5': '6s,s,6p,p,d,F 2.8',
    'Bi.15': '6s,s,6p,p,5d,d,F 2.6',
    'Po.6': '6s,s,6p,p,d,F 2.7',
    'At.7': '6s,s,6p,p,d,F 2.6',
    'Rn.8': '6s,s,6p,p,d,F 2.6',
    # 87-88:
    'Fr.1': '7s,s,6p,5d,F 5.0',
    'Fr.9': '6s,7s,6p,7p,6d,d,F 2.7,2.5',
    'Ra.2': '6s,s,6p,5d 4.5',
    'Ra.10': '6s,7s,6p,7p,6d,d,F 2.7,2.5',
    # 89-102:
    'Ac.11': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Th.12': '6s,7s,6p,7p,6d,d,5f,f,G 2.4',
    'Pa.13': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'U.14': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Np.15': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Pu.16': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Am.17': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Cm.18': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Bk.19': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Cf.20': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Es.21': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Fm.22': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'Md.23': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
    'No.24': '6s,7s,6p,7p,6d,d,5f,f,G 2.5'}


def create(symbol, conf):
    proj, rcs, *r0 = conf.split()
    rcs = [float(s) for s in rcs.split(',')]
    if r0:
        r0, = r0
        r0 = float(r0)
    else:
        r0 = 0.9 * min(rcs)
    with open('gen.out', 'w') as fd:
        gen = generate(
            symbol,
            None,
            proj,
            rcs,
            r0,
            None,
            5 if proj[-1].isupper() else 3,
            xc='PBE',
            pseudize=('poly', 5),
            output=fd)
        try:
            ok = gen.check_all()
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            return False
        setup = gen.make_paw_setup()
        setup.write_xml()
    return True


def create_all():
    for name, conf in PARAMS.items():
        # if not conf.split()[0][-1].isupper():
        #     continue
        dir = Path(name)
        symbol = name.split('.')[0]
        if (dir / f'{symbol}.PBE').is_file():
            continue
        dir.mkdir(exist_ok=True)
        with chdir(dir):
            create(symbol, conf)


def analyse(folders):
    from collections import defaultdict

    from ase.data import atomic_numbers, chemical_symbols
    from gpaw.atom.tools import convergence_check
    from gpaw.utilities.acwf import check

    data = defaultdict(list)
    for f in folders:
        y = [f.name]
        try:
            ecuts, efcc, de, ecuts_tot, ecuts_diff = convergence_check(f)
        except FileNotFoundError:
            y += [np.nan, np.nan, np.nan]
        else:
            y += [efcc[0] - efcc[-1], ecuts_tot[2], ecuts_diff[2]]
        maxabs, dea = check(f)
        symbol = f.name.split('.')[0]
        data[symbol].append(y + [maxabs, dea])
    x = []
    y = []
    z = []
    for symbol in chemical_symbols:
        if symbol not in data:
            continue
        best = np.inf
        for name, de, ec, ecd, maxabs, dea in data[symbol]:
            print(f'{name:7} {de:9.6f} {ecd:6.1f} {ec:6.1f} '
                  f'{maxabs:7.4f} {dea:7.4f}')
            Z = atomic_numbers[symbol]
            x.append(Z)
            y.append([de, ecd, ec, maxabs, dea])
            best = min(best, maxabs)
        z.append(best)
    print(sum(z) / len(z), len(z))
    return x, np.array(y)


def oldvsnew():
    x1, y1 = analyse(list(Path('old').glob('*/')))
    x2, y2 = analyse(list(Path('3').glob('*/')))
    import matplotlib.pyplot as plt
    for i in range(5):
        plt.plot(x1, y1[:, i], 'o-')
        plt.plot(x2, y2[:, i], 'x-')
        plt.show()


def optimize(f, x):
    y = f(tuple(x))
    while True:
        D = []
        for i, xi in enumerate(x):
            for d in [-1, 1]:
                if i > 0 and xi + d == 0:
                    continue
                x2 = list(x)
                x2[i] = xi + d
                y2 = f(tuple(x2))
                D.append((y2, x2))
        y2, x2 = min(D)
        if y2 >= y:
            return x, y
        y = y2
        x = x2


def opt(ecut=600):
    setup_paths[:0] = ['.']
    name = Path.cwd().name
    symbol = name.split('.')[0]
    conf = PARAMS[name]
    proj, rcs, *r0 = conf.split()
    rcs = [float(s) for s in rcs.split(',')]
    if r0:
        r0, = r0
        r0 = float(r0)
    else:
        r0 = round(0.9 * min(rcs), 1)

    if 'f' in proj:
        n = 5
    elif 'd' in proj:
        n = 4
    else:
        n = 3

    @cache
    def f(x):
        ok = None
        if world.rank == 0:
            r0x = r0 + x[0] * 0.1
            R = ','.join(f'{r0x + xi * 0.1:.1f}' for xi in x[1:])
            conf = f'{proj} {R} {r0x:.1f}'
            ok = create(symbol, conf)
        ok = broadcast(ok, comm=world)
        if not ok:
            return np.inf
        de, s = pfunc(symbol, ecut)
        y = abs(de / 0.001) + abs(s) * 10000
        if world.rank == 0:
            print(f'OPT: {conf:15} {de:10.6f} {s:10.6f} {y}')
        # return (de / 0.001)**2 + s**2 * 1000000
        return y

    rcs += [rcs[-1]] * (n - 1 - len(rcs))
    x = [0] + [int(round((r - r0) * 10))
               for r in rcs]

    x, y = optimize(f, x)
    if world.rank == 0:
        print(x, y)
    f.cache_clear()
    f(tuple(x))
    for name in acwf_structures:
        eos(name, symbol)
    ecuts_check(symbol)


def pfunc(symbol, ecut=600):
    name = 'FCC'
    kwargs = dict(
        xc='PBE',
        occupations=dict(name='fermi-dirac', width=0.0612),
        convergence=dict(density=1e-6),
        kpts=MonkhorstPack(density=3.0),
        experimental={'backwards_compatible': False})
    atoms = reference_structure(symbol, name)
    dft = DFT(
        atoms,
        mode=PW(ecut),
        **kwargs,
        txt='ecuts.txt',
        converge=False)
    try:
        dft.converge()
    except KohnShamConvergenceError:
        return np.inf, np.inf
    e = dft.calculate_energy()
    params = Parameters(**kwargs, mode=PW(1200))
    dft = dft.new(atoms, params, log=dft.log)
    try:
        dft.converge()
    except (KohnShamConvergenceError, AssertionError):
        return np.inf, np.inf
    e0 = dft.calculate_energy()

    kwargs['kpts'] = MonkhorstPack(density=6.0)
    name = 'Diamond'
    atoms = reference_structure(symbol, name)
    dft = DFT(
        atoms,
        mode=PW(1200),
        **kwargs,
        txt='stress.txt')
    s = dft.calculate_stress()
    return e - e0, s[0]


def workflow():
    from myqueue.workflow import run
    run(function=opt, cores=24, tmax='1d')


if __name__ == '__main__':
    import sys
    if 1:
        oldvsnew()
    elif 0:
        for name in {'Ac.11': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Th.12': '6s,7s,6p,7p,6d,d,5f,f,G 2.4',
                     'Pa.13': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'U.14': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Np.15': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Pu.16': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Am.17': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Cm.18': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Bk.19': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Cf.20': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Es.21': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Fm.22': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'Md.23': '6s,7s,6p,7p,6d,d,5f,f,G 2.5',
                     'No.24': '6s,7s,6p,7p,6d,d,5f,f,G 2.5'}:
            dir = Path(name)
            print(dir, end=' ')
            # dir.mkdir()
    elif len(sys.argv) == 1:
        create_all()
    elif 0:
        create(sys.argv[1], sys.argv[2])
    else:
        analyse((Path(a) for a in sys.argv[1:]))
