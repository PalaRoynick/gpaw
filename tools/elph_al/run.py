"""Reproducible Al PAW electron-phonon benchmark; see README.md."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.units import Hartree
from ase.utils.filecache import MultiFileJSONCache

from gpaw import FermiDirac, PW
from gpaw.old.calculator import GPAW
from gpaw.elph import ElectronPhononMatrix, Supercell
from gpaw.elph.gmatrix import degenerate_average as reduce_degenerate
from gpaw.mpi import world
from displacements import FreshDisplacementRunner


def groups(values, index, tolerance):
    return np.flatnonzero(abs(values - values[index]) < tolerance)


def make_path(atoms, count):
    labels = 'GXWKGLUX'
    special = atoms.cell.bandpath().special_points
    q, distance, ticks, segments = [], [], [0.], []
    reciprocal = atoms.cell.reciprocal()
    offset = 0.
    for left, right in zip(labels[:-1], labels[1:]):
        first, last = special[left], special[right]
        length = np.linalg.norm((last - first) @ reciprocal)
        for t in np.linspace(0, 1, count):
            q.append(first * (1 - t) + last * t)
            distance.append(offset + length * t)
            segments.append(left + right)
        offset += length
        ticks.append(offset)
    return np.array(q), np.array(distance), np.array(ticks), segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'paper', 'archive'],
                        default='smoke')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fd-cache', type=Path)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--ecut', type=float)
    parser.add_argument('--k-super', type=int)
    parser.add_argument('--k-primitive', type=int)
    parser.add_argument('--h', type=float, default=.20)
    parser.add_argument('--delta', type=float, default=.01)
    parser.add_argument('--basis', default='dzp')
    parser.add_argument('--points', type=int, default=12,
                        help='points per segment, with both endpoint limits')
    parser.add_argument('--gpu-displacements', action='store_true')
    parser.add_argument('--gamma-supercell',
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--stage', choices=['all', 'displacements', 'project',
                                            'bands', 'matrix'], default='all')
    args = parser.parse_args()
    if args.profile == 'archive' and args.fd_cache is None:
        parser.error('archive profile requires --fd-cache')
    repeat = args.repeat or dict(smoke=3, paper=4, archive=3)[args.profile]
    gamma = args.gamma_supercell
    if gamma is None:
        gamma = args.profile != 'smoke'
    ecut = args.ecut or (600 if args.profile == 'paper' else 200)
    ks = args.k_super or dict(smoke=2, paper=3, archive=5)[args.profile]
    kp = args.k_primitive or ks * repeat
    a = 4.079793 if args.profile == 'archive' else 4.04
    atoms = bulk('Al', 'fcc', a=a)
    sc = (repeat,) * 3
    out = args.output.resolve()
    fd = args.fd_cache.resolve() if args.fd_cache else out / 'elph'
    if world.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    world.barrier()
    settings = dict(profile=args.profile, a=a, supercell=sc, ecut=ecut,
                    k_super=ks, k_primitive=kp, h=args.h, delta=args.delta,
                    basis=args.basis, points=args.points,
                    gpu_displacements=args.gpu_displacements, fd_cache=str(fd),
                    gamma_centered_supercell=gamma)
    manifest = out / 'settings.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        previous.setdefault('gamma_centered_supercell', False)
        if previous != json.loads(json.dumps(settings)):
            raise ValueError('Settings changed; use a new output directory.')
    if world.rank == 0:
        manifest.write_text(json.dumps(settings, indent=2) + '\n')
    os.chdir(out)
    common = dict(xc='PBE', occupations=FermiDirac(.0516974),
                  symmetry='off',
                  convergence={'energy': 2e-6, 'density': 1e-6})
    if args.stage in ('all', 'displacements') and args.fd_cache is None:
        engine = GPAW
        extra = {}
        if args.gpu_displacements:
            from gpaw.new.ase_interface import GPAW as engine
            extra['parallel'] = {'gpu': True}
        fd_options = dict(common, symmetry={})
        calc = engine(mode=PW(ecut),
                      kpts={'size': (ks,) * 3, 'gamma': gamma}, h=args.h,
                      txt='displacements.txt', **fd_options, **extra)
        runner = FreshDisplacementRunner(
            atoms, calc, supercell=sc, name=str(fd), delta=args.delta)
        runner.run()
        world.barrier()
    if args.stage in ('all', 'project'):
        info = MultiFileJSONCache(str(fd))['info']
        grid = MultiFileJSONCache(str(fd))['eq']['Vt_sG'].shape[-3:]
        if tuple(info['supercell']) != sc:
            raise ValueError('Cache supercell does not match --repeat')
        calc = GPAW(mode='lcao', basis=args.basis, gpts=grid,
                    kpts={'size': (ks,) * 3, 'gamma': gamma},
                    parallel={'domain': 1, 'band': 1},
                    txt='projection.txt', **common)
        # No SCF needed: use the cached, self-consistent equilibrium D matrix.
        calc.initialize(atoms * sc)
        calc.initialize_positions(atoms * sc)
        Supercell(atoms, supercell=sc).calculate_supercell_matrix(
            calc, fd_name=str(fd))
        world.barrier()
    q, distance, ticks, segments = make_path(atoms, args.points)
    if args.stage in ('all', 'bands'):
        # Symmetry is safe for the self-consistent density. Only the final
        # explicit Bloch path must be evaluated without symmetry reduction.
        ground_options = dict(common, symmetry={})
        saved_grid = np.array(MultiFileJSONCache(str(fd))['eq'][
            'Vt_sG'].shape[-3:])
        if np.all(saved_grid % np.array(sc) == 0):
            grid_options = {'gpts': tuple(saved_grid // np.array(sc))}
        else:
            grid_options = {'h': args.h}  # legacy noncommensurate grid
        calc = GPAW(mode='lcao', basis=args.basis, nbands='nao',
                    kpts={'size': (kp,) * 3, 'gamma': gamma},
                    txt='ground-state.txt',
                    **ground_options, **grid_options)
        atoms.calc = calc
        atoms.get_potential_energy()
        pathcalc = calc.fixed_density(kpts=q, symmetry='off', nbands='nao',
                                      txt='bands.txt')
        pathcalc.write('path.gpw', 'all')
    if args.stage not in ('all', 'matrix'):
        return
    # Projection is k-parallel only. Matrix assembly runs once in serial:
    # the same results are available from the public distributed bloch_matrix.
    if world.size != 1:
        if world.rank == 0:
            print('Run --stage matrix in serial after the MPI stages.')
        return
    calc = GPAW('path.gpw', txt=None)
    # LCAO overlaps are not serialized in GPW files.
    calc.initialize_positions(calc.atoms)
    C = np.array([kpt.C_nM for kpt in calc.wfs.kpt_u])
    energies = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(q))])
    # Locate xy, yz and xz polarization orbitals in the GPAW real harmonics.
    d_indices = []
    offset = 0
    for bf in calc.wfs.setups[0].basis.bf_j:
        if bf.l == 2:
            d_indices.extend(offset + np.array([0, 1, 3]))
        offset += 2 * bf.l + 1
    if not d_indices:
        raise ValueError(
            'The Gamma_25 prime initial triplet requires d orbitals.')
    weights = np.sum(abs(C[0][:, d_indices])**2, axis=1)
    initial = np.sort(weights.argsort()[-3:])
    if np.ptp(energies[0, initial]) > .05:
        raise ValueError(
            'Initial t2g triplet is split by >50 meV; refine grid.')
    elph = ElectronPhononMatrix(atoms, 'supercell', str(fd),
                                load_sc_as_needed=False,
                                use_mean_minimum_images=True)
    omega, modes = elph.phonon.band_structure(q, modes=True)
    final_index = np.zeros(len(q), int)
    mode_index = np.full(len(q), 2, int)
    # Follow the Sigma_1 electron / longitudinal phonon at K backwards along
    # W-K. This implements the exception described in Sec. V.A of the paper.
    wk = np.flatnonzero(np.array(segments) == 'WK')
    previous_e = C[wk[-1], 0]
    target = q[wk[-1]] @ atoms.cell.reciprocal()
    target /= np.linalg.norm(target)
    polarizations = modes[wk[-1], :, 0]
    alignment = abs(polarizations @ target) / np.linalg.norm(
        polarizations, axis=1)
    start_mode = int(alignment.argmax())
    previous_u = modes[wk[-1], start_mode].ravel()
    for i in wk[::-1]:
        S = calc.wfs.S_qMM[i].copy()
        from gpaw.utilities.tools import tri2full
        tri2full(S)
        # Old GPAW stores the transpose of the physical overlap.
        metric = S.T
        if not np.allclose(C[i].conj() @ metric @ C[i].T,
                           np.eye(len(C[i])), atol=1e-7):
            raise ValueError('Bloch coefficients are not normalized in S^T.')
        index = np.argmax(abs(C[i].conj() @ metric @ previous_e))
        final_index[i] = index
        previous_e = C[i, index]
        index = np.argmax(abs(modes[i].reshape(3, -1).conj() @ previous_u))
        mode_index[i] = index
        previous_u = modes[i, index].ravel()
    components = {key: [] for key in ('V', 'D', 'P', 'R', 'total')}
    full = {key: [] for key in components}
    for i, qi in enumerate(q):
        parts = elph.bloch_matrix_pair(C[0], C[i], energies[0] / Hartree,
                                       energies[i] / Hartree, np.zeros(3), qi)
        final = groups(energies[i], final_index[i], .005)
        selected_modes = groups(omega[i], mode_index[i], 1e-5)
        for key in parts:
            full[key].append(parts[key])
            components[key].append(reduce_degenerate(
                parts[key], selected_modes, final, initial))
    np.savez_compressed('matrix-elements.npz', q=q, distance=distance,
                        ticks=ticks, energies=energies, phonon_energies=omega,
                        initial=initial, final_index=final_index,
                        mode_index=mode_index, phonon_modes=modes, **full)
    table = np.column_stack([distance, q, *components.values()])
    np.savetxt('couplings.csv', table, delimiter=',',
               header='distance,q1,q2,q3,V,D,P,R,total', comments='')
    direction = q[wk[-1]] @ atoms.cell.reciprocal()
    direction /= np.linalg.norm(direction)
    uK = modes[wk[-1], -1, 0]
    longitudinal_overlap = float(abs(uK @ direction) / np.linalg.norm(uK))
    summary = dict(initial_bands=initial.tolist(),
                   K_highest_mode_longitudinal_overlap=longitudinal_overlap,
                   phonon_mean_minimum_images=True,
                   initial_energies_eV=energies[0, initial].tolist(),
                   minimum_phonon_energy_eV=float(omega.min()),
                   max_coupling_eV={k: float(np.max(v))
                                    for k, v in components.items()},
                   note='Diagnostic calculation; convergence not established.')
    Path('summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
