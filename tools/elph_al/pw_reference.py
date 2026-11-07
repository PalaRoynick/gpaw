"""Serial direct-PW Al check of Eqs. (75)-(81), without an LCAO projection.

Supports diagonal primitive and the conventional-cubic Al supercells.
The cached grid must be commensurate with the primitive PW grid.
The finite-displacement cache is read only. This is an independent diagnostic,
not a general replacement for GPAW's localized-orbital interpolation engine.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.phonons import Phonons
from ase.units import Hartree, _amu, _me
from ase.utils.filecache import MultiFileJSONCache

from gpaw import PW, FermiDirac
from gpaw.old.calculator import GPAW
from gpaw.elph import Supercell, ElectronPhononMatrix
from gpaw.elph.gmatrix import degenerate_average
from gpaw.utilities import unpack_hermitian
from gpaw.mpi import world

from run import make_path, groups
from conventional import geometry, unfolded_kpoints, ConventionalPhonons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calculation', type=Path,
                        help='run.py calculation containing settings.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nbands', type=int, default=16)
    parser.add_argument('--bands-only', action='store_true')
    args = parser.parse_args()
    if world.size != 1:
        parser.error('Direct-PW diagnostic currently requires one MPI rank.')
    settings = json.loads((args.calculation / 'settings.json').read_text())
    fd = Path(settings['fd_cache'])
    cache = MultiFileJSONCache(str(fd))
    info = cache['info']
    repeat = np.array(info['supercell'])
    conventional = settings.get('supercell_kind') == 'conventional'
    if info['natom'] != (4 if conventional else 1):
        parser.error('Cache atom count does not match the declared geometry.')
    grid = np.array(cache['eq']['Vt_sG'].shape[-3:])
    if np.any(grid % repeat):
        parser.error('Cached grid must be divisible by the supercell repeats.')
    primitive_grid = grid // repeat
    atoms = bulk('Al', 'fcc', a=settings['a'])
    if conventional:
        atoms, super_atoms, matrix, offset, R = geometry(
            settings['a'], int(repeat[0]))
        k_sampling = unfolded_kpoints(
            matrix, settings['k_super'],
            settings.get('gamma_centered_supercell', True))
    else:
        super_atoms = atoms * tuple(repeat)
        matrix = np.diag(repeat)
        center = repeat // 2
        offset = np.ravel_multi_index(center, repeat)
        R = np.indices(repeat).reshape(3, -1).T - center
        k_sampling = dict(size=(settings['k_super'] * repeat).tolist(),
                          gamma=settings.get(
                              'gamma_centered_supercell', False))

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    q, distance, ticks, segments = make_path(atoms, settings['points'])
    if not (out / 'path.gpw').exists():
        calc = GPAW(mode=PW(settings['ecut']), nbands=args.nbands,
                    gpts=primitive_grid, xc='PBE',
                    kpts=k_sampling,
                    occupations=FermiDirac(.0516974),
                    convergence={'energy': 2e-6, 'density': 1e-6},
                    txt=str(out / 'ground-state.txt'))
        atoms.calc = calc
        atoms.get_potential_energy()
        calc = calc.fixed_density(kpts=q, symmetry='off', nbands=args.nbands,
                                  convergence={'bands': 'occupied'},
                                  txt=str(out / 'bands.txt'))
        # Dense diagonalization avoids convergence of a partially included
        # highest degenerate manifold in the iterative solver.
        calc.diagonalize_full_hamiltonian(nbands=args.nbands)
        calc.write(str(out / 'path.gpw'), 'all')
    if args.bands_only:
        return
    calc = GPAW(str(out / 'path.gpw'), txt=None)
    calc.initialize_positions(calc.atoms)
    wfs = calc.wfs
    energies = np.array([calc.get_eigenvalues(kpt=i) for i in range(len(q))])
    nb = calc.get_number_of_bands()
    P = np.array([kpt.P_ani[0] for kpt in wfs.kpt_u])
    dP = []
    for kpt in wfs.kpt_u:
        # PWLFC.derivative differentiates in r, hence its negative is d/dR.
        dP.append(-wfs.pt.derivative(kpt.psit_nG[:], q=kpt.q)[0])
    dP = np.array(dP)
    angular = []
    for l in wfs.setups[0].l_j:
        angular.extend([(l, m) for m in range(2 * l + 1)])
    t2g = [i for i, (l, m) in enumerate(angular) if l == 2 and m in (0, 1, 3)]
    weight = np.sum(abs(P[0, :, t2g])**2, axis=0)
    candidates = np.flatnonzero((energies[0] > calc.get_fermi_level())
                                & (energies[0] < calc.get_fermi_level() + 20))
    initial = np.sort(candidates[np.argsort(weight[candidates])[-3:]])
    if len(initial) != 3 or np.ptp(energies[0, initial]) > .01:
        raise ValueError('Could not identify the lowest Gamma t2g triplet.')
    if conventional:
        phonon = ConventionalPhonons(cache, settings['a'], int(repeat[0]))
    else:
        phonon = Phonons(atoms, supercell=tuple(repeat), name=str(fd),
                         delta=info['delta'], center_refcell=True,
                         use_mean_minimum_images=True)
        phonon.read(method='standard', symmetrize=10)
    omega, modes = phonon.band_structure(q, modes=True)
    final_index = np.zeros(len(q), dtype=int)
    mode_index = np.full(len(q), 2, dtype=int)

    def periodic(i, bands):
        return np.array([calc.get_pseudo_wave_function(
            band=int(n), kpt=i, periodic=True) for n in bands])

    wk = np.flatnonzero(np.array(segments) == 'WK')
    previous = periodic(wk[-1], [0])[0]
    target = q[wk[-1]] @ atoms.cell.reciprocal()
    target /= np.linalg.norm(target)
    polarizations = modes[wk[-1], :, 0]
    alignment = abs(polarizations @ target) / np.linalg.norm(
        polarizations, axis=1)
    start_mode = int(alignment.argmax())
    previous_mode = modes[wk[-1], start_mode].ravel()
    for i in wk[::-1]:
        waves = periodic(i, range(nb))
        overlap = np.einsum('nxyz,xyz->n', waves.conj(), previous)
        final_index[i] = abs(overlap).argmax()
        previous = waves[final_index[i]]
        overlap = modes[i].reshape(3, -1).conj() @ previous_mode
        mode_index[i] = abs(overlap).argmax()
        previous_mode = modes[i, mode_index[i]].ravel()
    vgradient, dH = Supercell.calculate_gradient(str(fd), indices=[0])
    D = unpack_hermitian(cache['eq']['dH_all_asp'][offset][0])
    Q = wfs.setups[0].dO_ii
    Rii = -wfs.setups[0].nabla_iiv
    fractions = np.indices(grid) / grid[:, None, None, None]
    coords = np.einsum('axyz,ab->bxyz', fractions, matrix)
    center = super_atoms.positions[offset] @ np.linalg.inv(atoms.cell.array)
    coords -= center[:, None, None, None]
    mapped = coords * primitive_grid[:, None, None, None]
    if not np.allclose(mapped, np.rint(mapped), atol=1e-8):
        raise ValueError('PW and displacement grids are not commensurate.')
    mapping = np.rint(mapped).astype(int) % primitive_grid[:, None, None, None]

    def on_supercell(i, bands):
        u = periodic(i, bands)
        return u[:, mapping[0], mapping[1], mapping[2]]

    psi_i = on_supercell(0, initial)
    dv = abs(np.linalg.det(super_atoms.cell.array)) / np.prod(grid)
    parts_all = {key: [] for key in ('V', 'D', 'P', 'R', 'total')}
    curves = {key: [] for key in parts_all}
    Pi = P[0, initial]
    dPi = dP[0, initial]
    for i, qi in enumerate(q):
        final = groups(energies[i], final_index[i], .005)
        selected_modes = groups(omega[i], mode_index[i], 1e-5)
        psi_f = on_supercell(i, final)
        psi_f *= np.exp(2j * np.pi * np.einsum('v,vxyz->xyz', qi, coords))
        Pf, dPf = P[i, final], dP[i, final]
        shape = (3, len(final), len(initial))
        kernels = {key: np.zeros(shape, complex)
                   for key in ('V', 'D', 'P', 'Q', 'Qright', 'R')}
        for v in range(3):
            kernels['V'][v] = dv * np.einsum(
                'mxyz,xyz,nxyz->mn', psi_f.conj(), vgradient[v, 0], psi_i,
                optimize=True)
            for a, dr in enumerate(R):
                delta_D = unpack_hermitian(dH[v][a][0])
                kernels['D'][v] += (np.exp(-2j * np.pi * qi @ dr)
                                    * (Pf.conj() @ delta_D @ Pi.T))
            kernels['P'][v] = (dPf[:, :, v].conj() @ D @ Pi.T
                               + Pf.conj() @ D @ dPi[:, :, v].T)
            kernels['Q'][v] = dPf[:, :, v].conj() @ Q @ Pi.T
            kernels['Qright'][v] = Pf.conj() @ Q @ dPi[:, :, v].T
            kernels['R'][v] = Pf.conj() @ Rii[:, :, v] @ Pi.T
        kernels = {key: np.einsum('lv,vmn->lmn', modes[i, :, 0], value)
                   for key, value in kernels.items()}
        parts = ElectronPhononMatrix._assemble_paw(
            kernels, energies[0, initial] / Hartree,
            energies[i, final] / Hartree)
        if np.allclose(qi, 0):
            parts = {key: value * 0 for key, value in parts.items()}
        else:
            if np.any(omega[i] <= 0):
                raise ValueError(
                    'Unstable phonon; direct-PW diagnostic stopped.')
            factor = Hartree / np.sqrt(2 * _amu / _me / Hartree * omega[i])
            parts = {key: value * factor[:, None, None]
                     for key, value in parts.items()}
        for key, value in parts.items():
            curves[key].append(degenerate_average(
                value, selected_modes, np.arange(len(final)),
                np.arange(len(initial))))
            full = np.zeros((3, nb, nb), complex)
            full[np.ix_(np.arange(3), final, initial)] = value
            parts_all[key].append(full)
    np.savez_compressed(out / 'matrix-elements.npz', q=q, distance=distance,
                        ticks=ticks, energies=energies, phonon_energies=omega,
                        initial=initial, final_index=final_index,
                        mode_index=mode_index, phonon_modes=modes, **parts_all)
    np.savetxt(out / 'couplings.csv',
               np.column_stack([distance, q, *curves.values()]), delimiter=',',
               header='distance,q1,q2,q3,V,D,P,R,total', comments='')
    summary = dict(method='Direct PW; no LCAO projection',
                   phonon_mean_minimum_images=True,
                   K_highest_mode_longitudinal_overlap=float(alignment[-1]),
                   initial_bands=initial.tolist(),
                   initial_energies_eV=energies[0, initial].tolist(),
                   max_coupling_eV={key: float(np.max(value))
                                    for key, value in curves.items()},
                   setup_fingerprint=wfs.setups[0].fingerprint,
                   settings=settings,
                   note='Diagnostic settings; convergence not established.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
