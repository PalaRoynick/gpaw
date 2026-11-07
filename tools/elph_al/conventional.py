"""Exact conventional-cell Al displacement geometry for the Chaput benchmark.

One atom is displaced in the conventional supercell. Translational equivalence
of the monatomic fcc Bravais lattice recovers the primitive 3-mode phonons.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.build.supercells import lattice_points_in_supercell
from ase.units import _hbar, _amu, _e

from gpaw import PW, FermiDirac
from gpaw.old.calculator import GPAW
from gpaw.mpi import world
from displacements import FreshDisplacementRunner


def geometry(a, repeat):
    primitive = bulk('Al', 'fcc', a=a)
    conventional = bulk('Al', 'fcc', a=a, cubic=True)
    size = (int(repeat),) * 3
    atoms = conventional * size
    matrix = np.rint(atoms.cell.array @ np.linalg.inv(
        primitive.cell.array)).astype(int)
    offset = np.ravel_multi_index(np.array(size) // 2, size) * 4
    R = np.rint((atoms.positions - atoms.positions[offset])
                @ np.linalg.inv(primitive.cell.array)).astype(int)
    return primitive, atoms, matrix, offset, R


def unfolded_kpoints(matrix, nk, gamma=True):
    """Unfold a diagonal supercell k mesh into the primitive BZ."""
    q = np.indices((nk,) * 3).reshape(3, -1).T / nk
    if not gamma:
        q += .5 / nk
    images = lattice_points_in_supercell(matrix.T)
    points = (q @ np.linalg.inv(matrix).T)[:, None] + images[None]
    points = points.reshape(-1, 3) % 1
    points -= np.floor(points + .5)
    return np.unique(np.round(points, 12), axis=0)


class ConventionalPhonons:
    """Primitive monatomic phonons from a conventional-cell displacement."""

    def __init__(self, cache, a, repeat):
        data = geometry(a, repeat)
        self.atoms, self.super_atoms, self.matrix, self.offset, self.R = data
        self.mass = self.atoms.get_masses()[0]
        self.cache = cache
        delta = cache['info']['delta']
        derivatives = [(cache[f'0{v}-']['forces']
                        - cache[f'0{v}+']['forces']) / (2 * delta)
                       for v in 'xyz']
        C = np.array(derivatives).swapaxes(0, 1)
        # Identify the cell containing -R modulo the supercell.
        inverse = np.linalg.inv(self.matrix)
        scaled = self.R @ inverse
        partner = []
        for r in scaled:
            delta_R = scaled + r
            delta_R -= np.rint(delta_R)
            partner.append(np.argmin(np.linalg.norm(delta_R, axis=1)))
        for _ in range(10):
            C = .5 * (C + C[partner].swapaxes(1, 2))
            C[self.offset] -= C.sum(axis=0)
        self.C_Nvv = C
        # All equal shortest images must contribute, including boundary ties.
        cell = self.super_atoms.cell.array
        raw = self.super_atoms.positions - \
            self.super_atoms.positions[self.offset]
        offsets = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        candidates = raw[:, None] + offsets @ cell
        lengths = np.linalg.norm(candidates, axis=2)
        self.images = [vectors[abs(d - d.min()) < 1e-8]
                       @ np.linalg.inv(self.atoms.cell.array)
                       for vectors, d in zip(candidates, lengths)]

    def dynamical_matrix(self, q):
        phase = np.array([np.exp(-2j * np.pi * vectors @ q).mean()
                          for vectors in self.images])
        return np.einsum('n,nij->ij', phase, self.C_Nvv) / self.mass

    def band_structure(self, points, modes=False, verbose=True):
        energies, vectors = [], []
        factor = _hbar * 1e10 / np.sqrt(_e * _amu)
        for q in np.asarray(points):
            D = self.dynamical_matrix(q)
            if not np.allclose(D, D.conj().T, atol=1e-9):
                raise ValueError('Non-Hermitian dynamical matrix')
            values, u = np.linalg.eigh(D)
            energies.append(np.sign(values) * np.sqrt(abs(values)) * factor)
            vectors.append(u.T[:, None] / np.sqrt(self.mass))
        energies = np.array(energies)
        return (energies, np.array(vectors)) if modes else energies


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, choices=(2, 3, 4), default=2)
    p.add_argument('--k-super', type=int, default=6)
    p.add_argument('--ecut', type=float, default=600)
    p.add_argument('--h', type=float, default=.16)
    p.add_argument('--points', type=int, default=24)
    p.add_argument('--delta', type=float, default=.01)
    p.add_argument('--gpu-displacements', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    if world.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    world.barrier()
    a = 4.04
    fd = output / 'elph'
    # Guarantee integer mapping from the cubic grid to the primitive PW grid.
    npc = 4 * int(np.ceil(a / (4 * args.h)))
    grid = (npc * args.repeat,) * 3
    settings = dict(profile='conventional', supercell_kind='conventional',
                    a=a, supercell=[args.repeat] * 3, ecut=args.ecut,
                    k_super=args.k_super, h=args.h, delta=args.delta,
                    points=args.points, fd_cache=str(fd),
                    gamma_centered_supercell=True,
                    number_of_atoms=4 * args.repeat**3,
                    unfolded_kpoints=4 * (args.repeat * args.k_super)**3,
                    gpu_displacements=args.gpu_displacements)
    path = output / 'settings.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Changed settings; use a new output directory.')
    if world.rank == 0:
        path.write_text(json.dumps(settings, indent=2) + '\n')
    engine, extra = GPAW, {}
    if args.gpu_displacements:
        from gpaw.new.ase_interface import GPAW as engine
        extra['parallel'] = {'gpu': True}
    calc = engine(mode=PW(args.ecut), gpts=grid, xc='PBE',
                  kpts={'size': (args.k_super,) * 3, 'gamma': True},
                  occupations=FermiDirac(.0516974),
                  convergence={'energy': 2e-6, 'density': 1e-6},
                  txt=str(output / 'displacements.txt'), **extra)
    conventional = bulk('Al', 'fcc', a=a, cubic=True)
    FreshDisplacementRunner(conventional, calc, select_atoms=[0],
                            supercell=(args.repeat,) * 3, name=str(fd),
                            delta=args.delta).run()
    world.barrier()


if __name__ == '__main__':
    main()
