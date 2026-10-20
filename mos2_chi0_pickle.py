"""Calculate chi0_wGG for a MoS2 monolayer and store it in a pickle file."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from ase.build import mx2
from ase.units import Ha

from gpaw import FermiDirac, GPAW, PW
from gpaw.mpi import world
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.pair import get_gs_and_context
from gpaw.response.qpd import SingleZQPWDescriptor


def build_ground_state(gpw: Path, args: argparse.Namespace) -> None:
    """Build a reusable MoS2 ground-state file if it does not already exist."""
    if gpw.is_file():
        return

    atoms = mx2('MoS2',
                kind='2H',
                a=args.lattice_constant,
                thickness=args.thickness,
                vacuum=args.vacuum)
    atoms.calc = GPAW(mode=PW(args.gs_ecut),
                      xc='PBE',
                      nbands=args.gs_nbands,
                      convergence={'bands': -args.converged_empty_bands},
                      setups={'Mo': '6'},
                      parallel={'band': 1, 'domain': 1},
                      occupations=FermiDirac(width=args.occupations_width),
                      kpts={'size': (args.kpts, args.kpts, 1),
                            'gamma': True},
                      txt=args.gs_txt)
    atoms.get_potential_energy()
    atoms.calc.write(str(gpw), mode='all')
    world.barrier()


def calculate_chi0_with_ecut(gs, context, wd, ecut, args: argparse.Namespace):
    chi0calc = Chi0Calculator(gs,
                              context,
                              wd=wd,
                              nbands=args.chi0_nbands,
                              ecut=ecut,
                              eta=args.eta,
                              intraband=False,
                              hilbert=False)
    chi0_data = chi0calc.calculate(q_c=args.q_c)

    coulomb = CoulombKernel.from_gs(gs, truncation=args.truncation)
    dyson_eqs = Chi0DysonEquations(chi0_data, coulomb,
                                   xc_kernel=None, cd=gs.cd)

    chi0_wGG = dyson_eqs.get_chi0_wGG(direction=args.direction)
    chi0_WGG = dyson_eqs.wblocks.all_gather(chi0_wGG)
    return chi0_data, chi0_WGG


def map_z_vectors(full_Gv: np.ndarray, z_Gv: np.ndarray) -> np.ndarray:
    indices = []
    for G_v in z_Gv:
        d_G = np.linalg.norm(full_Gv - G_v, axis=1)
        G = np.argmin(d_G)
        if d_G[G] > 1e-12:
            raise RuntimeError(f'Could not find {G_v} in full G-vector basis')
        indices.append(G)
    return np.array(indices)


def calculate_chi0(gpw: Path, args: argparse.Namespace) -> dict:
    """Return a serial dictionary containing gathered chi0_wGG data."""
    gs, context = get_gs_and_context(str(gpw), txt=args.response_txt,
                                     world=world, timer=None)

    frequencies = np.linspace(args.omega_min, args.omega_max, args.nomega)
    wd = get_frequency_descriptor(frequencies, gs=gs, nbands=args.chi0_nbands)

    full_chi0_data, full_chi0_WGG = calculate_chi0_with_ecut(
        gs, context, wd, args.chi0_ecut, args)

    z_ecut = {
        'class': SingleZQPWDescriptor,
        'kwargs': {'ecut_z': args.chi0_ecut / Ha},
    }
    z_chi0_data, z_chi0_WGG = calculate_chi0_with_ecut(
        gs, context, wd, z_ecut, args)

    full_Gv = full_chi0_data.qpd.get_reciprocal_vectors(add_q=False)
    z_Gv = z_chi0_data.qpd.get_reciprocal_vectors(add_q=False)
    z_in_full_G = map_z_vectors(full_Gv, z_Gv)
    full_z_chi0_WGG = full_chi0_WGG[:, z_in_full_G][:, :, z_in_full_G]
    abs_diff_WGG = np.abs(full_z_chi0_WGG - z_chi0_WGG)
    max_abs_diff = np.max(abs_diff_WGG)
    if not np.isfinite(max_abs_diff):
        raise AssertionError(
            'SingleZQPWDescriptor comparison produced non-finite values')
    if max_abs_diff > args.compare_tolerance:
        raise AssertionError(
            f'SingleZQPWDescriptor chi0 differs from the full-basis '
            f'submatrix by {max_abs_diff}; tolerance is '
            f'{args.compare_tolerance}')

    return {
        'chi0_wGG': full_chi0_WGG,
        'z_chi0_wGG': z_chi0_WGG,
        'z_from_full_chi0_wGG': full_z_chi0_WGG,
        'z_in_full_G': z_in_full_G,
        'z_compare_max_abs_diff': max_abs_diff,
        'omega_w': wd.omega_w.copy(),
        'omega_w_eV': wd.omega_w.copy() * Ha,
        'q_c': np.array(args.q_c),
        'direction': args.direction,
        'G_Gv': full_Gv,
        'z_G_Gv': z_Gv,
        'cell_cv': gs.gd.cell_cv.copy(),
        'pbc_c': gs.pbc.copy(),
        'parameters': {
            'gpw': str(gpw),
            'gs_ecut': args.gs_ecut,
            'gs_nbands': args.gs_nbands,
            'kpts': (args.kpts, args.kpts, 1),
            'chi0_ecut': args.chi0_ecut,
            'chi0_nbands': args.chi0_nbands,
            'eta': args.eta,
            'truncation': args.truncation,
            'compare_tolerance': args.compare_tolerance,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpw', type=Path, default=Path('mos2_gs.gpw'))
    parser.add_argument('--output', type=Path, default=Path('mos2_chi0.pkl'))

    parser.add_argument('--lattice-constant', type=float, default=3.184)
    parser.add_argument('--thickness', type=float, default=3.127)
    parser.add_argument('--vacuum', type=float, default=7.5)

    parser.add_argument('--gs-ecut', type=float, default=300.0)
    parser.add_argument('--gs-nbands', type=int, default=24)
    parser.add_argument('--converged-empty-bands', type=int, default=8)
    parser.add_argument('--kpts', type=int, default=6)
    parser.add_argument('--occupations-width', type=float, default=0.001)

    parser.add_argument('--chi0-ecut', type=float, default=40.0)
    parser.add_argument('--chi0-nbands', type=int, default=20)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--omega-min', type=float, default=0.0)
    parser.add_argument('--omega-max', type=float, default=5.0)
    parser.add_argument('--nomega', type=int, default=51)
    parser.add_argument('--q-c', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument('--direction', choices=['x', 'y', 'z'], default='x')
    parser.add_argument('--truncation', choices=['none', '2D'], default='none')
    parser.add_argument('--compare-tolerance', type=float, default=1e-10)

    parser.add_argument('--gs-txt', default='mos2_gs.txt')
    parser.add_argument('--response-txt', default='mos2_chi0.txt')
    args = parser.parse_args()
    if args.truncation == 'none':
        args.truncation = None
    return args


def main() -> None:
    args = parse_args()
    build_ground_state(args.gpw, args)
    data = calculate_chi0(args.gpw, args)

    if world.rank == 0:
        with args.output.open('wb') as fd:
            pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Wrote {args.output} with chi0_wGG shape '
              f'{data["chi0_wGG"].shape} and z_chi0_wGG shape '
              f'{data["z_chi0_wGG"].shape}; max diff '
              f'{data["z_compare_max_abs_diff"]:.3e}')


if __name__ == '__main__':
    main()
