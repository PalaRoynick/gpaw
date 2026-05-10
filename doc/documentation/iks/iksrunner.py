#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk, molecule
from ase.io import read
from ase.parallel import parprint
from gpaw import GPAW
from gpaw.grid_descriptor import GridDescriptor
from gpaw.iks.iks import iKS
from gpaw.mpi import world
from gpaw.typing import Array2D, ArrayND
from gpaw.xc.numericalxc import FermiAmaldi, NumericalXC

path_t = Union[str, Path]


def save(
    save_path: Path, name: path_t, array: ArrayND, gd: Optional[GridDescriptor] = None
) -> None:
    if gd is None:
        np.save(save_path / name, array)
    else:
        array = gd.collect(array)
        if gd.comm.rank == 0:
            np.save(save_path / name, array)


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("xc", type=str)
    parser.add_argument("mol", type=str)
    parser.add_argument("--method", type=str, default="default")
    parser.add_argument("--maxls", type=int, default=5)

    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nbands", type=int, default=None)
    parser.add_argument("--nkpts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--alphapseudo", type=float, default=0.001)
    parser.add_argument("--alphapaw", type=float, default=0.001)
    parser.add_argument("--vacuum", type=float, default=4.0)
    parser.add_argument("--mode", type=str, default="pw")
    parser.add_argument("--ecut", type=float, default=500.0)
    parser.add_argument("--gridspacing", type=float, default=0.15)
    parser.add_argument("--weights", type=str, default="inverse_density")
    parser.add_argument("--atomic-weights", type=str,
                        default="radial_inverse_density")
    parser.add_argument("--paw-potmode", type=str, default="lebedev")
    parser.add_argument("--ls", type=str, default="minres")
    parser.add_argument("--thr", type=float, default=-1.0)
    parser.add_argument("--stencil", type=int, default=3)
    parser.add_argument("--hess", type=str, default=None)
    parser.add_argument("--nfixed", type=int, default=1)
    parser.add_argument("--nsave", type=int, default=100)
    parser.add_argument("--eigsolver-method", type=str, default="dav")
    parser.add_argument("--eigsolver-niter", type=int, default=5)
    parser.add_argument("--interpolate-vt", action="store_true")
    parser.add_argument("--kin-band", action="store_true")
    parser.add_argument("--virialxc", action="store_true")
    parser.add_argument("--ci", type=str, default=None)
    parser.add_argument("--par", type=str, default="domain")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-fascf", action="store_true")
    parser.add_argument("--fixmagmom", action="store_true")
    parser.add_argument("--fwidth", type=float, default=0.0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--hsetups", type=str, default="paw")
    parser.add_argument(
        "-L", "--lnorm", type=int, default=2, help="Norm of the Lagrangian (default: 2)"
    )
    parser.add_argument("--scipy-scale", type=float, default=1.0)
    return parser.parse_args()


def parallel_arg(par):
    if par == "bd":
        parallel = {"domain": max(world.size // 2, 1),
                    "band": max(world.size // 2, 1)}
    elif par == "bk":
        parallel = {"kpt": max(world.size // 2, 1),
                    "band": max(world.size // 2, 1)}
    elif par == "kd":
        parallel = {"kpt": max(world.size // 2, 1),
                    "domain": max(world.size // 2, 1)}
    elif par == "bdk":
        parallel = {
            "domain": max(world.size // 3, 1),
            "kpt": max(world.size // 3, 1),
            "band": max(world.size // 3, 1),
        }
    elif par == "default":
        parallel = {}
    else:
        parallel = {par: world.size}
    return parallel


def get_atoms(mol, periodic, vacuum):
    if mol.endswith(".xyz"):
        atoms = read(mol)
    elif periodic:
        atoms = bulk(mol)
    else:
        try:
            atoms = molecule(mol)
        except KeyError:
            atoms = Atoms(mol)

    if not periodic:
        atoms.center(vacuum=vacuum)
    return atoms


def get_density_matrix(ci: Optional[path_t], spinpaired=True) -> Array2D:
    if ci is None:
        return None
    P_nn = np.load(ci)
    P_alpha = P_nn[::2, ::2]
    P_beta = P_nn[1::2, 1::2]
    if spinpaired:
        P_snn = (P_alpha + P_beta)[None]
    else:
        P_snn = np.concatenate((P_alpha[None], P_beta[None]), axis=0)
    return P_snn


hess_methods = [
    "trust-constr",
    "Newton-CG",
    "dogleg",
    # too much memory methods:
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
    "SLSQP",
]
no_hess_methods = [
    "L-BFGS-B",
    "CG",
    # too much memory methods
    "BFGS",
    "TNC",
]


def main():
    args = argparser()
    save_path = Path(args.outdir or f"{args.mol}_{args.xc}")
    save_path.mkdir(parents=True, exist_ok=True)
    parprint(f"Outputs are written to {save_path}")
    # calculator
    if not args.restart:
        mode = {"name": args.mode}
        if args.mode == "pw":
            mode["ecut"] = args.ecut
        calc_kwargs = dict(
            mode=mode,
            h=args.gridspacing,
            nbands=args.nbands,
            symmetry="off",
            kpts=(args.nkpts,) * 3,
            eigensolver={"name": args.eigsolver_method,
                         "niter": args.eigsolver_niter},
            parallel=parallel_arg(args.par),
            setups={"H": args.hsetups},
        )
        if args.fixmagmom:
            calc_kwargs["occupations"] = {
                "name": "fermi-dirac",
                "width": 0.0,
                "fixmagmom": args.fixmagmom,
            }
        # target
        target = GPAW(
            **calc_kwargs, xc=args.xc, txt=save_path / f"{args.mol}_{args.xc}.txt"
        )
        atoms = get_atoms(args.mol, args.periodic, args.vacuum)
        atoms.calc = target
        atoms.get_potential_energy()
        target.write(save_path / "target.gpw")
        # FA
        fa = FermiAmaldi(
            target.density,
            target.hamiltonian.poisson,
            setup_name=target.hamiltonian.xc.get_setup_name(),
            paw_potmode=args.paw_potmode,
            virial_xc_energy=args.virialxc,
            stencil=args.stencil,
            interpolate_vt=args.interpolate_vt,
        )
        if args.fixmagmom or args.fwidth > 0.0:
            calc_kwargs["occupations"] = {
                "name": "fermi-dirac",
                "width": args.fwidth,
                "fixmagmom": args.fixmagmom,
            }
        calc = GPAW(
            **calc_kwargs,
            convergence={"bands": "all"},
            xc=fa,
            txt=save_path / f"{args.mol}_iks.txt",
        )
        calc.atoms = atoms
        if not args.no_fascf:
            atoms.calc = calc
            atoms.get_potential_energy()
            calc.write(save_path / "fermi-amaldi-scf.gpw", mode="all")
    else:
        target = GPAW(
            save_path / "target.gpw",
            txt=save_path / f"{args.mol}_{args.xc}.txt",
        )
        calc = GPAW(save_path / "iks.gpw",
                    txt=save_path / f"{args.mol}_iks.txt")
        if args.nbands != calc.parameters["nbands"]:
            xc = calc.hamiltonian.xc
            calc = calc.new(
                xc=xc, nbands=args.nbands, txt=save_path /
                f"{args.mol}_iks.txt"
            )

    # iKS
    iks = iKS(
        target,
        calc,
        P_snn=get_density_matrix(args.ci),
        weights=args.weights,
        atomic_weights=args.atomic_weights,
        mask_threshold=args.thr,
        paw_potmode=args.paw_potmode,
        linear_solver=args.ls,
        stencil=args.stencil,
        nfixed=args.nfixed,
        kin_en_using_band=args.kin_band,
        use_coarse_grid=args.coarse,
        log_l1_ae=(args.hsetups != "ae"),
        norm=args.lnorm,
    )
    iks.initialize()
    save(save_path, "target_pseudo_density", iks.tdens.nt_sg, iks.tdens.finegd)
    for a, D_sp in iks.tdens.D_asp.items():
        save(save_path, f"target_D_{a}sp", D_sp)
    # warmup
    if not args.no_warmup:
        iks.warmup()
        if not args.restart:
            calc.write(save_path / "fermi-amaldi-warmup.gpw", mode="all")

    # checkpoint every `nsave` iterations
    def write_calc():
        calc.write(save_path / "iks.gpw", mode="all")
        if iks.gd.comm.rank == 0:
            save(save_path, "L_iter", iks.L_iter)
            save(save_path, "LD_iter", iks.LD_iter)

    calc.attach(write_calc, args.nsave)
    # choose optimizer and run
    if args.method == "default":
        iks.run(max_ls=args.maxls)
    elif args.method == "mix":
        iks.run_mix(args.alphapaw, method="L-BFGS-B")
    elif args.method == "fastest-descent":
        iks.run_fastest_descent(
            maxiter=args.maxiter,
            alpha_pseudo=args.alphapseudo,
            alpha_paw=args.alphapaw,
        )
    elif args.method.lower() == "adam":
        iks.run_adam(
            maxiter=args.maxiter,
            eta_paw=args.alphapaw,
            eta_pseudo=args.alphapseudo,
        )
    elif args.method in no_hess_methods:
        iks.run_scipy(
            args.method,
            scale=args.scipy_scale,
            options={"maxiter": args.maxiter, "gtol": 1e-24},
        )
    elif args.method in hess_methods:
        iks.run_scipy(
            args.method,
            scale=args.scipy_scale,
            options={"maxiter": args.maxiter, "gtol": 1e-24, "xtol": 1e-40},
            hess=args.hess,
        )
    elif args.method == "trust-identity":
        iks.run_scipy(
            method="trust-constr",
            scale=args.scipy_scale,
            options={"maxiter": args.maxiter, "gtol": 1e-6, "xtol": 1e-20},
            hessp=lambda x, p: p,
        )
    else:
        raise ValueError(f"Unknown method {args.method}")

    write_calc()


if __name__ == "__main__":
    raise SystemExit(main())
