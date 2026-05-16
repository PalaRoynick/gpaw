#!/usr/bin/env python3

import argparse
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from gpaw import GPAW, setup_paths
from gpaw.old.grid_descriptor import GridDescriptor
from gpaw.typing import ArrayND
from gpaw.xc.numericalxc import NumericalXC
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator

setup_paths.append('.')


def load(load_path: Path, name, gd: Optional[GridDescriptor] = None) -> ArrayND:
    array = np.load(load_path / (name + ".npy"))
    if gd is not None:
        array_og = array.copy()
        if array_og.ndim == 3:
            array = gd.empty()
        else:
            array = gd.empty(len(array_og))
        gd.distribute(array_og, array)

    return array


def load_stuff(load_path, paw_potmode="lebedev", no_ae=False) -> dict[str, ArrayND]:
    gs = GPAW(load_path)
    ham = gs.hamiltonian
    dens = gs.density
    data = {"n_atoms": len(gs.atoms)}
    if not no_ae:
        nae_g, finegd = dens.get_all_electron_density(
            gridrefinement=2, spos_ac=ham.spos_ac
        )
        # ae
        data["nae_g"] = nae_g.sum(0)
    else:
        dens.interpolate_pseudo_density()
    # pseudo
    data["n_sg"] = dens.nt_sg
    e_g = ham.finegd.zeros()
    v_sg = ham.finegd.zeros(ham.nspins)
    ham.xc.calculate(ham.finegd, dens.nt_sg, e_g=e_g, v_sg=v_sg)
    data["e_g"] = e_g
    data["vxc_sg"] = v_sg
    xc = ham.xc
    if isinstance(xc, NumericalXC):
        nxc = xc
        paw_potmode = nxc.paw_potmode.name
    else:
        nxc = NumericalXC.from_other_xc(
            xc, dens, paw_potmode, ham=ham, wfs=gs.wfs)

    nxc.initialize(dens, ham, gs.wfs)
    nxc.set_positions(ham.spos_ac)
    data["ae_vxc_sg"], gd = nxc.get_all_electron_xc()
    # atomic stuff
    for a, D_sp in dens.D_asp.items():
        data[f"D_{a}sp"] = dens.D_asp[a]
        dEdD_sp = np.zeros_like(dens.D_asp[a])
        ham.xc.calculate_paw_correction(
            gs.setups[a], dens.D_asp[a], dEdD_sp=dEdD_sp, a=a
        )
        data[f"dEdD_{a}sp"] = dEdD_sp
        if paw_potmode in ["lebedev", "angular"]:
            vxc_szg = nxc.vxc_xasz[0][a]
            vxct_szg = nxc.vxc_xasz[1][a]
        else:
            vxc_szg = np.zeros((1))
            vxct_szg = np.zeros((1))
        data[f"vxc_{a}szg"] = vxc_szg
        data[f"vxct_{a}szg"] = vxct_szg
    return data, nxc


def sample_field_along_line(n_field, cell_size, grid_spacing, point1, point2):
    point1 = np.array(point1, dtype=float)
    point2 = np.array(point2, dtype=float)
    cell_size = np.array(cell_size, dtype=float)
    grid_spacing = np.array(grid_spacing, dtype=float)
    ndim = n_field.ndim
    if point1.shape[0] != ndim or point2.shape[0] != ndim:
        raise ValueError(
            "point1 and point2 must each have length = n_field.ndim")
    # Determine how many sample points to take:
    #   We want sample spacing ≲ min(grid_spacing).
    line_vec = point2 - point1
    line_length = np.linalg.norm(line_vec)
    if line_length == 0:
        # Degenerate case: both points coincide.
        # return np.array([0.0]), np.array([_interpolate_at(point1)])
        return None, None
    # Roughly one sample per grid cell (along smallest axis)
    min_dx = np.min(grid_spacing)
    n_samples = max(int(np.ceil(line_length / min_dx)), 1) + 1
    # Build the N coordinate arrays along each axis:
    #   For axis i, there are n_field.shape[i] points, from 0 to cell_size[i].
    grid_axes = []
    for i in range(ndim):
        Ni = n_field.shape[i]
        Li = cell_size[i]
        if Ni < 2:
            raise ValueError(f"n_field.shape[{i}] = {
                             Ni} (needs ≥ 2 for interpolation)")
        # We place the grid points at 0, grid_spacing[i], 2*grid_spacing[i], …, (Ni-1)*grid_spacing[i].
        # In principle Li ≈ (Ni-1)*grid_spacing[i], but we trust the user to be consistent.
        grid_axes.append(np.linspace(0.0, Li, Ni))
    # Create the interpolator object
    interpolator = RegularGridInterpolator(
        tuple(grid_axes),
        n_field,
        bounds_error=False,
        fill_value=None,  # extrapolated points become NaN
    )
    # Parameterize the line by t ∈ [0, 1]:
    #   r(t) = point1 + t * (point2 - point1)
    t_vals = np.linspace(0.0, 1.0, n_samples)
    # Build an array of shape (n_samples, ndim) containing all sample positions:
    sample_positions = point1[None, :] + np.outer(t_vals, line_vec)
    # Interpolate
    values = interpolator(sample_positions)
    # Compute distances along the line:
    distances = t_vals * line_length
    return distances, values


def savefig(path, filename):
    plt.savefig(
        path / (filename + ".jpg"), bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot results from IKS calculations in GPAW."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the directory containing the results.",
    )
    parser.add_argument("--cut-x", type=int, default=None,
                        help="Cut along x-axis.")
    parser.add_argument("--cut-y", type=int, default=None,
                        help="Cut along y-axis.")
    parser.add_argument(
        "--z-lower", type=int, default=0, help="Lower z index for cuts."
    )
    parser.add_argument(
        "--z-upper", type=int, default=None, help="Upper z index for cuts."
    )
    parser.add_argument(
        "--plot-diff",
        action="store_true",
        help="Plot differences from target densities.",
    )
    parser.add_argument(
        "--lower-cp", type=int, default=0, help="Lower index for atomic plots."
    )
    parser.add_argument(
        "--upper-cp", type=int, default=None, help="Upper index for atomic plots."
    )
    parser.add_argument(
        "--no-ae", action="store_true", help="Do not plot all-electron parts."
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix for output files."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path
    if not path.is_dir():
        print(f"Error: {args.path} is not a valid directory.")

    try:
        L_iter = load(path, "L_iter")
        LD_iter = load(path, "LD_iter")
    except FileNotFoundError:
        L_iter = None
        LD_iter = None

    target = GPAW(path / "target.gpw")
    finegd = target.hamiltonian.finegd

    cf = norm(finegd.cell_cv, axis=0)
    interpolateg = partial(
        sample_field_along_line,
        grid_spacing=norm(finegd.h_cv, axis=0),
        cell_size=cf,
        point1=(0, 0, 0),
        point2=cf,
    )
    z = finegd.coords(2, pad=False)
    prefix = args.prefix
    # L_iter
    if L_iter is not None:
        plt.plot(np.log(L_iter), label="$\\mathcal{L}$ (smooth)", color="blue")
        plt.plot(np.log(LD_iter), color="red",
                 label="$\\mathcal{L}_D$ (atomic)")
        plt.xlabel("Iteration")
        plt.ylabel("$\\log(\\mathcal{L})$")
        plt.legend()
        savefig(path, prefix + "L_iter")
        plt.close()

    # load data
    iks, nxc = load_stuff(path / "iks.gpw", None, args.no_ae)
    target, _ = load_stuff(path / "target.gpw",
                           nxc.paw_potmode.name, args.no_ae)
    # fa_scf, _ = load_stuff(
    #     path / "fermi-amaldi-scf.gpw", nxc.paw_potmode.name, args.no_ae
    # )
    fa_warmup, _ = load_stuff(
        path / "fermi-amaldi-warmup.gpw", nxc.paw_potmode.name, args.no_ae
    )

    nx, ny, nz = target["n_sg"].shape[1:]
    print("shape:", nx, ny, nz)

    # cuts for the plot
    cut_x = args.cut_x or nx // 2
    cut_y = args.cut_y or ny // 2
    cz = slice(args.z_lower, args.z_upper)
    cut = (0, cut_x, cut_y, cz)
    print(target["n_sg"].shape)
    dat = [target, fa_warmup, iks]
    names = ["Target", "FA", "iKS"]
    linestyles = ["-", "-.", "--", ":"]
    colors = ["black", "darkgreen", "red", "orange"]

    # smooth density
    fig, ax = plt.subplots()
    for name, df, linestyle, color in zip(names, dat, linestyles, colors[:-1]):
        if name.lower() != "target":
            l1 = finegd.integrate(abs(df["n_sg"] - target["n_sg"]).sum(axis=0))
            label = "%s, $L_1=%.4f [a.u.]$" % (name, l1)
        else:
            label = name
        ax.plot(
            z[cz],
            df["n_sg"][cut],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        if args.plot_diff and name.lower() != "target":
            l1 = finegd.integrate(abs(df["n_sg"] - target["n_sg"]).sum(axis=0))
            d = df["n_sg"] - target["n_sg"]
            ax.plot(
                z[cz],
                d[cut],
                color=color,
                linewidth=1.0,
                label="diff %s $L_1=%.4f [a.u.]$" % (name, l1),
            )
    ax.set_title(
        "smooth pseudo density \n (cut at $i_x=%d$, $i_y=%d$)" % (
            cut[1], cut[2])
    )
    ax.legend()
    ax.set_xlabel("$z[a. u.]$")
    ax.set_ylabel("$\\tilde{n}(%d, %d, z) [a. u.]$" % (cut[1], cut[2]))
    savefig(path, prefix + "nt_sg_cut")

    fig, ax = plt.subplots()
    # cz = slice(nz // 2 - 60, nz // 2 + 60)
    for name, df, linestyle, color in zip(names, dat, linestyles, colors[:-1]):
        r, n = interpolateg(df["n_sg"][0])
        if name.lower() != "target":
            l1 = finegd.integrate(
                np.abs(df["n_sg"] - target["n_sg"]).sum(axis=0))
            label = "%s, $L_1=%.3f [a.u.]$" % (name, l1)
        else:
            label = name

        ax.plot(
            r,
            n,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
    ax.set_title(
        "smooth pseudo density along the diagonal $(0, 0, 0)$ to $(Lx, Ly, Lz)$"
    )
    ax.legend()
    ax.set_xlabel("$r[a. u.]$")
    ax.set_ylabel("$\\tilde{n}(r) [a. u.]$")
    savefig(path, prefix + "nt_sg_diagonal")

    # ae density
    if not args.no_ae:
        fig, ax = plt.subplots()
        for name, df, color, linestyle in zip(names, dat, colors[:-1], linestyles):
            cutae = (nx // 2, ny // 2, cz)
            if name.lower() != "target":
                l1 = finegd.integrate(np.abs(df["nae_g"] - target["nae_g"]))
                label = "%s, $L_1=%.3f [a.u.]$" % (name, l1)
            else:
                label = name

            ax.plot(
                z[cz],
                df["nae_g"][cutae],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
            )
        ax.set_title("ae density cut at $i_x=%d$, $i_y=%d$" %
                     (cutae[0], cutae[1]))
        ax.legend()
        savefig(path, prefix + "nae_g_cut")

        fig, ax = plt.subplots()
        for name, df, color, linestyle in zip(names, dat, colors, linestyles[:-1]):
            if name.lower() != "target":
                l1 = finegd.integrate(np.abs(df["nae_g"] - target["nae_g"]))
                label = "%s, $L_1=%.3f [a.u.]$" % (name, l1)
            else:
                label = name

            r, n = sample_field_along_line(
                df["nae_g"],
                cf,
                norm(finegd.h_cv, axis=0),
                (0.25, 0.5, 0.25) * cf,
                (0.25, 0.5, 0.5) * cf,
            )
            ax.plot(r, n, label=label, linestyle=linestyle,
                    color=color, linewidth=2.0)
        ax.set_title(
            "ae density along the line \n $(Lx/4, Ly/2, Lz/4)$ to $(Lx/4, Ly/2, Lz/2)$"
        )
        ax.set_xlabel("$r[a. u.]$")
        ax.set_ylabel("$n_{ae}(r) [a. u.]$")
        ax.legend()
        savefig(path, prefix + "nae_g_diagonal")

    # xc kernel
    fig, ax = plt.subplots()
    for name, df, color, linestyle in zip(names, dat, colors[:-1], linestyles):
        ax.plot(
            z[cz],
            df["e_g"][cut[1:]],
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
    ax.set_title("xc kernel single cut at $i_x=%d$, $i_y=%d$" %
                 (cut[1], cut[2]))
    ax.set_xlabel("$z[a. u.]$")
    ax.set_ylabel("$e_g(z) [a. u.]$")
    ax.legend()
    savefig(path, prefix + "e_g_cut")

    fig, ax = plt.subplots()
    for name, df, color, linestyle in zip(names, dat, colors[:-1], linestyles):
        r, e = interpolateg(df["e_g"])
        ax.plot(
            r, e, label=name.upper(), color=color, linestyle=linestyle, linewidth=2.0
        )
    ax.set_title("xc kernel along the diagonal (0, 0, 0) to (Lx, Ly, Lz)")
    ax.set_xlabel("$r[a. u.]$")
    ax.set_ylabel("$e_g(r) [a. u.]$")
    ax.legend()
    savefig(path, prefix + "e_g_diagonal")

    fig, ax = plt.subplots()
    for name, df, color, linestyle in zip(names, dat, colors[:-1], linestyles):
        r, e = sample_field_along_line(
            df["e_g"],
            cf,
            norm(finegd.h_cv, axis=0),
            (0.25, 0.25, 0.25) * cf,
            (0.75, 0.75, 0.75) * cf,
        )
        ax.plot(r, e, label=name, linestyle=linestyle,
                color=color, linewidth=2.0)
    ax.set_title(
        "xc kernel along $(Lx/4, Ly/4, Lz/4)$ to $(3Lx/4, 3Ly/4, 3Lz/4)$")
    ax.set_xlabel("$r[a. u.]$")
    ax.set_ylabel("$e_{g}(r) [a. u.]$")
    ax.legend()
    savefig(path, prefix + "e_g_diagonal_line")

    # ae xc potential
    fig, ax = plt.subplots()
    for name, df, linestyle, color in zip(names, dat, linestyles, colors[:-1]):
        ax.plot(
            z[cz],
            df["ae_vxc_sg"][cut],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=name,
        )

    ax.set_title("ae xc potential (cut at $i_x=%d$, $i_y=%d$)" %
                 (cut[1], cut[2]))
    ax.legend()
    ax.set_xlabel("$z[a. u.]$")
    ax.set_ylabel("$\\tilde{v_{xc}}(%d, %d, z) [a. u.]$" % (cut[1], cut[2]))
    savefig(path, prefix + "ae_vxc_sg_cut")

    fig, ax = plt.subplots()
    for name, df, linestyle, color in zip(names, dat, linestyles, colors[:-1]):
        r, v = interpolateg(df["ae_vxc_sg"][0])
        ax.plot(
            r,
            v,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=name,
        )

    ax.set_title(
        "AE xc potential along the diagonal $(0, 0, 0)$ to $(L_x, L_y, L_z)$")
    ax.legend()
    ax.set_xlabel("$r[a. u.]$")
    ax.set_ylabel("$v_{xc}(r) [a. u.]$ ")
    savefig(path, prefix + "ae_vxc_sg_diagonal")
    # smooth xc potential
    fig, ax = plt.subplots()
    for name, df, linestyle, color in zip(names, dat, linestyles, colors[:-1]):
        ax.plot(
            z[cz],
            df["vxc_sg"][cut],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=name,
        )
        if args.plot_diff and name.lower() != "target":
            l1 = finegd.integrate(
                abs(df["vxc_sg"] - target["vxc_sg"]).sum(axis=0))
            d = df["vxc_sg"] - target["vxc_sg"]
            ax.plot(
                z[cz],
                d[cut],
                color=color,
                linewidth=1.0,
                label="diff %s" % name,
            )

    ax.set_title("smooth xc potential (cut at $i_x=%d$, $i_y=%d$)" %
                 (cut[1], cut[2]))
    ax.legend()
    ax.set_xlabel("$z[a. u.]$")
    ax.set_ylabel("$\\tilde{v_{xc}}(%d, %d, z) [a. u.]$" % (cut[1], cut[2]))
    savefig(path, prefix + "vxc_sg_cut")

    fig, ax = plt.subplots()
    for name, df, linestyle, color in zip(names, dat, linestyles, colors[:-1]):
        r, v = interpolateg(df["vxc_sg"][0])
        ax.plot(
            r,
            v,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=name,
        )

    ax.set_title(
        "smooth xc potential along the diagonal $(0, 0, 0)$ to $(L_x, L_y, L_z)$"
    )
    ax.legend()
    ax.set_xlabel("$r[a. u.]$")
    ax.set_ylabel("$\\tilde{v}_{xc}(r) [a. u.]$ ")
    savefig(path, prefix + "vxc_sg_diagonal")

    fig, ax = plt.subplots()
    hf = norm(finegd.h_cv, axis=0)
    for name, df, linestyle, color in zip(names, dat, linestyles, colors[:-1]):
        r, v = sample_field_along_line(
            df["vxc_sg"][0], cf, hf, cf / 4.0, 3 * cf / 4.0)
        ax.plot(
            r,
            v,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=name,
        )

    ax.set_title(
        "smooth xc potential along the line $(L_x/4, L_y/4, L_z/4)$ to $(3L_x/4, 3L_y/4, 3L_z/4)$"
    )
    ax.legend()
    ax.set_xlabel("$r[a. u.]$")
    ax.set_ylabel("$\\tilde{v}_{xc}(r) [a. u.]$")
    savefig(path, prefix + "vxc_sg_diagonal_line")

    # atomic parts
    dat = [target, fa_warmup, iks]
    names = ["Target", "FA", "iKS"]
    linestyles = ["-", "-.", "--", ":"]
    colors = ["black", "darkgreen", "red", "orange"]
    markers = [("v", 6), ("*", 5), ("o", 3), ("s", 2)]
    cp = slice(args.lower_cp, args.upper_cp)
    cut = (0, cp)
    n_atoms = iks["n_atoms"]
    for a in range(n_atoms):
        if not target[f"D_{a}sp"].any():
            continue
        fig, ax = plt.subplots()
        p = np.arange(0, len(target[f"D_{a}sp"][0]))
        for name, df, marker, color in zip(names, dat, markers, colors[:-1]):
            ax.plot(
                p[cp],
                df[f"D_{a}sp"][cut],
                linestyle=":",
                marker=marker[0],
                markersize=marker[1],
                label=name,
                color=color,
            )
        ax.set_title(f"atomic density matrix a={a}")
        ax.set_xlabel("p (ij indices)")
        ax.set_ylabel("$D^{%d}_{sp} [a.u.]$" % a)
        ax.legend()
        savefig(path, prefix + f"D_{a}sp_cut")

    for a in range(n_atoms):
        if not target[f"D_{a}sp"].any():
            continue
        fig, ax = plt.subplots()
        p = np.arange(0, len(target[f"D_{a}sp"][0]))
        for name, df, marker, color in zip(names, dat, markers, colors[:-1]):
            ax.plot(
                target[f"D_{a}sp"][0],
                df[f"D_{a}sp"][0],
                marker=marker[0],
                markersize=marker[1],
                linewidth=0.0,
                color=color,
                label=name,
            )
        s = target[f"D_{a}sp"].std()
        ax.plot(
            [target[f"D_{a}sp"].min() - 3 * s,
             target[f"D_{a}sp"].max() + 3 * s],
            [target[f"D_{a}sp"].min() - 3 * s,
             target[f"D_{a}sp"].max() + 3 * s],
            linestyle=":",
            color="black",
            linewidth=2.0,
        )
        ax.set_xlabel("$D^{target}_{ij}$")
        ax.set_ylabel("$D^{%d}_{ij}$" % a)
        ax.set_xlim(
            target[f"D_{a}sp"].min() - 1.25 *
            s, target[f"D_{a}sp"].max() + 1.5 * s
        )
        ax.set_ylim(
            target[f"D_{a}sp"].min() - 1.25 *
            s, target[f"D_{a}sp"].max() + 1.5 * s
        )
        ax.set_title(f"atomic density matrix comparison a={a}")
        ax.legend()
        savefig(path, prefix + f"D_{a}sp_comparison")

    for a in range(n_atoms):
        if not target[f"dEdD_{a}sp"].any():
            continue
        fig, ax = plt.subplots()
        p = np.arange(0, len(target[f"dEdD_{a}sp"][0]))
        for name, df, marker, color in zip(names, dat, markers, colors[:-1]):
            ax.plot(
                p[cp],
                df[f"dEdD_{a}sp"][cut],
                linestyle=":",
                marker=marker[0],
                markersize=marker[1],
                color=color,
                label=name,
            )
        ax.set_title(
            f"xc correction $%s$ a={a}"
            % r"\left(\frac{\partial E_{xc}}{\partial D_{ij}}\right)"
        )
        ax.set_xlabel("index p (compact ij indices)")
        ax.set_ylabel("$\\Delta{v}_{ij}$")
        ax.legend()
        savefig(path, prefix + f"dEdD_{a}sp_cut")

    for a in range(n_atoms):
        if not target[f"D_{a}sp"].any():
            continue
        fig, ax = plt.subplots()
        p = np.arange(0, len(target[f"D_{a}sp"][0]))
        for name, df, marker, color in zip(names, dat, markers, colors[:-1]):
            ax.plot(
                target[f"dEdD_{a}sp"][cut],
                df[f"dEdD_{a}sp"][cut],
                marker=marker[0],
                markersize=marker[1],
                linewidth=0.0,
                color=color,
                label=name,
            )
        s = target[f"dEdD_{a}sp"].std()
        ax.plot(
            [target[f"dEdD_{a}sp"].min() - s, target[f"dEdD_{a}sp"].max() + s],
            [target[f"dEdD_{a}sp"].min() - s, target[f"dEdD_{a}sp"].max() + s],
            linestyle=":",
            color="black",
            linewidth=2.0,
        )
        ax.set_xlabel("$\\Delta{v}^{target}_{ij}$")
        ax.set_ylabel("$\\Delta{v}_{ij}$")
        ax.set_xlim(
            target[f"dEdD_{a}sp"].min() - 0.05 * s,
            target[f"dEdD_{a}sp"].max() + 0.1 * s,
        )
        ax.set_ylim(
            target[f"dEdD_{a}sp"].min() - 0.05 * s,
            target[f"dEdD_{a}sp"].max() + 0.1 * s,
        )
        ax.set_title(
            f"atomic xc corrections comparison $%s$ for a={a} "
            % r"\left(\frac{\partial \Delta{E}_{xc}}{\partial D_{ij}^a}\right)"
        )
        ax.legend()
        savefig(path, prefix + f"dEdD_{a}sp_comparison")

    for a in range(n_atoms):
        if not target[f"vxc_{a}szg"].any():
            continue
        for n in [0, 2, 5, 13, 20, 32, 48]:
            fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
            cut = (0, n)
            for name, df, color, linestyle in zip(
                ["FA", "Target", "iKS"],
                [fa_warmup, target, iks],
                ["darkgreen", "black", "red"],
                ["-.", "-", "--"],
            ):
                ax[0].plot(
                    df[f"vxc_{a}szg"][cut],
                    linewidth=2.0,
                    label=name,
                    color=color,
                    linestyle=linestyle,
                )
                ax[1].plot(
                    df[f"vxct_{a}szg"][cut],
                    linewidth=2.0,
                    label=name,
                    color=color,
                    linestyle=linestyle,
                )
            ax[0].set_title(f"atomic potential {n=}, {a=}")
            ax[0].legend()
            ax[1].set_title("atomic potential pseudo")
            savefig(path, prefix + f"vxc_{a}s{n}g")


if __name__ == "__main__":
    raise SystemExit(main())
