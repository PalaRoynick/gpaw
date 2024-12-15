from math import pi, sqrt

import numpy as np

from gpaw.density import Density
from gpaw.lfc import LFC, BasisFunctions
from gpaw.mpi import MPIComm
from gpaw.pw.density import ReciprocalSpaceDensity
from gpaw.typing import ArrayND
from gpaw.utilities import unpack_density


def copy_density(density: Density, spos_ac: ArrayND, kptband_comm: MPIComm) -> Density:
    density_args = (
        density.gd,
        density.finegd,
        density.nspins,
        density.collinear,
        density.charge,
        density.redistributor,
    )
    if isinstance(density, ReciprocalSpaceDensity):
        ecut = density.pd2.ecut
        new_dens = ReciprocalSpaceDensity(
            ecut, *density_args, background_charge=density.background_charge
        )
    else:
        new_dens = density.__class__(
            *density_args, background_charge=density.background_charge
        )

    new_dens.initialize(density.setups, density.timer, density.magmom_av, density.hund)
    new_dens.set_positions(spos_ac, density.atom_partition)
    new_dens.initialize_from_other_density(density, kptband_comm)
    return new_dens


def calculate_all_electron_density(density, spos_ac, kptband_comm):
    n_sg = density.nt_sG.copy()
    gd = density.gd
    splines = {}
    phi_aj = []
    phit_aj = []
    nc_a = []
    nct_a = []
    for a, id in enumerate(density.setups.id_a):
        if id in splines:
            phi_j, phit_j, nc, nct = splines[id]
        else:
            # Load splines:
            phi_j, phit_j, nc, nct = density.setups[a].get_partial_waves()[:4]
            splines[id] = (phi_j, phit_j, nc, nct)
        phi_aj.append(phi_j)
        phit_aj.append(phit_j)
        nc_a.append([nc])
        nct_a.append([nct])
    # Create localized functions from splines
    phi = BasisFunctions(gd, phi_aj)
    phit = BasisFunctions(gd, phit_aj)
    nc = LFC(gd, nc_a)
    nct = LFC(gd, nct_a)
    phi.set_positions(spos_ac)
    phit.set_positions(spos_ac)
    nc.set_positions(spos_ac)
    nct.set_positions(spos_ac)
    D_asp = density.atomdist.to_aux(density.D_asp)
    rho_MM = np.zeros((phi.Mmax, phi.Mmax))
    for s in range(density.nspins):
        M1 = 0
        for a, setup in enumerate(density.setups):
            ni = setup.ni
            D_sp = D_asp.get(a)
            if D_sp is None:
                raise ValueError("D_sp is None")
            rank = D_asp.partition.rank_a[a]
            D_asp.partition.comm.broadcast(D_sp, rank)
            M2 = M1 + ni
            rho_MM[M1:M2, M1:M2] = unpack_density(D_sp[s])
            M1 = M2

        phi.construct_density(rho_MM, n_sg[s], -1)
        phit.construct_density(-rho_MM, n_sg[s], -1)
        nc.add(n_sg[s])
        tmp_sg = np.zeros_like(n_sg)
        nct.add(tmp_sg[s])
        n_sg[s] -= tmp_sg[s]

    return n_sg


def ae2density(n_sg, density, spos_ac, kptband_comm, skip_core=False):
    """reverse operation of density.get_all_electron_density"""
    n_sg = n_sg.copy()
    gd = density.redistributor.aux_gd
    new_density = copy_density(density, spos_ac, kptband_comm)
    spos_ch = []  # for coreholes
    if density.nspins > 1 and not skip_core:
        # Support for corehole in spin-polarized system
        spos_ch = []
        n_ch_a = []
        n_ch = []

    # Add corrections to pseudo-density to get the AE-density
    splines = {}
    phi_aj = []
    phit_aj = []
    nc_a = []
    nct_a = []
    for a, id in enumerate(density.setups.id_a):
        if id in splines:
            phi_j, phit_j, nc, nct = splines[id]
        else:
            # Load splines:
            phi_j, phit_j, nc, nct = density.setups[a].get_partial_waves()[:4]
            splines[id] = (phi_j, phit_j, nc, nct)
        phi_aj.append(phi_j)
        phit_aj.append(phit_j)
        nc_a.append([nc])
        nct_a.append([nct])
        if density.setups[a].data.has_corehole and not skip_core and density.nspins > 1:
            work_setup = density.setups[a].data
            rmax = nc.get_cutoff()
            # work_setup.phicorehole_g
            phi_ch = work_setup.phicorehole_g
            phi_ch = np.where(abs(phi_ch) < 1e-160, 0, phi_ch)
            n_ch = np.dot(work_setup.fcorehole, phi_ch**2) / (4 * pi)
            n_ch_spl = work_setup.rgd.spline(n_ch, rmax, points=1000)
            n_ch_a.append([n_ch_spl])
            spos_ch.append(spos_ac[a])

    # Create localized functions from splines
    phi = BasisFunctions(gd, phi_aj)
    phit = BasisFunctions(gd, phit_aj)
    nc = LFC(gd, nc_a)
    nct = LFC(gd, nct_a)
    phi.set_positions(spos_ac)
    phit.set_positions(spos_ac)
    nc.set_positions(spos_ac)
    nct.set_positions(spos_ac)
    if spos_ch:
        nch = LFC(gd, n_ch_a)
        nch.set_positions(spos_ch)

    I_sa = np.zeros((density.nspins, len(spos_ac)))
    a_W = np.empty(len(phi.M_W), np.intc)
    W = 0
    for a in phi.atom_indices:
        nw = len(phi.sphere_a[a].M_w)
        a_W[W : W + nw] = a
        W += nw

    x_W = phi.create_displacement_arrays()[0]
    D_asp = density.atomdist.to_aux(density.D_asp)

    rho_MM = np.zeros((phi.Mmax, phi.Mmax))
    for s, I_a in enumerate(I_sa):
        M1 = 0
        for a, setup in enumerate(density.setups):
            ni = setup.ni
            D_sp = D_asp.get(a)
            if D_sp is None:
                D_sp = np.empty((density.nspins, ni * (ni + 1) // 2))
            else:
                I_a[a] = -(
                    (setup.Nct) / density.nspins
                    - sqrt(4 * pi) * np.dot(D_sp[s], setup.Delta_pL[:, 0])
                )

                if not skip_core:
                    I_a[a] += setup.Nc / density.nspins
                    if density.setups[a].data.has_corehole and density.nspins > 1:
                        I_a[a] -= pow(-1, s) * density.setups[a].data.fcorehole / 2

            rank = D_asp.partition.rank_a[a]
            D_asp.partition.comm.broadcast(D_sp, rank)
            M2 = M1 + ni
            rho_MM[M1:M2, M1:M2] = unpack_density(D_sp[s])
            M1 = M2

        assert np.all(n_sg[s].shape == phi.gd.n_c)
        phi.lfc.ae_valence_density_correction(-rho_MM, n_sg[s], a_W, I_a, x_W)
        phit.lfc.ae_valence_density_correction(+rho_MM, n_sg[s], a_W, I_a, x_W)

    a_W = np.empty(len(nc.M_W), np.intc)
    W = 0
    for a in nc.atom_indices:
        nw = len(nc.sphere_a[a].M_w)
        a_W[W : W + nw] = a
        W += nw
    scale = 1.0 / density.nspins

    for s, I_a in enumerate(I_sa):
        if not skip_core:
            nc.lfc.ae_core_density_correction(-scale, n_sg[s], a_W, I_a)
            # correct for ch here
            if spos_ch:
                nch.lfc.ae_core_density_correction(
                    scale * pow(-1, s), n_sg[s], a_W, I_a
                )
        nct.lfc.ae_core_density_correction(scale, n_sg[s], a_W, I_a)
        D_asp.partition.comm.sum(I_a)
        N_c = gd.N_c
        g_ac = np.around(N_c * spos_ac).astype(int) % N_c - gd.beg_c

        if not skip_core:
            for I, g_c in zip(I_a, g_ac):
                if (g_c >= 0).all() and (g_c < gd.n_c).all():
                    n_sg[s][tuple(g_c)] -= I / gd.dv

    new_density.initialize_directly_from_arrays(n_sg, None, density.D_asp)
    return new_density

