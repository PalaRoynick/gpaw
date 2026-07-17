import numpy as np
from ase.units import Bohr, Ha
from gpaw.core import UGArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.dft import GPAW
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.wannier.overlaps import find_directions


def pseudo_spread_integral(psit_nR: UGArray) -> tuple[np.ndarray, np.ndarray]:
    grid = psit_nR.desc
    assert grid.dtype == float
    dI_dc = find_directions(grid.icell, [1, 1, 1])
    # dI_dc = np.zeros((1, 3))
    nt_R = grid.new(dtype=complex).empty()
    z_dn = np.empty((len(dI_dc), len(psit_nR)), complex)
    for z_n, dI_c in zip(z_dn, dI_dc):
        eiGR_R = grid.eikr(dI_c)
        for n, psit_R in enumerate(psit_nR):
            nt_R.data[:] = psit_R.data**2 * eiGR_R
            z_n[n] = nt_R.integrate()
    return z_dn.T, dI_dc @ grid.icell


def paw_correction_spread_integral(G_dv,
                                   pos_av,
                                   P_ani: AtomArrays,
                                   dO_aii: AtomArrays) -> np.ndarray:
    z_dn = np.zeros((len(G_dv), P_ani.mydims[0]), complex)
    for a, P_ni in P_ani.items():
        dO_ii = dO_aii[a]
        e_d = np.exp(-2.j * np.pi * np.dot(G_dv, pos_av[a]))
        z_dn += np.einsum('ni, d, ij, nj -> dn',
                          P_ni.conj(), e_d, dO_ii, P_ni)
    return z_dn.T


def spread_and_center(z_d: np.ndarray,
                      G_dv: np.ndarray) -> tuple[float, np.ndarray]:
    relpos_d = (np.angle(z_d) / (2 * np.pi) + 1) % 1
    G_d = np.linalg.norm(G_dv, axis=1)
    n_dv = G_dv / G_d[:, np.newaxis]
    M_vv = n_dv.T @ n_dv
    pos_v = np.linalg.solve(M_vv, n_dv.T @ relpos_d)
    c_d = G_d**-2
    s = (0.5 * (c_d.sum() - abs(z_d)**2 @ c_d))**0.5
    return s, pos_v


def spread(wfs: PWFDWaveFunctions) -> tuple[np.ndarray, np.ndarray]:
    psit_nG = wfs.psit_nX
    grid = psit_nG.desc.minimal_uniform_grid()
    z_nd, G_dv = pseudo_spread_integral(psit_nG.ifft(grid=grid))
    z_nd += paw_correction_spread_integral(
        G_dv,
        wfs.relpos_ac @ grid.cell_cv,
        wfs.P_ani,
        {a: setup.dO_ii for a, setup in enumerate(wfs.setups)})
    return z_nd, G_dv


def main():
    import sys
    dft = GPAW(sys.argv[1]).dft
    dft.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    for wfs in dft.ibzwfs:
        z_nd, G_dv = spread(wfs)
        for n, z_d in enumerate(z_nd):
            eig = wfs.eig_n[n] * Ha
            s, pos_v = spread_and_center(z_d, G_dv)
            x, y, z = pos_v * Bohr
            s *= Bohr
            print(f'{n:3} {eig:10.3f} {s:7.1f} '
                  f'({x:7.3f}, {y:7.3f}, {z:7.3f})')


if __name__ == '__main__':
    main()
