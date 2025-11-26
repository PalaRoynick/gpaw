import numpy as np
from gpaw.core import UGArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.dft import GPAW
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.wannier.overlaps import find_directions


def pseudo_spread(psit_nR: UGArray) -> np.ndarray:
    grid = psit_nR.desc
    assert grid.dtype == float
    dI_dc = find_directions(grid.icell, [1, 1, 1])
    dI_dc = np.zeros((1, 3))
    nt_R = grid.new(dtype=complex).empty()
    z_dn = np.empty((len(dI_dc), len(psit_nR)), complex)
    for z_n, dI_c in zip(z_dn, dI_dc):
        eiGR_R = grid.eikr(dI_c)
        for n, psit_R in enumerate(psit_nR):
            nt_R.data[:] = psit_R.data**2 * eiGR_R
            z_n[n] = nt_R.integrate()
    return z_dn, dI_dc @ grid.icell


def paw_correction_spread(G_dv,
                          pos_av,
                          P_ani: AtomArrays,
                          dO_aii: AtomArrays) -> np.ndarray:
    z_dn = np.zeros((len(G_dv), P_ani.mydims[0]), complex)
    for a, P_ni in P_ani.items():
        dO_ii = dO_aii[a]
        e_d = np.exp(-2.j * np.pi * np.dot(G_dv, pos_av[a]))
        z_dn += np.einsum('ni, d, ij, nj -> dn',
                          P_ni.conj(), e_d, dO_ii, P_ni)
    return z_dn


def spread(wfs: PWFDWaveFunctions):
    psit_nG = wfs.psit_nX
    grid = psit_nG.desc.minimal_uniform_grid()
    z_dn, G_dv = pseudo_spread(psit_nG.ifft(grid=grid))
    print(z_dn, G_dv)
    z_dn += paw_correction_spread(G_dv,
                                  wfs.relpos_ac @ grid.cell_cv,
                                  wfs.P_ani,
                                  {a: setup.dO_ii
                                   for a, setup in enumerate(wfs.setups)})
    print(z_dn)


def main():
    import sys
    dft = GPAW(sys.argv[1]).dft
    dft.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    for wfs in dft.ibzwfs:
        spread(wfs)


if __name__ == '__main__':
    main()
