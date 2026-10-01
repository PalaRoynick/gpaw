import numpy as np
import pytest

from gpaw.new.pw.hybrids import ibz2bz
from gpaw.dft import DFT
from gpaw.new.symmetry import SymmetrizationPlan


@pytest.mark.serial
@pytest.mark.parametrize(
    'gs',
    ['bi2i6_pw',
     'fancy_si_pw',
     'al_pw',
     'fe_pw',
     'co_pw',
     'gaas_pw',
     'v2br4_pw',
     'srvo3_pw'])
def test_ibz2bz(in_tmp_dir, gpw_files, gs):
    """Tests gpaw.ibz2bz.py

    Tests functionalities to take wavefunction and projections from
    ibz to full bz by comparing calculations with and without symmetry.
    """
    dft = DFT.from_gpw_file(gpw_files[gs])
    grid = dft.density.nt_sR.desc.new(dtype=complex)
    psit_K, _ = ibz2bz(
        dft.ibzwfs,
        dft.setups,
        dft.relpos_ac,
        grid,
        grid.fft_plans())

    ibz = dft.ibzwfs.ibz
    sym = ibz.symmetries
    l_aj = [setup.l_j for setup in dft.setups]
    symplan = SymmetrizationPlan(sym, l_aj)
    print(l_aj)

    e_skn = dft.ibzwfs.get_all_eigs()
    print(e_skn.shape)
    for K, psit in enumerate(psit_K):
        k = ibz.bz2ibz_K[K]
        K0 = ibz.ibz2bz_k[k]
        s = ibz.s_K[K]
        tr = ibz.time_reversal_K[K]
        psit0 = psit_K[K0]
        print(K, K0, k, dft.ibzwfs._wfs_u[k].k)
        print(dft.ibzwfs._wfs_u[k].eig_n)
        print(psit0.P_ani[0] / dft.ibzwfs._wfs_u[k].P_ani[0][:2])
        x = abs(psit0.P_ani[0] - dft.ibzwfs._wfs_u[k].P_ani[0][:2]).max()
        assert x < 1e-10
        P_ani = symplan.rotate_projections(
            #psit0.P_ani,
            dft.ibzwfs._wfs_u[k].P_ani,
            symnum=s, time_reversal=tr, kpt_c=psit.kpt_c,
            relpos_ac=dft.relpos_ac)
        d = P_ani.data[:2] - psit.P_ani.data
        print(P_ani[0][0] / psit.P_ani[0][0])
        #print(P_ani[0][1] / psit.P_ani[0][1])
        print(K, K0, ibz.bz2ibz_K[K], s, psit.kpt_c, psit0.kpt_c,
              sym.rotation_scc[s], tr)
        if abs(d).max() > 0.00001:
            print(abs(d).max())
            print(e_skn[:, k])
            break


if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    test_ibz2bz(42, {name: name}, name)
