import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW, restart
from gpaw.atom.basis import BasisMaker
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import world
from gpaw.utilities.bandstructure import (
    create_bandstructure,
    create_non_ft_bandstructure,
)


def test_lcao_lcao_hamiltonian(in_tmp_dir, add_cwd_to_setup_paths):
    if world.rank == 0:
        basis = BasisMaker.from_symbol('Li').generate(1, 1)
        basis.write_xml()
    world.barrier()

    if 1:
        a = 2.7
        bulk = Atoms('Li', pbc=True, cell=[a, a, a])
        calc = GPAW(gpts=(8, 8, 8), kpts=(4, 4, 4), mode='lcao', basis='szp')
        bulk.calc = calc
        e = bulk.get_potential_energy()
        niter = calc.get_number_of_iterations()
        calc.write('temp.gpw')

    atoms, calc = restart('temp.gpw')
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    eigs = calc.get_eigenvalues(kpt=2)

    if world.rank == 0:
        eigs2 = sorted(np.linalg.eigvals(np.linalg.solve(S_kMM[2],
                                                         H_skMM[0, 2])).real)
        assert abs(sum(eigs - eigs2)) < 1e-8

        energy_tolerance = 0.0003
        niter_tolerance = 1  # Pulay vs MSR1
        assert e == pytest.approx(-1.82847, abs=energy_tolerance)
        assert niter == pytest.approx(5, abs=niter_tolerance)


def test_bandstructure_interpolation_uses_exact_lcao_matrices(
    in_tmp_dir, add_cwd_to_setup_paths
):
    if world.rank == 0:
        basis = BasisMaker.from_symbol('Li').generate(1, 1)
        basis.write_xml()
    world.barrier()

    a = 2.7
    bulk = Atoms('Li', pbc=True, cell=[a, a, a])
    calc = GPAW(gpts=(8, 8, 8),
                kpts=(2, 2, 2),
                mode='lcao',
                basis='szp',
                symmetry='off',
                txt=None)
    bulk.calc = calc
    bulk.get_potential_energy()

    bandpath = bulk.cell.bandpath(npoints=4)
    bs_from_matrices = create_bandstructure(
        calc,
        bandpath,
        cutoff=5,
        shift_kn=None,
    )
    bs_direct = create_non_ft_bandstructure(calc, bandpath)

    assert bs_from_matrices.energies == pytest.approx(bs_direct.energies,
                                                      abs=1e-10)

    shift_kn = np.zeros((len(calc.get_ibz_k_points()), 3))
    bs_from_zero_shift = create_bandstructure(
        calc,
        bandpath,
        cutoff=5,
        shift_kn=shift_kn,
    )
    assert bs_from_zero_shift.energies == pytest.approx(
        bs_direct.energies, abs=1e-10)
