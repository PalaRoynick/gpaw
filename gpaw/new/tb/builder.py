from __future__ import annotations

from math import pi
from types import SimpleNamespace

import numpy as np
# from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from ase.units import Bohr, Ha

from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArraysLayout, AtomDistribution
from gpaw.core.domain import Domain
from gpaw.core.matrix import Matrix
from gpaw.lcao.tci import TCIExpansions
from gpaw.lfc import BasisFunctions
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import zips as zip
from gpaw.new.lcao.builder import LCAODFTComponentsBuilder, create_lcao_ibzwfs
from gpaw.new.lcao.hamiltonian import CollinearHamiltonianMatrixCalculator
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.setup import Setup
from gpaw.spline import Spline
from gpaw.utilities.timing import NullTimer
from gpaw.typing import Array3D
from gpaw.new.scf import SCFContext


class TBHamiltonianMatrixCalculator(CollinearHamiltonianMatrixCalculator):
    def _calculate_potential_matrix(self,
                                    wfs: LCAOWaveFunctions,
                                    V_xMM: Array3D = None) -> Matrix:
        return wfs.V_MM


class TBHamiltonian:
    def __init__(self,
                 basis: BasisFunctions):
        self.basis = basis

    def apply(self):
        raise NotImplementedError

    def create_hamiltonian_matrix_calculator(
            self,
            potential) -> TBHamiltonianMatrixCalculator:
        ncomponents = potential.dH_asii.dims[0]
        dH_saii = [{a: dH_sii[s]
                    for a, dH_sii in potential.dH_asii.items()}
                   for s in range(ncomponents)]

        V_sxMM = [np.zeros(0) for _ in range(ncomponents)]

        return TBHamiltonianMatrixCalculator(V_sxMM, dH_saii, self.basis)


class NoGrid(Domain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gd = SimpleNamespace(
            get_grid_spacings=lambda: [0, 0, 0],
            cell_cv=self.cell_cv,
            pbc_c=self.pbc_c,
            N_c=[0, 0, 0],
            dv=0.0)
        self.size = np.array([0, 0, 0])
        self.zerobc_c = np.zeros(3, bool)

    def empty(self, shape=(), comm=serial_comm, xp=None):
        return DummyFunctions(self, shape, comm)

    def ranks_from_fractional_positions(self, relpos_ac):
        return AtomDistribution.from_number_of_atoms(
            len(relpos_ac), self.comm).rank_a


class DummyFunctions(DistributedArrays[NoGrid]):
    def __init__(self,
                 grid: NoGrid,
                 dims: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm):
        DistributedArrays. __init__(self, dims, (),
                                    comm, grid.comm, None, np.nan,
                                    grid.dtype)
        self.desc = grid

    def integrate(self, other=None):
        if other is None:
            return np.ones(self.dims)
        return np.zeros(self.dims + other.dims)

    def new(self, zeroed=False):
        return self

    def __getitem__(self, index):
        if isinstance(index, int):
            dims = self.dims[1:]
        else:
            dims = self.dims
        return DummyFunctions(self.desc, dims, comm=self.comm)

    def moment(self):
        return np.zeros(3)

    def to_xp(self, xp):
        return self

    def gather(self):
        return self

    def scatter_from(self, a):
        pass


class PSCoreDensities:
    xp = np

    def __init__(self, grid, relpos_ac):
        self.layout = AtomArraysLayout([1] * len(relpos_ac),
                                       grid.comm)

    def to_uniform_grid(self, out, scale):
        pass

    def new(self, desc, atomdist):
        return self


class TBPotentialCalculator(PotentialCalculator):
    def __init__(self,
                 xc,
                 setups,
                 atoms,
                 repulsions,
                 domain_comm):
        super().__init__(xc, None, setups,
                         relpos_ac=atoms.get_scaled_positions(),
                         environment=None)
        self.atoms = atoms.copy()
        self.repulsions = repulsions
        self.domain_comm = domain_comm
        self.force_av = None
        self.stress_vv = None

    def calculate_pseudo_potential(self, density, ibzwfs, vHt_r):
        vt_sR = density.nt_sR

        atoms = self.atoms
        energy, force_av, stress_vv = pairpot(
            atoms, self.repulsions)
        energy /= Ha
        self.force_av = force_av * Bohr / Ha

        vol = abs(np.linalg.det(atoms.cell[atoms.pbc][:, atoms.pbc]))
        self.stress_vv = stress_vv / vol * Bohr**atoms.pbc.sum() / Ha

        V_aL = AtomArraysLayout([9] * len(self.atoms),
                                density.D_asii.layout.atomdist).zeros()
        for a, V_L in V_aL.items():
            V_L[0] = self.setups[a].W

        return ({'coulomb': 0.0,
                 'zero': 0.0,
                 'xc': energy,
                 'external': 0.0},
                vt_sR,
                None,
                DummyFunctions(density.nt_sR.desc),
                V_aL,
                np.nan)

    def _move(self, relpos_ac, ndensities):
        self.atoms.set_scaled_positions(relpos_ac)
        self.force_av = None
        self.stress_vv = None

    def force_contributions(self, density, potential):
        return {}, {}, {a: self.force_av[a:a + 1]
                        for a in density.D_asii.keys()}

    def stress_contribution(self, ibzwfs, density, potential):
        return self.stress_vv


class DummyXC:
    no_forces = False
    xc = None
    exx_fraction = 0.0

    def __init__(self, xc):
        self.xc = xc

    def calculate_paw_correction(self, setup, D_sp, dH_sp):
        return self.xc.calculate_paw_correction(setup, D_sp, dH_sp)


class TBSCFLoop:
    def __init__(self, hamiltonian, occ_calc, eigensolver, comm):
        self.hamiltonian = hamiltonian
        self.occ_calc = occ_calc
        self.eigensolver = eigensolver
        self.comm = comm
        self.convergence = {}

    def iterate(self,
                ibzwfs,
                density,
                potential,
                energies,
                pot_calc,
                convergence=None,
                maxiter=None,
                calculate_forces=None,
                log=None):
        self.eigensolver.iterate(ibzwfs, density, potential, self.hamiltonian)
        e_band, e_entropy, e_extrapolation = ibzwfs.calculate_occs(
            self.occ_calc,
            nelectrons=density.nvalence - density.charge)
        energies.set(band=e_band,
                     entropy=e_entropy,
                     extrapolation=e_extrapolation)

        yield SCFContext(
            log,
            1,
            energies,
            ibzwfs, density, potential,
            0.0, 0.0, 0.0,
            self.comm, calculate_forces,
            pot_calc, False)

        # potential, _, _ = pot_calc.calculate(
        #     density, None, potential.vHt_x)


class DummyBasis:
    def __init__(self, setups, atomdist, band_comm):
        self.my_atom_indices = atomdist.indices
        Mmax = setups.nao
        myM = (Mmax + band_comm.size - 1) // band_comm.size
        self.Mstart = min(band_comm.rank * myM, Mmax)
        self.Mstop = min((band_comm.rank + 1) * myM, Mmax)

    def add_to_density(self, nt_sR, f_asi):
        pass

    def construct_density(self, rho_MM, nt_G, q):
        pass


class TBDFTComponentsBuilder(LCAODFTComponentsBuilder):
    def fix_setups(self):
        for setup in self.setups.setups.values():
            try:
                setup.vt_g
            except AttributeError:
                setup.vt_g, setup.W = calculate_pseudo_potential(
                    setup, self.xc.xc)

    def check_cell(self, cell):
        pass

    def create_uniform_grids(self):
        grid = NoGrid(
            self.atoms.cell.complete() / Bohr,
            self.atoms.pbc,
            dtype=self.dtype,
            comm=self.communicators['d'])
        return grid, grid

    def get_pseudo_core_densities(self):
        return PSCoreDensities(self.grid, self.relpos_ac)

    def get_pseudo_core_ked(self):
        return PSCoreDensities(self.grid, self.relpos_ac)

    def create_basis_set(self):
        self.fix_setups()
        self.basis = DummyBasis(self.setups,
                                self.atomdist,
                                self.communicators['b'])
        return self.basis

    def create_hamiltonian_operator(self):
        return TBHamiltonian(self.basis)

    def create_potential_calculator(self):
        reps = self.params.mode.repulsions or {}
        symbols = [setup.symbol for setup in self.setups.setups.values()]
        repulsions = {}
        for sym1 in symbols:
            for sym2 in symbols:
                rep = reps.get((sym1, sym2),
                               reps.get((sym2, sym1), None))
                if rep is None:
                    rep = Repulsion.default(sym1, sym2)
                repulsions[(sym1, sym2)] = rep

        xc = DummyXC(self.xc)
        return TBPotentialCalculator(xc, self.setups, self.atoms,
                                     repulsions,
                                     self.communicators['d'])

    def create_scf_loop(self):
        occ_calc = self.create_occupation_number_calculator()
        hamiltonian = self.create_hamiltonian_operator()
        eigensolver = self.create_eigensolver(hamiltonian)
        return TBSCFLoop(hamiltonian, occ_calc, eigensolver,
                         self.communicators['w'])

    def create_ibz_wave_functions(self,
                                  basis: BasisFunctions,
                                  potential,
                                  *,
                                  coefficients=None):
        ibzwfs, tciexpansions = create_lcao_ibzwfs(
            basis,
            self.ibz, self.communicators, self.setups,
            self.relpos_ac, self.grid, self.dtype,
            self.nbands, self.ncomponents, self.atomdist, self.nelectrons)

        vtphit: dict[Setup, list[Spline]] = {}

        for setup in self.setups.setups.values():
            vt_r = setup.vt_g
            vt_r[-1] = 0.0  # ???
            vt = setup.rgd.spline(vt_r, points=300)
            vtphit_j = []
            for phit in setup.basis_functions_J:
                rc = phit.get_cutoff()
                r_g = np.linspace(0, rc, 150)
                phit_g = phit.map(r_g)
                vt_g = vt.map(r_g) / (4 * pi)**0.5
                vtphit_j.append(Spline.from_data(phit.l, rc, vt_g * phit_g))
            vtphit[setup] = vtphit_j

        setup_I = list(self.setups.setups.values())
        vtciexpansions = TCIExpansions([s.basis_functions_J
                                        for s in setup_I],
                                       [vtphit[s] for s in setup_I],
                                       tciexpansions.I_a)

        kpt_qc = np.array([wfs.kpt_c for wfs in ibzwfs])
        manytci = vtciexpansions.get_manytci_calculator(
            self.setups, self.grid._gd, self.relpos_ac,
            kpt_qc, self.dtype, NullTimer())

        manytci.Pindices = manytci.Mindices
        my_atom_indices = self.atomdist.indices

        ells = {}
        for setup in setups.setups.values():
            symbol = setup.symbol
            ells[symbol] = [phit.l for phit in setup.basis_functions_J]

        fudge_factors = create_hop_factors(
            {symbol: ['spdf'[l] for l in ll] for symbol, ll in ells.items()},
            self.params.hops)

        domain_comm = self.communicators['d']

        for wfs, V_MM in zip(ibzwfs, manytci.P_qIM(my_atom_indices)):
            wfs.V_MM = wfs.T_MM.new()
            M1, M2 = wfs.V_MM.dist.my_row_range()
            V_mM = wfs.V_MM.data
            V_mM[:] = V_MM.T[M1:M2].toarray()
            domain_comm.sum(V_mM)
            wfs.V_MM.add_hermitian_conjugate()

        def mmsl():
            M = 0
            for setup in setups:
                for l in ells[setup.symbol]:
                    yield M, M + 2 * l + 1, setup.symbol, l
                    M += 2 * l + 1

        for wfs in ibzwfs:
            M1, M2 = wfs.V_MM.dist.my_row_range()
            V_MM = wfs.V_MM.data

            V_mM *= fudge_factor / domain_comm.size

            for Ma1, Ma2, symbola, la in mmsl():
                for Mb1, Mb2, symbolb, lb in mmsl():
                    if Ma1 == Mb1:
                        continue
                    Ma1 = max(Ma1 - M1, 0)
                    Ma2 = min(Ma2 - M1, M2 - M1)
                    if Ma1 >= Ma2:
                        continue
                    f = factors[symbola + symbolb]['spdf'[la] + 'spdf'[lb]]
                    V_MM[Ma1:Ma2, Mb1:Mb2] *= 0.5 * f

        return ibzwfs


def create_hop_factors(ells: dict[str, list[str]],
                       hops: dict[str, dict[str, float]]
                       ) -> dict[str, dict[str, float]]:
    """Create hopping matrix-element scaling factors.

    >>> create_hop_factors({'C': ['s', 'p']}, {'C':{'pp': 0.75}})
    {}
    """
    factors = {}

    for symbol, ll in ells.items():
        hop = hops.get(symbol, {})
        ff = factors[symbol + symbol] = {}
        for l1 in ll:
            for l2 in ll:
                f = hop.get(l1 + l2, hop.get(l2 + l1, 1.0))
                ff[l1 + l2] = f

    for symbol1, ll1 in ells.items():
        for symbol2, ll2 in ells.items():
            if symbol1 == symbol2:
                continue
            hop = hops.get(symbol1 + symbol2)
            if hop is None:
                hop = {ll[::-1]: f
                       for ll, f in hops.get(symbol2 + symbol1, {}).items()}
            ff = factors[symbol1 + symbol2] = {}
            for l1 in ll1:
                for l2 in ll2:
                    f = hop.get(l1 + l2, 1.0))
                    ff[l1 + l2] = f
    return factors


class Repulsion:
    def __init__(self, f, dfdr=None, rc=None):
        self.f = f
        self.dfdr = dfdr
        if rc is None:
            for rc in np.linspace(1.0, 10.0, 31):
                if f(rc) == 0.0:
                    break
            else:
                1 / 0
        self.rc = rc

    @staticmethod
    def default(sym1, sym2):
        # radii[symbol] = covalent_radii[atomic_numbers[symbol]]
        return Repulsion(lambda r: 0.0, lambda r: 0.0, rc=0.5)


def pairpot(atoms, repulsions):
    """Simple pair-potential for testing.

    >>> from ase import Atoms
    >>> r = covalent_radii[1]
    >>> atoms = Atoms('H2', [(0, 0, 0), (0, 0, 2 * r)])
    >>> e, f, s = pairpot(atoms)
    >>> print(f'{e:.6f} eV')
    -9.677419 eV
    >>> f
    array([[0., 0., 0.],
           [0., 0., 0.]])

    """
    symbol_a = atoms.symbols
    rcutmax = max(rep.rc for rep in repulsions)
    energy = 0.0
    force_av = np.zeros((len(atoms), 3))
    stress_vv = np.zeros((3, 3))

    # PARALLELIZE!!!
    for i, j, d, D_v in zip(*neighbor_list('ijdD', atoms, rcutmax)):
        rep = repulsions[(symbol_a[i], symbol_a[j])]
        if d < rep.rc:
            energy += 0.5 * rep.f(d)
            f = -0.5 * rep.dfdr(d)
            F_v = D_v * f / d
            force_av[i] += F_v
            force_av[j] -= F_v
            # print(i, j, d, D_v, F_v)
            stress_vv += np.outer(F_v, D_v)

    return energy, force_av, stress_vv


def calculate_pseudo_potential(setup: Setup, xc):
    phit_jg = np.array(setup.data.phit_jg)
    rgd = setup.rgd

    # Density:
    nt_g = np.einsum('jg, j, jg -> g',
                     phit_jg, setup.f_j, phit_jg) / (4 * pi)
    nt_g += setup.data.nct_g * (1 / (4 * pi)**0.5)

    # XC:
    vt_g = rgd.zeros()
    xc.calculate_spherical(rgd, nt_g[np.newaxis], vt_g[np.newaxis])

    # Zero-potential:
    vt_g += setup.data.vbar_g / (4 * pi)**0.5

    # Coulomb:
    g_g = setup.ghat_l[0].map(rgd.r_g)
    Q = -rgd.integrate(nt_g) / rgd.integrate(g_g)
    rhot_g = nt_g + Q * g_g
    vHtr_g = rgd.poisson(rhot_g)

    W = rgd.integrate(g_g * vHtr_g, n=-1) / (4 * pi)**0.5

    vtr_g = vt_g * rgd.r_g + vHtr_g

    vtr_g[1:] /= rgd.r_g[1:]
    vtr_g[0] = vtr_g[1]

    return vtr_g * (4 * pi)**0.5, W


def poly():
    """Polynomium used for pair potential."""
    import matplotlib.pyplot as plt
    c = np.linalg.solve([[1, 0.5, 0.25, 0.125],
                         [1, 1, 1, 1],
                         [0, 1, 1, 0.75],
                         [0, 1, 2, 3]],
                        [0, -1, 0, 0])
    print(c)
    d = np.linspace(0.5, 2, 101)
    plt.plot(d, c[0] + c[1] / d + c[2] / d**2 + c[3] / d**3)
    plt.show()


if __name__ == '__main__':
    poly()
