"""Module for electron-phonon supercell properties.

Performance note: ``_local_qMM`` is not scaling well with MPI ranks.
Possible fix is to batch the phase-weighted k-sum so that ``V1t_sxMM[1:]``
is read once instead of once per k-point. Would however reduce readability.
"""

from __future__ import annotations

import numpy as np

from ase import Atoms
from ase.units import Bohr
from ase.utils.filecache import MultiFileJSONCache

from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UGArray, UGDesc
from gpaw.dft import GPAW
from gpaw.new.brillouin import MonkhorstPackKPoints
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.response import ResponseContext, ResponseContextInput, timer
from gpaw.typing import ArrayND
from gpaw.utilities import unpack_hermitian

from .filter import fourier_filter

sc_version = 1
# v1: saves natom, supercell, g_sNNMM.shape and dtype


class VersionError(Exception):
    """Error raised for wrong cache versions."""
    pass


class DFTAdapter:
    """Lightweight DFT data extracted from a GPAW calculator.
    """

    def __init__(self, *,
                 grid: UGDesc,
                 dH_asii_template: AtomArrays,
                 ibzwfs: IBZWaveFunctions) -> None:
        self.grid = grid
        self.dH_asii_template = dH_asii_template
        self.ibzwfs = ibzwfs  # might still be overkill
        ibz = ibzwfs.ibz
        bz = ibz.bz
        if not isinstance(bz, MonkhorstPackKPoints):
            raise ValueError(
                'Supercell el-ph requires a Monkhorst-Pack k-point grid; '
                f'got {type(bz).__name__}.')
        self.gamma_only = bz.gamma_only
        self.Nk_c = np.array(bz.size_c)
        self.trs = len(ibz.kpt_kc) != len(bz.kpt_Kc)
        gamma_idx = np.where(np.abs(ibz.kpt_kc).sum(axis=1) < 1e-13)[0]
        self.gamma_k = int(gamma_idx[0]) if gamma_idx.size else None

    @classmethod
    def from_calc(cls, calc) -> DFTAdapter:
        return cls(
            grid=calc.dft.density.nt_sR.desc,
            dH_asii_template=calc.dft.potential.dH_asii,
            ibzwfs=calc.dft.ibzwfs)


def _dH_asp_to_aii(dH_asp: dict, template: AtomArrays) -> AtomArrays:
    dH_aii = template.new()
    for a in dH_aii.keys():
        for s, dH_p in enumerate(dH_asp[a]):
            dH_aii[a][s] = unpack_hermitian(dH_p)
    return dH_aii


def _finalize_g_sMM(g_sMM: ArrayND,
                    g_s_gammaMM: ArrayND | None,
                    kpt_comm,
                    Nk_c: np.ndarray) -> ArrayND:
    """Reduce per-rank q-sums, apply TRS, normalise; return the real part.

    ``g_sMM`` is the per-rank IBZ-q sum at ``R = (0, 0, 0)``. With TRS folding
    half the BZ, ``g_s_gammaMM`` carries the Γ contribution on the rank that
    owns it (zeros elsewhere) so a comm-sum broadcasts it everywhere.
    """
    kpt_comm.sum(g_sMM)
    if g_s_gammaMM is not None:
        kpt_comm.sum(g_s_gammaMM)
        g_sMM = g_sMM + g_sMM.conj() - g_s_gammaMM
    g_sMM /= np.prod(Nk_c)
    if not np.all(np.abs(g_sMM.imag) < 1e-10):
        raise ValueError(
            f'MAX Im(g_MM): {np.amax(np.abs(g_sMM.imag)):.2e}')
    return g_sMM.real


class Supercell:
    """Class for supercell-related stuff."""

    def __init__(self, atoms: Atoms,
                 supercell_name: str = 'supercell',
                 supercell: tuple[int, int, int] = (1, 1, 1),
                 indices=None,
                 context: ResponseContextInput = '-') -> None:
        """Initialize supercell class.

        Parameters
        ----------
        atoms: Atoms
            The atoms to work on. Primitive cell.
        supercell_name: str
            User specified name of the generated JSON cache.
            Default is 'supercell'.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        context:
            Log/timer/comm bundle. Accepts a :class:`ResponseContext`, a dict
            of its kwargs, or a filename ('-' for stdout, None to silence).
            A fresh context is created from the input; the calculator's own
            logger/timer are not reused.
        """
        self.atoms = atoms
        self.supercell_name = supercell_name
        self.supercell = supercell
        if indices is None:
            self.indices = np.arange(len(atoms))
        else:
            self.indices = indices
        self.context = ResponseContext.from_input(context)

    @timer('Init')
    def _create_gpaw_calculator(self, calcdict: dict,
                                fd_name: str = 'elph') -> DFTAdapter:
        """Build an LCAO calculator.

        Parameters
        ----------
        calcdict: dict
            GPAW keyword arguments forwarded to the calculator.  ``basis``
            and ``parallel`` have defaults (``'dzp'`` and
            ``{'domain': 1, 'band': 1}``) which the user can override.  The
            following are set internally from the finite-difference cache
            and may not be overridden: ``mode``, ``gpts``, ``spinpol``,
            ``symmetry``.  Passing ``h`` is also forbidden because the
            grid is fixed by the cache.
        fd_name: str
            Name of the finite-difference JSON cache (default ``'elph'``).
        """
        cache = MultiFileJSONCache(fd_name)
        pot_shape = list(np.array(cache['eq']['Vt_sG']).shape)
        assert pot_shape[0] in (1, 2), 'only colinear spins'

        internal = {
            'mode': 'lcao',
            'gpts': pot_shape[1:],
            'spinpol': pot_shape[0] == 2,
            'symmetry': {'point_group': False},
        }
        defaults = {
            'basis': 'dzp',
            'parallel': {'domain': 1, 'band': 1},
        }
        if 'h' in calcdict:
            raise ValueError(
                "'h' cannot be set; grid is determined from cache")
        for key, val in internal.items():
            if key in calcdict and calcdict[key] != val:
                raise ValueError(
                    f"'{key}' is set internally to {val!r}"
                    ' and cannot be overridden')

        gpaw_kwargs = {**defaults, **calcdict, **internal}
        calc = GPAW(**gpaw_kwargs)
        calc.create_new_calculation(self.atoms * self.supercell)
        return DFTAdapter.from_calc(calc)

    @timer('Supercell entry')
    def _calculate_supercell_entry(self, a, v,
                                   v1t_sR: UGArray, dH1_aii: AtomArrays,
                                   dH_aii: AtomArrays,
                                   dft: DFTAdapter,
                                   ) -> tuple[ArrayND, ArrayND | None]:
        """One displacement → ``(g_sMM, g_s_gammaMM)``.

        ``g_sMM`` is the per-rank IBZ-q sum (no q axis); ``g_s_gammaMM`` is
        populated only on the Γ-owning rank when TRS folds the BZ, otherwise
        ``None``. Caller passes both to :func:`_finalize_g_sMM`.
        """
        ibzwfs = dft.ibzwfs
        nspins = ibzwfs.nspins
        # Basis / setups / dtype are shared across kpts; pull from any one.
        wfs0 = ibzwfs._wfs_u[0]
        bfs = wfs0.basis
        nao = wfs0.setups.nao
        dtype = wfs0.dtype

        # Per-displacement precomputes
        V1t_sxMM = np.array([bfs.calculate_potential_matrices(v1t_R.data)
                             for v1t_R in v1t_sR])
        dPdR_aqvMi = wfs0.tci_derivatives.manytci.P_aqMi(
            bfs.my_atom_indices, derivative=True)

        g_sMM = np.zeros((nspins, nao, nao), dtype)
        g_s_gammaMM = np.zeros_like(g_sMM) if dft.trs else None

        for wfs in ibzwfs:
            g_qMM = self._local_qMM(wfs, V1t_sxMM)
            g_qMM += self._nonlocal_qMM(wfs, a, v, dH1_aii, dH_aii,
                                        dPdR_aqvMi)
            g_sMM[wfs.spin] += g_qMM
            if dft.trs and wfs.k == dft.gamma_k:
                assert g_s_gammaMM is not None
                g_s_gammaMM[wfs.spin] = g_qMM
        return g_sMM, g_s_gammaMM

    @timer('Local qMM')
    def _local_qMM(self, wfs, V1t_sxMM: ArrayND) -> ArrayND:
        """Single-kpt LCAO matrix elements ``<M | dv_eff/dx | N>``."""
        bfs = wfs.basis
        dtype = wfs.dtype
        V_xMM = V1t_sxMM[wfs.spin]
        geff_MM = np.array(V_xMM[0], dtype=dtype)
        if np.issubdtype(dtype, np.complexfloating):
            phase_x = np.exp(-2j * np.pi * bfs.sdisp_xc[1:] @ wfs.kpt_c)
            geff_MM += np.einsum('x,xMN->MN', phase_x, V_xMM[1:],
                                 optimize=True)
            geff_MM += np.einsum('x,xMN->NM', phase_x.conj(), V_xMM[1:],
                                 optimize=True)
        return geff_MM

    @timer('Nonlocal qMM')
    def _nonlocal_qMM(self, wfs, a, v,
                      dH1_aii: AtomArrays, dH_aii: AtomArrays,
                      dPdR_aqvMi) -> ArrayND:
        """Single-kpt PAW projector terms: ``dH^a`` and ``dP^a`` (atom a)."""
        bfs = wfs.basis
        g_MM = np.zeros((wfs.setups.nao, wfs.setups.nao), wfs.dtype)

        for a_, dH1_sii in dH1_aii.items():
            if a_ not in bfs.my_atom_indices:
                continue
            P_Mi = wfs.P_aMi[a_]
            g_MM += P_Mi.conj() @ dH1_sii[wfs.spin] @ P_Mi.T

        if a in bfs.my_atom_indices:
            P_Mi = wfs.P_aMi[a]
            dP_Mi = dPdR_aqvMi[a][wfs.q][v]
            P1HP_MM = dP_Mi.conj() @ dH_aii[a][wfs.spin] @ P_Mi.T
            g_MM += P1HP_MM + P1HP_MM.T.conjugate()
        return g_MM

    def calculate_supercell_matrix(
        self, calcdict: dict, fd_name: str = 'elph',
        filter: str | None = None,
    ) -> None:
        """Calculate matrix elements of the el-ph coupling in the LCAO basis.

        This function calculates the matrix elements between LCAOs and local
        atomic gradients of the effective potential. The matrix elements are
        calculated for the supercell used to obtain finite-difference
        approximations to the derivatives of the effective potential wrt to
        atomic displacements.

        Parameters
        ----------
        calcdict: dict
            GPAW keyword arguments used to build the LCAO calculator
            internally via :meth:`_create_gpaw_calculator`.
        fd_name: str
            User specified name of the finite difference JSON cache.
            Default is 'elph'.
        filter: str
            Fourier filter atomic gradients of the effective potential. The
            specified components (``normal`` or ``umklapp``) are removed
            (default: None).
        """
        dft = self._create_gpaw_calculator(calcdict, fd_name)

        # JSON cache
        supercell_cache = MultiFileJSONCache(self.supercell_name)

        ibzwfs = dft.ibzwfs
        nspins = ibzwfs.nspins
        nao = ibzwfs._wfs_u[0].setups.nao
        self._log_summary(dft)

        # FD cache: equilibrium read once; per-displacement entries inside
        # the loop via _read_displacement_cache.
        fd_cache = MultiFileJSONCache(fd_name)
        if 'dr_version' not in fd_cache['info']:
            self.context.print('Cache created with old version. '
                               'Use older GPAW with electronphonon.py')
            raise VersionError
        assert tuple(self.supercell) == tuple(fd_cache['info']['supercell'])
        dH_aii = _dH_asp_to_aii(fd_cache['eq']['dH_all_asp'],
                                dft.dH_asii_template)

        with supercell_cache.lock('basis') as handle:
            if handle is not None:
                basis_info = self.set_basis_info(ibzwfs._wfs_u[0])
                handle.save(basis_info)

        if dft.gamma_only:
            self.context.print(
                'WARNING: Gamma-point calculation. '
                'Overlap with neighboring cell cannot be removed')

        # Calculate < i k | grad H | j k >, i.e. matrix elements in LCAO basis

        # NOTE: keep the xoutput naming scheme for backward-compat caches.
        for a in self.indices:
            for v in range(3):
                xoutput = 3 * a + v
                with supercell_cache.lock(str(xoutput)) as handle:
                    if handle is None:
                        continue

                    self.context.print(
                        f'{"xyz"[v]}-gradient of atom {a}')

                    with self.context.timer('Gradients'):
                        V1t_sG, dH1_asp = self._read_displacement_cache(
                            fd_cache, a, v, dft.grid, filter)
                        v1t_sR = dft.grid.empty(dims=(nspins,))
                        v1t_sR.scatter_from(V1t_sG)
                        dH1_aii = _dH_asp_to_aii(dH1_asp,
                                                 dft.dH_asii_template)
                    g_sMM, g_s_gammaMM = self._calculate_supercell_entry(
                        a, v, v1t_sR, dH1_aii, dH_aii, dft)
                    if not dft.gamma_only:
                        g_sMM = _finalize_g_sMM(g_sMM, g_s_gammaMM,
                                                ibzwfs.kpt_comm, dft.Nk_c)

                    # Reshape to global unit cell indices
                    N = np.prod(self.supercell)
                    nao_cell = nao // N
                    g_sNMNM = g_sMM.reshape((nspins, N, nao_cell, N, nao_cell))
                    g_sNNMM = g_sNMNM.swapaxes(2, 3).copy()
                    handle.save(g_sNNMM)

                if a == self.indices[0] and v == 0:
                    with supercell_cache.lock('info') as handle:
                        if handle is not None:
                            info = {
                                'sc_version': sc_version,
                                'natom': len(self.atoms),
                                'supercell': self.supercell,
                                'gshape': g_sNNMM.shape,
                                'gtype': g_sNNMM.dtype.name,
                            }
                            handle.save(info)

        self.context.write_timer()

    def set_basis_info(self, wfs) -> dict:
        """Return LCAO basis info (``M_a``, ``nao_a``) for atoms in the
        reference cell."""
        setups = wfs.setups
        bfs = wfs.basis
        natoms = len(self.atoms)
        return {'M_a': [bfs.M_a[a] for a in range(natoms)],
                'nao_a': [setups[a].nao for a in range(natoms)]}

    def _read_displacement_cache(self, cache: MultiFileJSONCache,
                                 a, v: int,
                                 grid: UGDesc,
                                 filter: str | None
                                 ) -> tuple[ArrayND, dict]:
        """Read FD cache for one displacement and return ``(V1t_sG, dH1_asp)``.
        """
        delta = cache['info']['delta']
        name = f'{a}{"xyz"[v]}'
        Vtm_sG = np.asarray(cache[name + '-']['Vt_sG'])
        Vtp_sG = np.asarray(cache[name + '+']['Vt_sG'])
        V1t_sG = (Vtp_sG - Vtm_sG) / (2 * delta / Bohr)

        dHm_asp = cache[name + '-']['dH_all_asp']
        dHp_asp = cache[name + '+']['dH_all_asp']
        dH1_asp = {atom: (dHp_asp[atom] - dHm_asp[atom]) / (2 * delta / Bohr)
                   for atom in dHm_asp}

        # New GPAW gather/scatter expects shape ``size_c - zerobc_c``;
        # legacy caches (full ``size_c``) get the leading zero-boundary
        # point sliced off on non-PBC axes.
        expected_c = grid.size_c - grid.zerobc_c
        pb = (~grid.pbc_c) & (V1t_sG.shape[-3:] != expected_c)
        if pb.any():
            slices = tuple(slice(1, None) if p else slice(None) for p in pb)
            V1t_sG = V1t_sG[..., slices[0], slices[1], slices[2]]

        if filter is not None:
            fourier_filter(self.atoms, self.supercell, V1t_sG,
                           components=filter)
        return V1t_sG, dH1_asp

    def _log_summary(self, dft: DFTAdapter) -> None:
        """Print a summary of the supercell-matrix calculation."""
        ibzwfs = dft.ibzwfs
        nibzk = len(ibzwfs.ibz.kpt_kc)
        nbzk = len(ibzwfs.ibz.bz.kpt_Kc)
        ndisp = 3 * len(self.indices)
        nao = ibzwfs._wfs_u[0].setups.nao
        self.context.print('Supercell matrix calculation')
        self.context.print('  Supercell:        '
                           f'{tuple(int(n) for n in self.supercell)}')
        self.context.print(f'  Atoms (prim):     {len(self.atoms)}')
        self.context.print(f'  Displacements:    {ndisp}')
        self.context.print(f'  Spins:            {ibzwfs.nspins}')
        self.context.print(f'  k-points (BZ):    {nbzk}')
        self.context.print(f'  k-points (IBZ):   {nibzk}'
                           f'  (TRS folded: {dft.trs})')
        self.context.print(f'  Basis funcs (M):  {nao}')
        self.context.print(f'  MPI Cores (total):    {self.context.comm.size}')
        self.context.print(f'  MPI Cores (k-points): {ibzwfs.kpt_comm.size}')
        self.context.print('')

    @classmethod
    def load_supercell_matrix(cls, name: str = 'supercell'
                              ) -> tuple[ArrayND, dict]:
        """Load supercell matrix from cache.

        Parameters
        ----------
        name: str
            User specified name of the cache.
        """
        # TODO: load by indices?
        supercell_cache = MultiFileJSONCache(name)
        if 'sc_version' not in supercell_cache['info']:
            print('Cache created with old version. Use electronphonon.py')
            raise VersionError
        shape = supercell_cache['info']['gshape']
        dtype = supercell_cache['info']['gtype']
        natom = supercell_cache['info']['natom']
        nx = natom * 3
        g_xsNNMM = np.empty([nx, ] + list(shape), dtype=dtype)
        for x in range(nx):
            g_xsNNMM[x] = supercell_cache[str(x)]
        basis_info = supercell_cache['basis']
        return g_xsNNMM, basis_info
