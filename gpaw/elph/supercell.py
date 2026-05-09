"""Module for electron-phonon supercell properties."""

from __future__ import annotations

import numpy as np

from ase import Atoms
from ase.units import Bohr
from ase.utils.filecache import MultiFileJSONCache

from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UGArray, UGDesc
from gpaw.dft import GPAW
from gpaw.new.logger import Logger
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

    Holds only the arrays/scalars needed by the elph supercell code so
    inner functions don't depend on full GPAW objects.
    """

    def __init__(self, *,
                 grid: UGDesc,
                 dH_asii_template: AtomArrays,
                 log: Logger,
                 kd) -> None:
        self.grid = grid
        self.dH_asii_template = dH_asii_template
        self.log = log
        self.kd = kd
        # Derived k-point info. Will move to a native ibzwfs source later.
        self.trs = len(kd.ibzk_kc) != len(kd.bzk_kc)
        gamma_idx = np.where(np.abs(kd.ibzk_kc).sum(axis=1) < 1e-13)[0]
        self.gamma_k = int(gamma_idx[0]) if gamma_idx.size else None

    @classmethod
    def from_calc(cls, calc) -> DFTAdapter:
        return cls(
            grid=calc.dft.density.nt_sR.desc,
            dH_asii_template=calc.dft.potential.dH_asii,
            log=calc.log,
            kd=calc.wfs.kd)


def _dH_asp_to_aii(dH_asp: dict, template: AtomArrays) -> AtomArrays:
    """Unpack packed-Hermitian per-atom dict into a native ``AtomArrays``.

    ``template`` provides the layout and ``dims=(ncomponents,)``.
    """
    dH_aii = template.new()
    for a in dH_aii.keys():
        for s, dH_p in enumerate(dH_asp[a]):
            dH_aii[a][s] = unpack_hermitian(dH_p)
    return dH_aii

def _finalize_g_sMM(g_sMM: ArrayND,
                    g_s_gammaMM: ArrayND | None,
                    kd) -> ArrayND:
    """Reduce per-rank q-sums, apply TRS, normalise; return the real part.

    ``g_sMM`` is the per-rank IBZ-q sum at ``R = (0, 0, 0)``. With TRS folding
    half the BZ, ``g_s_gammaMM`` carries the Γ contribution on the rank that
    owns it (zeros elsewhere) so a comm-sum broadcasts it everywhere.
    """
    kd.comm.sum(g_sMM)
    if g_s_gammaMM is not None:
        kd.comm.sum(g_s_gammaMM)
        g_sMM = g_sMM + g_sMM.conj() - g_s_gammaMM
    g_sMM /= np.prod(kd.N_c)
    if not np.all(np.abs(g_sMM.imag) < 1e-10):
        raise ValueError(
            f'MAX Im(g_MM): {np.amax(np.abs(g_sMM.imag)):.2e}')
    return g_sMM.real


class Supercell:
    """Class for supercell-related stuff."""

    def __init__(self, atoms: Atoms, supercell_name: str = "supercell",
                 supercell: tuple = (1, 1, 1), indices=None) -> None:
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
        """
        self.atoms = atoms
        self.supercell_name = supercell_name
        self.supercell = supercell
        if indices is None:
            self.indices = np.arange(len(atoms))
        else:
            self.indices = indices

    def _create_gpaw_calculator(self, calcdict, fd_name='elph'):
        """Build an LCAO calculator and return its ``(wfs, timer)``.

        Parameters
        ----------
        calcdict: dict
            GPAW keyword arguments forwarded to the calculator.  ``basis``
            defaults to ``'dzp'`` if not supplied.  The following parameters
            are set internally from the finite-difference cache and may not
            be overridden: ``mode``, ``gpts``, ``spinpol``, ``symmetry``,
            ``parallel``.  Passing ``h`` is also forbidden because the grid
            is fixed by the cache.
        fd_name: str
            Name of the finite-difference JSON cache (default ``'elph'``).

        Returns
        -------
        wfs, dft, timer:
            ``wfs`` is the backwards-compat wave-function shim, ``dft`` a
            :class:`DFTAdapter` with extracted DFT data, ``timer`` the
            calculator's ``Timer``.
        """
        cache = MultiFileJSONCache(fd_name)
        pot_shape = list(np.array(cache['eq']['Vt_sG']).shape)
        assert pot_shape[0] in (1, 2), 'only colinear spins'

        internal = {
            'mode': 'lcao',
            'gpts': pot_shape[1:],
            'spinpol': pot_shape[0] == 2,
            'symmetry': {'point_group': False},
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

        gpaw_kwargs = {'basis': 'dzp', **calcdict, **internal}
        calc = GPAW(**gpaw_kwargs)
        calc.create_new_calculation(self.atoms * self.supercell)
        return calc.wfs, DFTAdapter.from_calc(calc), calc.timer

    def _calculate_supercell_entry(self, a, v,
                                   v1t_sR: UGArray, dH1_aii: AtomArrays,
                                   wfs, dH_aii: AtomArrays,
                                   dft: DFTAdapter,
                                   ) -> tuple[ArrayND, ArrayND | None]:
        """One displacement → ``(g_sMM, g_s_gammaMM)``.

        ``g_sMM`` is the per-rank IBZ-q sum (no q axis); ``g_s_gammaMM`` is
        populated only on the Γ-owning rank when TRS folds the BZ, otherwise
        ``None``. Caller passes both to :func:`_finalize_g_sMM`.
        """
        bfs = wfs.basis_functions
        nspins = wfs.nspins
        nao = wfs.setups.nao
        dtype = wfs.dtype

        # Per-displacement precomputes (size << old g_sqMM).
        V1t_sxMM = np.array([bfs.calculate_potential_matrices(v1t_R.data)
                             for v1t_R in v1t_sR])
        P_aqMi = getattr(wfs, 'P_aqMi', None)
        dPdR_aqvMi = wfs.manytci.P_aqMi(bfs.my_atom_indices, derivative=True)

        g_sMM = np.zeros((nspins, nao, nao), dtype)
        g_s_gammaMM = np.zeros_like(g_sMM) if dft.trs else None

        for kpt in wfs.kpt_u:
            g_qMM = self._local_qMM(kpt, V1t_sxMM, wfs)
            g_qMM += self._nonlocal_qMM(kpt, a, v, dH1_aii, dH_aii, wfs,
                                        P_aqMi, dPdR_aqvMi)
            g_sMM[kpt.s] += g_qMM
            if dft.trs and kpt.k == dft.gamma_k:
                g_s_gammaMM[kpt.s] = g_qMM
        return g_sMM, g_s_gammaMM

    def _local_qMM(self, kpt, V1t_sxMM: ArrayND, wfs) -> ArrayND:
        """Single-kpt LCAO matrix elements ``<M | dv_eff/dx | N>``."""
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        V_xMM = V1t_sxMM[kpt.s]
        geff_MM = np.array(V_xMM[0], dtype=dtype)
        if np.issubdtype(dtype, np.complexfloating):
            kpt_c = wfs.kd.ibzk_kc[kpt.k]
            phase_x = np.exp(-2j * np.pi * bfs.sdisp_xc[1:] @ kpt_c)
            geff_MM += np.einsum('x,xMN->MN', phase_x, V_xMM[1:],
                                 optimize=True)
            geff_MM += np.einsum('x,xMN->NM', phase_x.conj(), V_xMM[1:],
                                 optimize=True)
        return geff_MM

    def _nonlocal_qMM(self, kpt, a, v,
                      dH1_aii: AtomArrays, dH_aii: AtomArrays,
                      wfs, P_aqMi, dPdR_aqvMi) -> ArrayND:
        """Single-kpt PAW projector terms: ``dH^a`` (all atoms) + ``dP^a`` (atom a)."""
        bfs = wfs.basis_functions
        nao = wfs.setups.nao
        g_MM = np.zeros((nao, nao), wfs.dtype)

        for a_, dH1_sii in dH1_aii.items():
            if a_ not in bfs.my_atom_indices:
                continue
            P_Mi = (kpt.P_aMi[a_] if P_aqMi is None
                    else P_aqMi[a_][kpt.q])
            g_MM += P_Mi.conj() @ dH1_sii[kpt.s] @ P_Mi.T

        if a in bfs.my_atom_indices:
            P_Mi = (kpt.P_aMi[a] if P_aqMi is None
                    else P_aqMi[a][kpt.q])
            dP_Mi = dPdR_aqvMi[a][kpt.q][v]
            P1HP_MM = dP_Mi.conj() @ dH_aii[a][kpt.s] @ P_Mi.T
            g_MM += P1HP_MM + P1HP_MM.T.conjugate()
        return g_MM

    def calculate_supercell_matrix(
        self, calcdict: dict, fd_name: str = "elph",
        filter: str = None
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

        wfs, dft, timer = self._create_gpaw_calculator(calcdict, fd_name)

        # JSON cache
        supercell_cache = MultiFileJSONCache(self.supercell_name)

        # REFACTOR LATER: remove these when not needed anymore
        kd = dft.kd
        nao = wfs.setups.nao
        nspins = wfs.nspins

        # REFACTOR LATER: move into loop and load only what is needed to save memory
        # Calculate finite-difference gradients (in Hartree / Bohr)
        V1t_xsG, dH1_xasp = self.calculate_gradient(fd_name, self.indices)

        # New GPAW stores Vt_sG with full shape independent of PBC;
        # truncate along non-periodic axes where sizes disagree.
        # REFACTOR LATER: this might not be needed at the end. Check before merge
        pb = (~dft.grid.pbc_c) & (V1t_xsG.shape[-3:] != dft.grid.size_c)
        if pb.any():
            slices = tuple(slice(1, None) if p else slice(None) for p in pb)
            V1t_xsG = V1t_xsG[..., slices[0], slices[1], slices[2]]

        # Equilibrium atomic Hamiltonian matrix (projector coefficients)
        fd_cache = MultiFileJSONCache(fd_name)
        assert tuple(self.supercell) == tuple(fd_cache["info"]["supercell"])
        dH_aii = _dH_asp_to_aii(fd_cache["eq"]["dH_all_asp"], dft.dH_asii_template)

        # Save basis information, after we checked the data is kosher
        with supercell_cache.lock("basis") as handle:
            if handle is not None:
                basis_info = self.set_basis_info(wfs)
                handle.save(basis_info)

        # Fourier filter the atomic gradients of the effective potential
        if filter is not None:
            for s in range(nspins):
                fourier_filter(self.atoms, self.supercell, V1t_xsG[:, s],
                               components=filter)

        if kd.gamma:
            dft.log('WARNING: Gamma-point calculation. '
                    'Overlap with neighboring cell cannot be removed')

        # Calculate < i k | grad H | j k >, i.e. matrix elements in LCAO basis

        # Do each cartesian component separately
        # NOTE: need to keep the gastly loop and xoutput naming scheme for backward compatibiluty of the caches
        for i, a in enumerate(self.indices):
            for v in range(3):
                # Corresponding array index
                xoutput = 3 * a + v
                xinput = 3 * i + v

                # If exist already, don't recompute
                with supercell_cache.lock(str(xoutput)) as handle:
                    if handle is None:
                        continue

                    dft.log(f'{["x", "y", "z"][v]}-gradient of atom {a}')

                    with timer('Supercell entry'):
                        v1t_sR = dft.grid.empty(dims=(nspins,))
                        v1t_sR.scatter_from(V1t_xsG[xinput])
                        dH1_aii = _dH_asp_to_aii(dH1_xasp[xinput],
                                                 dft.dH_asii_template)
                        g_sMM, g_s_gammaMM = self._calculate_supercell_entry(
                            a, v, v1t_sR, dH1_aii, wfs, dH_aii, dft)
                    if not (kd.gamma or kd.N_c is None):
                        g_sMM = _finalize_g_sMM(g_sMM, g_s_gammaMM, kd)

                    # Reshape to global unit cell indices
                    N = np.prod(self.supercell)
                    # Number of basis function in the primitive cell
                    assert (nao % N) == 0, "Alarm ...!"
                    nao_cell = nao // N
                    g_sNMNM = g_sMM.reshape((nspins, N, nao_cell, N, nao_cell))
                    g_sNNMM = g_sNMNM.swapaxes(2, 3).copy()
                    handle.save(g_sNNMM)

                if xinput == 0:
                    with supercell_cache.lock("info") as handle:
                        if handle is not None:
                            info = {
                                "sc_version": sc_version,
                                "natom": len(self.atoms),
                                "supercell": self.supercell,
                                "gshape": g_sNNMM.shape,
                                "gtype": g_sNNMM.dtype.name,
                            }
                            handle.save(info)

    def set_basis_info(self, *args) -> dict:
        """Store LCAO basis info for atoms in reference cell in attribute.

        Parameters
        ----------
        args: tuple
            Either a single LCAO ``wfs`` object, or the pair ``(M_a, nao_a)``
          """
        assert len(args) in (1, 2)
        if len(args) == 1:
            wfs = args[0]
            setups = wfs.setups
            bfs = wfs.basis_functions
            nao_a = [setups[a].nao for a in range(len(self.atoms))]
            M_a = [bfs.M_a[a] for a in range(len(self.atoms))]
        else:
            M_a = args[0]
            nao_a = args[1]
        return {"M_a": M_a, "nao_a": nao_a}

    @classmethod
    def calculate_gradient(cls, fd_name: str,
                           indices=None) -> tuple[ArrayND, list]:
        """Calculate gradient of effective potential and projector coeffs.

        This function loads the generated json files and calculates
        finite-difference derivatives.

        Parameters
        ----------
        fd_name: str
            name of finite difference JSON cache
        """
        cache = MultiFileJSONCache(fd_name)
        if "dr_version" not in cache["info"]:
            print("Cache created with old version. Use electronphonon.py")
            raise VersionError
        natom = cache["info"]["natom"]
        delta = cache["info"]["delta"]

        # Array and dict for finite difference derivatives
        V1t_xsG = []
        dH1_xasp = []

        if indices is None:
            indices = np.arange(natom)

        x = 0
        for a in indices:
            for v in "xyz":
                name = "%d%s" % (a, v)
                # Potential and atomic density matrix for atomic displacement
                Vtm_sG = cache[name + "-"]["Vt_sG"]
                dHm_asp = cache[name + "-"]["dH_all_asp"]
                Vtp_sG = cache[name + "+"]["Vt_sG"]
                dHp_asp = cache[name + "+"]["dH_all_asp"]

                # FD derivatives in Hartree / Bohr
                V1t_sG = (Vtp_sG - Vtm_sG) / (2 * delta / Bohr)
                V1t_xsG.append(V1t_sG)

                dH1_asp = {}
                for atom in dHm_asp.keys():
                    dH1_asp[atom] = (dHp_asp[atom] - dHm_asp[atom]) / (
                        2 * delta / Bohr
                    )
                dH1_xasp.append(dH1_asp)
                x += 1
        return np.array(V1t_xsG), dH1_xasp

    @classmethod
    def load_supercell_matrix(cls, name: str = "supercell"
                              ) -> tuple[ArrayND, dict]:
        """Load supercell matrix from cache.

        Parameters
        ----------
        name: str
            User specified name of the cache.
        """
        # TODO: load by indices?
        supercell_cache = MultiFileJSONCache(name)
        if "sc_version" not in supercell_cache["info"]:
            print("Cache created with old version. Use electronphonon.py")
            raise VersionError
        shape = supercell_cache["info"]["gshape"]
        dtype = supercell_cache["info"]["gtype"]
        natom = supercell_cache["info"]["natom"]
        nx = natom * 3
        g_xsNNMM = np.empty([nx, ] + list(shape), dtype=dtype)
        for x in range(nx):
            g_xsNNMM[x] = supercell_cache[str(x)]
        basis_info = supercell_cache["basis"]
        return g_xsNNMM, basis_info
