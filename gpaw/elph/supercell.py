"""Module for electron-phonon supercell properties."""

import numpy as np
from ase import Atoms
from ase.parallel import parprint
from ase.units import Bohr
from ase.utils.filecache import MultiFileJSONCache

from gpaw.lcao.tightbinding import TightBinding
from gpaw.dft import GPAW
from gpaw.new.ase_interface import ASECalculator
from gpaw.typing import ArrayND
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.tools import tri2full

from .filter import fourier_filter

sc_version = 2
# v1: saves natom, supercell, g_sNNMM.shape and dtype
# v2: separate V, D, P, Q and R kernels; centered projector derivatives.


class VersionError(Exception):
    """Error raised for wrong cache versions."""
    pass


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

    def _create_gpaw_calculator(
            self, calcdict, fd_name='elph') -> ASECalculator:
        """Create empty LCAO calculator to give us projectors.

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
            Used to read the potential shape and spin information needed to
            build the calculator.
        """
        cache = MultiFileJSONCache(fd_name)
        pot_shape = list(np.array(cache['eq']['Vt_sG']).shape)
        assert pot_shape[0] in (1, 2), "only colinear spins"

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
                    " and cannot be overridden")

        gpaw_kwargs = {'basis': 'dzp', **calcdict, **internal}
        calc = GPAW(**gpaw_kwargs)
        calc.create_new_calculation(self.atoms * self.supercell)
        return calc

    def _calculate_supercell_entry(self, a, v, V1t_sG, dH1_asp, wfs,
                                   dH_asp, *, components=False):
        kpt_u = wfs.kpt_u
        setups = wfs.setups
        nao = setups.nao
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        nspins = wfs.nspins

        # Array for different k-point components
        shape = (nspins, len(kpt_u) // nspins, nao, nao)
        kernels = {key: np.zeros(shape, dtype) for key in ('V', 'D', 'P',
                                                           'Q', 'R')}
        # Same ordering and center as ASE Displacement.define_offset().
        offset = np.ravel_multi_index(np.asarray(self.supercell) // 2,
                                      self.supercell)
        displaced = offset * len(self.atoms) + a

        # 1) Integrate grid/basis products once, then form each Bloch matrix.
        # Use the cell blocks with the physical Bloch phase convention.
        # The direct legacy potential-matrix API has the opposite transpose.
        V_sxMM = [bfs.calculate_potential_matrices(V) for V in V1t_sG]
        for kpt in kpt_u:
            V_xMM = V_sxMM[kpt.s]
            geff_MM = np.array(V_xMM[0], dtype=dtype)
            if bfs.gamma and dtype == float:
                tri2full(geff_MM, 'L')
            else:
                k_c = bfs.ibzk_qc[kpt.q]
                phase = np.exp(-2j * np.pi * bfs.sdisp_xc[1:] @ k_c)
                geff_MM += 2 * np.einsum('x,xMN->MN', phase, V_xMM[1:],
                                         optimize=True)
                geff_MM = .5 * (geff_MM + geff_MM.conj().T)
            kernels['V'][kpt.s, kpt.q] = geff_MM

        # 2) Gradient of non-local part (projectors)
        P_aqMi = getattr(wfs, 'P_aqMi', None)
        # 2a) dH^a part has contributions from all atoms
        for kpt in kpt_u:
            # Matrix elements
            gp_MM = np.zeros((nao, nao), dtype)
            for a_, dH1_sp in dH1_asp.items():
                if a_ not in bfs.my_atom_indices:
                    continue
                dH1_ii = unpack_hermitian(dH1_sp[kpt.s])
                if P_aqMi is None:
                    P_Mi = kpt.P_aMi[a_]
                else:
                    P_Mi = P_aqMi[a_][kpt.q]
                gp_MM += P_Mi.conj() @ dH1_ii @ P_Mi.T
            # wfs.gd.comm.sum(gp_MM)
            kernels['D'][kpt.s, kpt.q] += gp_MM

        # 2b) dP^a part has only contributions from the same atoms
        # For the contribution from the derivative of the projectors
        manytci = wfs.manytci
        local = [displaced] if displaced in bfs.my_atom_indices else []
        dPdR_aqvMi = manytci.P_aqMi(local, derivative=True)
        for kpt in kpt_u:
            if displaced in bfs.my_atom_indices:
                if P_aqMi is None:
                    P_Mi = kpt.P_aMi[displaced]
                else:
                    P_Mi = P_aqMi[displaced][kpt.q]
                dP_Mi = dPdR_aqvMi[displaced][kpt.q][v]
                dH_ii = unpack_hermitian(dH_asp[displaced][kpt.s])
                P1HP_MM = dP_Mi.conj() @ dH_ii @ P_Mi.T
                kernels['P'][kpt.s, kpt.q] = P1HP_MM + P1HP_MM.conj().T
                setup = setups[displaced]
                kernels['Q'][kpt.s, kpt.q] = (
                    dP_Mi.conj() @ setup.dO_ii @ P_Mi.T)
                # Nuclear R of Eq. (73) is minus the spatial nabla_iiv.
                kernels['R'][kpt.s, kpt.q] = (
                    -P_Mi.conj() @ setup.nabla_iiv[:, :, v] @ P_Mi.T)

        if components:
            return kernels
        return sum(kernels[key] for key in ('V', 'D', 'P'))

    def calculate_supercell_matrix(
        self, calc: ASECalculator | dict, fd_name: str = "elph",
        filter: str = None
    ) -> None:
        """Calculate matrix elements of the el-ph coupling in the LCAO basis.

        This function calculates the matrix elements between LCAOs and local
        atomic gradients of the effective potential. The matrix elements are
        calculated for the supercell used to obtain finite-difference
        approximations to the derivatives of the effective potential wrt to
        atomic displacements.

        The resulting g_xsNNMM is saved into a JSON cache.

        Parameters
        ----------
        calc: GPAW
            LCAO calculator for the calculation of the supercell matrix.
        fd_name: str
            User specified name of the finite difference JSON cache.
            Default is 'elph'.
        filter: str
            Fourier filter atomic gradients of the effective potential. The
            specified components (``normal`` or ``umklapp``) are removed
            (default: None).
        """

        atoms_N = self.atoms * self.supercell
        if isinstance(calc, dict):
            calc = self._create_gpaw_calculator(calc, fd_name)
        elif not getattr(calc, 'initialized', False):
            calc.initialize(atoms_N)
            calc.initialize_positions(atoms_N)
        assert calc.wfs.mode == 'lcao', 'LCAO mode required.'
        assert not calc.symmetry.point_group, (
            'Point group symmetry unsupported.')
        supercell_cache = MultiFileJSONCache(self.supercell_name)

        # Extract useful objects from the calculator
        wfs = calc.wfs
        gd = calc.wfs.gd
        kd = calc.wfs.kd
        bd = calc.wfs.bd
        nao = wfs.setups.nao
        nspins = wfs.nspins
        # FIXME: Domain parallelisation broken
        assert gd.comm.size == 1
        # FIXME: Band parallelisation broken - M is band parallel
        assert bd.comm.size == 1

        # Equilibrium atomic Hamiltonian matrix (projector coefficients)
        fd_cache = MultiFileJSONCache(fd_name)
        fd_info = fd_cache['info']
        if (tuple(fd_info['supercell']) != tuple(self.supercell) or
                fd_info['natom'] != len(self.atoms)):
            raise ValueError('Finite-displacement and projection supercells '
                             'must match, including the primitive atom count.')
        for key, value in [('cell', self.atoms.cell.array),
                           ('numbers', self.atoms.numbers),
                           ('positions', self.atoms.positions)]:
            if key in fd_info and not np.allclose(fd_info[key], value):
                raise ValueError(
                    'Finite-displacement geometry mismatch: ' + key)
        dH_asp = fd_cache["eq"]["dH_all_asp"]
        if 'info' in supercell_cache and (
                supercell_cache['info'].get('sc_version') != sc_version):
            raise VersionError('Use a new output cache for PAW kernels.')
        if not np.allclose(gd.cell_cv, atoms_N.cell.array / Bohr):
            raise ValueError('Projection calculator cell does not match '
                             'the finite-displacement supercell.')

        # Calculate finite-difference gradients (in Hartree / Bohr)
        V1t_xsG, dH1_xasp = self.calculate_gradient(fd_name, self.indices)
        # Preserve New GPAW's non-periodic coarse-grid compatibility.
        pb = (~gd.pbc_c) & (V1t_xsG.shape[-3:] != gd.n_c)
        if pb.any():
            slices = tuple(slice(1, None) if p else slice(None) for p in pb)
            V1t_xsG = V1t_xsG[..., slices[0], slices[1], slices[2]]

        # Check that the grid is the same as in the calculator
        assert np.all(V1t_xsG.shape[-3:] == (gd.N_c + gd.pbc_c - 1)), \
            "Mismatch in grids."

        # Save basis information, after we checked the data is kosher
        with supercell_cache.lock("basis") as handle:
            if handle is not None:
                basis_info = self.set_basis_info(calc)
                handle.save(basis_info)

        # Fourier filter the atomic gradients of the effective potential
        if filter is not None:
            for s in range(nspins):
                fourier_filter(self.atoms, self.supercell, V1t_xsG[:, s],
                               components=filter)

        if kd.gamma:
            print("WARNING: Gamma-point calculation. \
                   Overlap with neighboring cell cannot be removed")
        else:
            # Bloch to real-space converter
            tb = TightBinding(atoms_N, calc)

        # Calculate < i k | grad H | j k >, i.e. matrix elements in LCAO basis

        # Do each cartesian component separately
        for i, a in enumerate(self.indices):
            for v in range(3):
                # Corresponding array index
                xoutput = 3 * a + v
                xinput = 3 * i + v

                # If exist already, don't recompute
                with supercell_cache.lock(str(xoutput)) as handle:
                    if handle is None:
                        continue

                    parprint("%s-gradient of atom %u" %
                             (["x", "y", "z"][v], a))

                    kernels = self._calculate_supercell_entry(
                        a, v, V1t_xsG[xinput], dH1_xasp[xinput], wfs, dH_asp,
                        components=True
                    )

                    # Extract R_c=(0, 0, 0) block by Fourier transforming
                    N = np.prod(self.supercell)
                    assert (nao % N) == 0, "Alarm ...!"
                    nao_cell = nao // N
                    entry = {}
                    for key, g_sqMM in kernels.items():
                        if kd.gamma or kd.N_c is None:
                            g_sMM = g_sqMM[:, 0]
                        else:
                            g_sMM = np.array([
                                tb.bloch_to_real_space(g_sqMM[s],
                                                       R_c=(0, 0, 0))[0]
                                for s in range(nspins)])
                        g_sNMNM = g_sMM.reshape(nspins, N, nao_cell,
                                                N, nao_cell)
                        entry[key] = g_sNMNM.swapaxes(2, 3).copy()
                    g_sNNMM = entry['V']
                    handle.save(entry)
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
            If the LCAO calculator is not available (e.g. if the supercell is
            loaded from file), the ``load_supercell_matrix`` member function
            provides the required info as arguments.

        """
        assert len(args) in (1, 2)
        if len(args) == 1:
            calc = args[0]
            setups = calc.wfs.setups
            bfs = calc.wfs.basis_functions
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
        """Load the pseudo-Hamiltonian derivative V+D+P from cache.

        Energy-dependent PAW corrections require ElectronPhononMatrix.

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
            entry = supercell_cache[str(x)]
            if isinstance(entry, dict):
                entry = sum(entry[key] for key in ('V', 'D', 'P'))
            g_xsNNMM[x] = entry
        basis_info = supercell_cache["basis"]
        return g_xsNNMM, basis_info
