r"""Module for calculating electron-phonon matrix.

Electron-phonon interaction::

                  __
                  \     l   +         +
        H      =   )   g   c   c   ( a   + a  ),
         el-ph    /_    ij  i   j     l     l
                 l,ij

where the electron phonon coupling is given by::

                      ______
             l       / hbar         ___
            g   =   /-------  < i | \ /  V   * e  | j > .
             ij   \/ 2 M w           'u   eff   l
                          l

Here, l denotes the vibrational mode, w_l and e_l is the frequency and
mass-scaled polarization vector, respectively, M is an effective mass, i, j are
electronic state indices and nabla_u denotes the gradient wrt atomic
displacements. The implementation supports calculations of the el-ph coupling
in both finite and periodic systems, i.e. expressed in a basis of molecular
orbitals or Bloch states.
"""
import ase.units as units
import numpy as np
from ase import Atoms
from ase.phonons import Phonons
from ase.utils.filecache import MultiFileJSONCache
from ase.utils.timing import Timer, timer

from gpaw.mpi import normalize_communicator
from gpaw.old.calculator import GPAW
from gpaw.typing import ArrayND

from .supercell import Supercell, VersionError

OPTIMIZE = "optimal"


def degenerate_average(g_lmn, modes, final, initial):
    """Return sqrt(<|g|^2>) of Eq. (97), PRB 100, 174304.

    Sum complex PAW components before calling this function. Indices select
    complete degenerate subspaces, with axes (mode, final, initial).
    """
    return np.sqrt(np.mean(abs(g_lmn[np.ix_(modes, final, initial)])**2))


class ElectronPhononMatrix:
    """Class for containing the electron-phonon matrix"""

    def __init__(self, atoms: Atoms, supercell_cache: str, phonon,
                 *, indices=None, load_sc_as_needed: bool = True,
                 use_mean_minimum_images: bool = False, comm=None,
                 paw_corrections: bool = True) -> None:
        """Initialize with base class args and kwargs.

        Parameters
        ----------
        atoms: Atoms
            Primitive cell object
        supercell_cache: str
            Name of JSON cache containing supercell matrix
        phonon: str, dict, :class:`~ase.phonons.Phonons`
            Can be either name of phonon cache generated with
            electron-phonon DisplacementRunner or dictionary
            of arguments used in Phonons run or Phonons object.
        load_sc_as_needed: bool
            Load supercell matrix elements only as needed.
            Greatly reduces memory requirement for large systems,
            but introduces huge filesystem overhead
        indices: list
            List of atoms (indices) to use. Default: Use all.
        paw_corrections: bool
            Include overlap and transformation derivatives (default True).
            False reads the pseudo-Hamiltonian derivative for diagnostics.
        """
        world = normalize_communicator(comm)
        self.comm = world
        self.use_mean_minimum_images = use_mean_minimum_images

        self.world = world
        self.paw_corrections = paw_corrections
        self._kernel = None
        if not load_sc_as_needed:
            assert indices is None, "Use 'load_sc_as_needed' with 'indices'"

        self.timer = Timer()

        self.atoms = atoms
        if indices is None:
            indices = np.arange(len(atoms))
        if isinstance(indices, np.ndarray):
            self.indices = indices.tolist()
        else:
            self.indices = list(indices)

        self._set_phonon_cache(phonon, atoms)
        self._set_supercell_cache(supercell_cache, load_sc_as_needed)

        self.timer.start("Read phonons")
        if hasattr(self, 'supercell_cache'):
            shape = tuple(self.supercell_cache['info']['supercell'])
            offset = np.ravel_multi_index(np.asarray(shape) // 2, shape)
            if (tuple(self.phonon.supercell) != shape or
                    self.phonon.refcell_offset != offset):
                raise ValueError('Phonons and PAW kernels must use the same '
                                 'supercell and centered reference cell.')

        if set(self.phonon.indices) != set(self.indices):
            raise ValueError('Atom indices for ElectronPhononMatrix must '
                             'match the Phonons indices.')

        self._read_phonon_cache()
        self.timer.stop("Read phonons")

    def _set_supercell_cache(self, supercell_cache: str,
                             load_sc_as_needed: bool):
        self.supercell_cache = MultiFileJSONCache(supercell_cache)
        self.sc_version = self.supercell_cache['info']['sc_version']
        if self.paw_corrections and self.sc_version < 2:
            raise VersionError('Regenerate the supercell cache for full PAW '
                               'couplings, or explicitly set '
                               'paw_corrections=False for legacy results.')
        self.R_cN = self.phonon.lattice_vectors_array.T

        self._paw_entries = None
        if self.paw_corrections and not load_sc_as_needed:
            self._paw_entries = {
                str(x): self.supercell_cache[str(x)]
                for x in range(3 * len(self.atoms))}

        if load_sc_as_needed or self.paw_corrections:
            self._yield_g_NNMM = self._yield_g_NNMM_as_needed
            self.g_xNNMM = None
        else:
            self.g_xsNNMM, _ = Supercell.load_supercell_matrix(supercell_cache)
            self._yield_g_NNMM = self._yield_g_NNMM_from_var

    def _set_phonon_cache(self, phonon, atoms):
        if isinstance(phonon, Phonons):
            self.phonon = phonon
        elif isinstance(phonon, str):
            info = MultiFileJSONCache(phonon, comm=self.comm)['info']
            if info.get('dr_version') not in (1, 2):
                raise VersionError('Unsupported finite-displacement cache.')
            # v1 has all atoms; read it without rewriting the source cache.
            indices = np.asarray(
                info.get('indices', range(len(atoms)))).tolist()
            self.phonon = Phonons(
                atoms, supercell=tuple(info['supercell']), name=phonon,
                select_atoms=indices, delta=info['delta'], center_refcell=True,
                use_mean_minimum_images=self.use_mean_minimum_images)
        elif isinstance(phonon, dict):
            options = dict(phonon)
            options.setdefault('center_refcell', True)
            options.setdefault('use_mean_minimum_images',
                               self.use_mean_minimum_images)
            self.phonon = Phonons(atoms, **options)
        else:
            raise TypeError('Bad input type for keyword "phonon".')

    def _read_phonon_cache(self):
        if self.phonon.C_avNav is None:
            # Use unmodified central differences and impose the ASR.
            # This also avoids the centered-cell Frederiksen indexing bug
            # in older ASE releases (fixed in the pinned development ASE).
            self.phonon.read(method='standard', symmetrize=10)

    def _yield_g_NNMM_as_needed(self, x, s):
        entries = self._paw_entries or self.supercell_cache
        entry = entries[str(x)]
        if isinstance(entry, dict):
            if self._kernel == 'Qright':
                return entry['Q'][s].transpose(1, 0, 3, 2).conj()
            if self._kernel is not None:
                return entry[self._kernel][s]
            entry = sum(entry[key] for key in ('V', 'D', 'P'))
        return entry[s]

    def _yield_g_NNMM_from_var(self, x, s):
        return self.g_xsNNMM[x, s]

    def _get_lattice_vectors(self):
        return self.phonon.lattice_vectors_array.T

    @classmethod
    def _gather_all_wfc(cls, wfs, s):
        """Return complete wave function on rank 0"""
        c_knM = np.zeros((wfs.kd.nbzkpts, wfs.bd.nbands, wfs.setups.nao),
                         dtype=complex)
        for k in range(wfs.kd.nbzkpts):
            for n in range(wfs.bd.nbands):
                c_knM[k, n] = wfs.get_wave_function_array(n, k, s, False)
        wfs.world.broadcast(c_knM, 0)
        return c_knM

    @timer("Bloch matrix q k")
    def _bloch_matrix(self, var1: ArrayND, C2_nM: ArrayND,
                      k_c: ArrayND, q_c: ArrayND,
                      prefactor: bool, s: int | None = None,
                      phonon_data=None) -> ArrayND:
        """Calculates elph matrix entry for a given k and q.

        The first argument must either be
        C1_nM, the ket wavefunction at k_c
        OR
        or a preprocessed g_xNMn, where the ket side was taken care of.
        """
        if var1.ndim == 2:
            C1_nM = var1
            precalc = False
            assert s is not None
        elif var1.ndim == 4:
            g_xNMn = var1
            precalc = True
        else:
            raise ValueError("var1 must be C1_nM or g_xNMn")
        if phonon_data is None:
            phonon_data = self.phonon.band_structure(
                np.asarray(q_c)[None], modes=True)
        omega_ql, u_ql = phonon_data
        u_l = u_ql[0]
        assert len(u_l.shape) == 3

        # Defining system sizes
        nmodes = u_l.shape[0]
        nbands = C2_nM.shape[0]
        ninitial = var1.shape[0] if not precalc else var1.shape[-1]
        nao = C2_nM.shape[1]
        ndisp = 3 * len(self.indices)

        # Allocate array for couplings
        g_lnn = np.zeros((nmodes, nbands, ninitial), dtype=complex)

        # Mass scaled polarization vectors
        u_lx = u_l.reshape(nmodes, ndisp)

        # Multiply phase factors
        phase_m = np.exp(-2.0j * np.pi * np.einsum("i,im->m", k_c + q_c,
                                                   self.R_cN))
        if not precalc:
            phase_n = np.exp(2.0j * np.pi * np.einsum("i,in->n", k_c,
                                                      self.R_cN))

        # Do each cartesian component separately
        for i, a in enumerate(self.indices):
            for v in range(3):
                xinput = 3 * a + v
                xoutput = 3 * i + v
                if not precalc:
                    g_NNMM = self._yield_g_NNMM(xinput, s)
                    assert nao == g_NNMM.shape[-1]
                    # some of these things take a long time. make it fast
                    with self.timer("g_MM"):
                        g_MM = np.einsum("mnop,m,n->op", g_NNMM, phase_m,
                                         phase_n, optimize=OPTIMIZE)
                        assert g_MM.shape[0] == g_MM.shape[1]
                    with self.timer("g_nn"):
                        g_nn = np.dot(C2_nM.conj(), np.dot(g_MM, C1_nM.T))
                        # g_nn = np.einsum('no,op,mp->nm', C2_nM.conj(),
                        #                   g_MM, C1_nM)
                else:
                    with self.timer("g_Mn"):
                        g_Mn = np.einsum("mon,m->on", g_xNMn[xoutput], phase_m)
                    with self.timer("g_nn"):
                        g_nn = np.dot(C2_nM.conj(), g_Mn)
                with self.timer("g_lnn"):
                    # g_lnn += np.einsum('i,kl->ikl', u_lx[:, x], g_nn,
                    #                   optimize=OPTIMIZE)
                    g_lnn += np.multiply.outer(u_lx[:, xoutput], g_nn)

        # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
        if prefactor:
            # ASE polarization vectors already contain 1/sqrt(atomic mass).
            amu = units._amu  # atomic mass unit
            me = units._me  # electron mass
            g_lnn /= np.sqrt(2 * amu / me / units.Hartree *
                             omega_ql[0, :, np.newaxis, np.newaxis])
            # Convert to eV
            return g_lnn * units.Hartree  # eV
        else:
            return g_lnn * units.Hartree / units.Bohr  # eV / Ang

    @timer("g ket part")
    def _precalculate_ket(self, c_knM, kd, s: int):
        g_xNkMn = []
        phase_kn = np.exp(2.0j * np.pi * np.einsum("ki,in->kn", kd.bzk_kc,
                                                   self.R_cN))
        for a in self.indices:
            for v in range(3):
                x = 3 * a + v
                g_NNMM = self._yield_g_NNMM(x, s)
                g_NkMM = np.einsum("mnop,kn->mkop", g_NNMM, phase_kn,
                                   optimize=OPTIMIZE)
                g_NkMn = np.einsum("mkop,knp->mkon", g_NkMM, c_knM,
                                   optimize=OPTIMIZE)
                g_xNkMn.append(g_NkMn)
        return np.array(g_xNkMn)

    @staticmethod
    def _assemble_paw(kernels, eps1_n, eps2_n):
        """Eqs. (75)-(79), PRB 100, 174304; energies in Hartree.

        Axis order is (mode, final band, initial band). Q is the
        left-projector derivative; R is the nuclear derivative integral.
        """
        ei = np.asarray(eps1_n)[None, None, :]
        ef = np.asarray(eps2_n)[None, :, None]
        result = {key: kernels[key] for key in ('V', 'D')}
        result['P'] = (kernels['P'] - ei * kernels['Q']
                       - ef * kernels['Qright'])
        result['R'] = -(ef - ei) * kernels['R']
        result['total'] = sum(result.values())
        return result

    def bloch_matrix_pair(self, C1_nM, C2_nM, eps1_n, eps2_n,
                          k_c, q_c, *, spin=0, prefactor=True,
                          accoustic=True):
        """Return V, D, P, R and total for one pair of Bloch points.

        Coefficients and eigenvalues must use the same LCAO basis as the
        projection cache. Energies are absolute eigenvalues in Hartree,
        not Fermi-level shifted. The result has axes (mode, final, initial)
        and units eV (prefactor=True). With prefactor=False, the result is
        in eV/Angstrom/sqrt(amu), because ASE modes are mass weighted.
        This entry point also supports arbitrary path points without a
        full commensurate electronic mesh.
        """
        if not self.paw_corrections or self.sc_version < 2:
            raise VersionError('bloch_matrix_pair requires PAW cache v2.')
        kernels = {}
        phonon_data = self.phonon.band_structure(
            np.asarray(q_c)[None], modes=True)
        try:
            for key in ('V', 'D', 'P', 'Q', 'Qright', 'R'):
                self._kernel = key
                kernels[key] = self._bloch_matrix(
                    C1_nM, C2_nM, np.asarray(k_c), np.asarray(q_c),
                    prefactor=False, s=spin, phonon_data=phonon_data)
        finally:
            self._kernel = None
        result = self._assemble_paw(kernels, eps1_n, eps2_n)
        omega = phonon_data[0][0]
        acoustic = np.zeros(len(omega), dtype=bool)
        if accoustic and np.allclose(q_c, 0):
            acoustic[:3] = True
        if prefactor:
            if np.any(omega[~acoustic] <= 0):
                raise ValueError('Non-positive phonon energy away from the '
                                 'Gamma translations; check convergence.')
            factor = np.zeros(len(omega))
            factor[~acoustic] = units.Bohr / np.sqrt(
                2 * units._amu / units._me / units.Hartree * omega[~acoustic])
            result = {key: value * factor[:, None, None]
                      for key, value in result.items()}
        else:
            for value in result.values():
                value[acoustic] = 0
        return result

    def bloch_matrix(self, calc: GPAW, k_qc: ArrayND = None,
                     savetofile: bool = True, prefactor: bool = True,
                     accoustic: bool = True, *,
                     return_components: bool = False):
        r"""Calculate el-ph coupling in the Bloch basis for the electrons.

        This function calculates the electron-phonon coupling between the
        specified Bloch states, i.e.::

                      ______
            mnl      / hbar               ^
           g    =   /-------  < m k + q | e  . grad V  | n k >
            kq    \/ 2 M w                 ql        q
                          ql

        With ``prefactor=False`` the mass-weighted derivative is returned
        in eV/Angstrom/sqrt(amu). ASE modes already contain the atomic
        mass factor; do not include it again.

        Parameters
        ----------
        calc: GPAW
            Converged calculator object containing the LCAO wavefuntions
            (don't use point group symmetry)
        k_qc: np.ndarray
            q-vectors of the phonons. Must only contain values commensurate
            with k-point sampling of calculator. Default: all kpoints used.
        savetofile: bool
            If true (default), saves matrix to gsqklnn.npy
        prefactor: bool
            if false, don't multiply with sqrt prefactor (Default: True)
        accoustic: bool
            if True, for 3 accoustic modes set g=0 for q=0 (Default: True)
        """
        kd = calc.wfs.kd
        assert kd.nbzkpts == kd.nibzkpts, "Elph matrix requires FULL BZ"

        wfs = calc.wfs
        if k_qc is None:
            k_qc = kd.get_bz_q_points(first=True)
        elif not isinstance(k_qc, np.ndarray):
            k_qc = np.array(k_qc)
        assert k_qc.ndim == 2

        g_sqklnn = np.zeros([wfs.nspins, k_qc.shape[0], kd.nbzkpts,
                             3 * len(self.indices), wfs.bd.nbands,
                             wfs.bd.nbands], dtype=complex)
        if return_components:
            if not self.paw_corrections:
                raise ValueError('Components require paw_corrections=True.')
            output = {key: np.zeros_like(g_sqklnn)
                      for key in ('V', 'D', 'P', 'R', 'total')}

        for s in range(wfs.nspins):
            # Collect all wfcs on rank 0
            with self.timer("Gather wavefunctions to root"):
                c_knM = self._gather_all_wfc(wfs, s)

            # precalculate k (ket) of g
            if self.paw_corrections:
                eps_kn = np.array([calc.get_eigenvalues(kpt=k, spin=s)
                                   for k in range(kd.nbzkpts)]) / units.Hartree
            else:
                g_xNkMn = self._precalculate_ket(c_knM, kd, s)

            for q, q_c in enumerate(k_qc):
                # Find indices of k+q for the k-points
                kplusq_k = kd.find_k_plus_q(q_c)  # works on FBZ
                # Note: calculations require use of FULL BZ,
                # so NO symmetry
                print("Spin {}/{}; q-point {}/{}".format(
                    s + 1, wfs.nspins, q + 1, len(k_qc)))

                for k in range(kd.nbzkpts):
                    k_c = kd.bzk_kc[k]
                    kplusq_c = (k_c + q_c) % 1.
                    kplusq_c -= kplusq_c.round()
                    # print(kplusq_c, kd.bzk_kc[kplusq_k[k]])
                    assert np.allclose(kplusq_c, kd.bzk_kc[kplusq_k[k]])
                    ckplusq_nM = c_knM[kplusq_k[k]]
                    if self.paw_corrections:
                        parts = self.bloch_matrix_pair(
                            c_knM[k], ckplusq_nM, eps_kn[k],
                            eps_kn[kplusq_k[k]], k_c, q_c, spin=s,
                            prefactor=prefactor, accoustic=accoustic)
                        g_lnn = parts['total']
                        if return_components:
                            for key in output:
                                output[key][s, q, k] = parts[key]
                    else:
                        g_lnn = self._bloch_matrix(
                            g_xNkMn[:, :, k], ckplusq_nM, k_c, q_c, prefactor)
                    if np.allclose(q_c, [0.0, 0.0, 0.0]) and accoustic:
                        g_lnn[0:3] = 0.0
                    g_sqklnn[s, q, k] += g_lnn

        if self.world.rank == 0 and savetofile:
            if return_components:
                np.savez('gsqklnn-components.npz', **output)
            else:
                np.save("gsqklnn.npy", g_sqklnn)

        return output if return_components else g_sqklnn

    def __del__(self):
        if self.world.rank == 0:
            try:
                self.timer.write()
            except ValueError:
                pass


#   def lcao_matrix(self, u_l, omega_l):
#         """Calculate the el-ph coupling in the electronic LCAO basis.

#         For now, only works for Gamma-point phonons.

#         This method is not tested.

#         Parameters
#         ----------
#         u_l: ndarray
#             Mass-scaled polarization vectors (in units of 1 / sqrt(amu)) of
#             the phonons.
#         omega_l: ndarray
#             Vibrational frequencies in eV.
#         """

#         # Supercell matrix (Hartree / Bohr)
#         assert self.g_xsNNMM is not None, "Load supercell matrix."
#         assert self.g_xsNNMM.shape[2:4] == (1, 1)
#         g_xsMM = self.g_xsNNMM[:, :, 0, 0, :, :]
#         # Number of atomic orbitals
#         # nao = g_xMM.shape[-1]
#         # Number of phonon modes
#         nmodes = u_l.shape[0]

#         #
#         u_lx = u_l.reshape(nmodes, 3 * len(self.atoms))
#         # np.dot uses second to last index of second array
#         g_lsMM = np.dot(u_lx, g_xsMM.transpose(2, 0, 1, 3))

#         # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
#         amu = units._amu  # atomic mass unit
#         me = units._me   # electron mass
#         g_lsMM /= np.sqrt(2 * amu / me / units.Hartree *
#                           omega_l[:, :, np.newaxis, np.newaxis])
#         # Convert to eV
#         g_lsMM *= units.Hartree

#         return g_lsMM
