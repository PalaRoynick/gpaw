r"""Module for calculating electron-phonon matrix.

Electron-phonon interaction::

                  __
                  \     l   +         +
        H      =   )   g   c   c   ( a   + a  ),
         el-ph    /_    ij  i   j     l     l
                 l,ij

where the electron phonon coupling is given by::

                      ________
             l       /  hbar          ___
            g   =   /---------  < i | \ /  V   * e  | j > .
             ij   \/ 2 M   w           'u   eff   l
                        ref l

Here, l denotes the vibrational mode, w_l and e_l is the frequency and
mass-scaled polarization vector, respectively, M_ref is a reference mass
(in this code, the reference mass is 1 amu to be consistent with the
mass-scaled phonon eigenvectors from ASE),
i and j are electronic state indices and nabla_u denotes the gradient wrt
atomic displacements. The implementation supports calculations of the el-ph
coupling in both finite and periodic systems, i.e. expressed in a basis of
molecular orbitals or Bloch states.
"""
import ase.units as units
import numpy as np
from ase import Atoms
from ase.phonons import Phonons
from ase.utils.filecache import MultiFileJSONCache
from ase.utils.timing import Timer, timer

from gpaw.elph.supercell import Supercell
from gpaw.mpi import normalize_communicator
from gpaw.old.calculator import GPAW
from gpaw.typing import ArrayND

OPTIMIZE = 'optimal'


class ElectronPhononMatrix:
    """Class for containing the electron-phonon matrix"""

    def __init__(self,
                 atoms: Atoms,
                 supercell_cache: str,
                 phonon: Phonons | str | dict,
                 *,
                 indices: list[int] | None = None,
                 load_sc_as_needed: bool = True,
                 use_mean_minimum_images: bool = False,
                 comm=None):
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
        """
        # XXX Since no parallelization has been implemented for this class,
        #     should the communicator input be removed?
        self.comm = normalize_communicator(comm)
        self.use_mean_minimum_images = use_mean_minimum_images

        if not load_sc_as_needed:
            assert indices is None, 'Use "load_sc_as_needed" with "indices"'

        self.timer = Timer()

        self.atoms = atoms
        if indices is None:
            indices = list(range(len(atoms)))
        self.indices = indices

        self.timer.start('Reading caches')

        self._set_phonon_cache(phonon, atoms)
        self.lattice_vectors_array = self.phonon.lattice_vectors_array
        if set(self.phonon.indices) != set(self.indices):
            raise ValueError('Atom indices for ElectronPhononMatrix do not '
                             'match atom indices for the Phonons class.\n')
        self._read_phonon_cache()

        self._set_supercell_cache(supercell_cache, load_sc_as_needed)

        self.timer.stop('Reading caches')

    def _set_phonon_cache(self, phonon, atoms):
        if isinstance(phonon, Phonons):
            self.phonon = phonon
        elif isinstance(phonon, str):
            info = MultiFileJSONCache(phonon, comm=self.comm)['info']
            assert 'dr_version' in info, 'use valid cache created by elph'
            assert info['dr_version'] == 2, ('Displacements cache is outdated,'
                                             ' please recalculate!')
            self.phonon = Phonons(
                atoms,
                supercell=tuple(info['supercell']),
                name=phonon,
                select_atoms=info['indices'].tolist(),
                delta=info['delta'],
                center_refcell=True,
                use_mean_minimum_images=self.use_mean_minimum_images,
            )
        elif isinstance(phonon, dict):
            # this would need to be updated if Phonon defaults change
            assert phonon.get('center_refcell', True)
            self.phonon = Phonons(
                atoms, **phonon,
                use_mean_minimum_images=self.use_mean_minimum_images)
        else:
            raise TypeError('Bad input type for keyword "phonon".')

    def _set_supercell_cache(self, supercell_cache: str,
                             load_sc_as_needed: bool):
        self.supercell_cache = MultiFileJSONCache(
            supercell_cache, comm=self.comm)

        if load_sc_as_needed:
            self._yield_g_NNMM = self._yield_g_NNMM_as_needed
            self.g_xNNMM = None
        else:
            self.g_xsNNMM, _ = Supercell.load_supercell_matrix(supercell_cache)
            self._yield_g_NNMM = self._yield_g_NNMM_from_var

    def _read_phonon_cache(self):
        if self.phonon.C_avNav is None:
            self.phonon.read(symmetrize=10)

    def _yield_g_NNMM_as_needed(self, x, s):
        return self.supercell_cache[str(x)][s]

    def _yield_g_NNMM_from_var(self, x, s):
        return self.g_xsNNMM[x, s]

    def _get_lattice_vectors(self):
        """Recover lattice vectors of elph calculation"""
        supercell = tuple(self.supercell_cache["info"]["supercell"])
        ph = Phonons(self.atoms, supercell=supercell, center_refcell=True)
        return ph.compute_lattice_vectors()

    @classmethod
    def _gather_all_wfc(cls, wfs, s):
        """Return complete wave function on rank 0"""
        c_knM = np.zeros((wfs.kd.nbzkpts, wfs.bd.nbands, wfs.setups.nao),
                         dtype=complex)
        for k in range(wfs.kd.nbzkpts):
            for n in range(wfs.bd.nbands):
                c_knM[k, n] = wfs.get_wave_function_array(n, k, s, False)
        return c_knM

    @timer("Bloch matrix q k")
    def _bloch_matrix(self, var1: ArrayND, C2_nM: ArrayND,
                      k_c: ArrayND, q_c: ArrayND,
                      prefactor: bool, s: int | None = None) -> ArrayND:
        """Calculates elph matrix entry for a given k and q.

        The first argument must either be
        C1_nM, the ket wavefunction at k_c
        OR
        or a preprocessed g_xNMn, where the ket side was taken care of.
        """

        R_Nc = self.lattice_vectors_array

        if var1.ndim == 2:
            C1_nM = var1
            precalc = False
            assert s is not None
        elif var1.ndim == 4:
            g_xNMn = var1
            precalc = True
        else:
            raise ValueError("var1 must be C1_nM or g_xNMn")
        omega_ql, u_ql = self.phonon.band_structure(q_c[np.newaxis, :],
                                                    modes=True)
        u_l = u_ql[0]
        assert len(u_l.shape) == 3

        # Defining system sizes
        nmodes = u_l.shape[0]
        nbands = C2_nM.shape[0]
        nao = C2_nM.shape[1]
        ndisp = 3 * len(self.indices)

        # Allocate array for couplings
        g_lnn = np.zeros((nmodes, nbands, nbands), dtype=complex)

        # Mass scaled polarization vectors
        u_lx = u_l.reshape(nmodes, ndisp)

        # Multiply phase factors
        phase_m = np.exp(2.0j * np.pi * R_Nc @ (k_c + q_c))

        if not precalc:
            phase_n = np.exp(-2.0j * np.pi * R_Nc @ k_c)

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
            # potential BUG: M needs to be unit cell mass according to
            # some sources
            amu = units._amu  # atomic mass unit
            me = units._me  # electron mass
            g_lnn /= np.sqrt(2 * amu / me / units.Hartree *
                             omega_ql[0, :, np.newaxis, np.newaxis])
            # Convert to eV
            return g_lnn * units.Hartree  # eV
        else:
            return g_lnn * units.Hartree / units.Bohr  # eV / Ang

    @timer('g ket part')
    def _precalculate_ket(self, g_xNkMn, c_knM, bzk_kc, s: int):
        phase_kN = np.exp(-2.0j * np.pi * np.einsum(
            'ki,ni->kn', bzk_kc, self.lattice_vectors_array))

        for i, a in enumerate(self.indices):
            for v in range(3):
                x_cache = 3 * a + v
                x_index = 3 * i + v
                g_NNMM = self._yield_g_NNMM_as_needed(x_cache, s)
                g_NkMM = np.einsum('mnop,kn->mkop', g_NNMM, phase_kN,
                                   optimize=OPTIMIZE)
                g_xNkMn[x_index] = np.einsum('mkop,knp->mkon', g_NkMM, c_knM,
                                             optimize=OPTIMIZE)

    def bloch_matrix(self, calc: GPAW, k_qc: ArrayND = None,
                     savetofile: bool = True, prefactor: bool = True,
                     accoustic: bool = True) -> ArrayND:
        r"""Calculate el-ph coupling in the Bloch basis for the electrons.

        This function calculates the electron-phonon coupling between the
        specified Bloch states, i.e.::

                      ______
            mnl      / hbar               ^
           g    =   /-------  < m k + q | e  . grad V  | n k >
            kq    \/ 2 M w                 ql        q
                          ql

        In case the ``prefactor=False`` is given, the bare matrix
        element (in units of eV / Å) without the sqrt prefactor is returned.

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

        for s in range(wfs.nspins):
            # Collect all wfcs on rank 0
            with self.timer("Gather wavefunctions to root"):
                c_knM = self._gather_all_wfc(wfs, s)

            # Precalculate ket of g
            g_xNkMn = np.zeros(
                (3 * len(self.indices),
                 np.prod(self.supercell_cache["info"]["supercell"]),
                 wfs.kd.nbzkpts,
                 wfs.setups.nao,
                 wfs.bd.nbands),
                dtype=complex)
            self._precalculate_ket(g_xNkMn, c_knM, kd.bzk_kc, s)

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
                    g_lnn = self._bloch_matrix(g_xNkMn[:, :, k], ckplusq_nM,
                                               k_c, q_c, prefactor)
                    if np.allclose(q_c, [0.0, 0.0, 0.0]) and accoustic:
                        g_lnn[0:3] = 0.0
                    g_sqklnn[s, q, k] += g_lnn

        if self.comm.rank == 0 and savetofile:
            np.save("gsqklnn.npy", g_sqklnn)

        return g_sqklnn

    def __del__(self):
        if self.comm.rank == 0:
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
