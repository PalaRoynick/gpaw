import warnings

import numpy as np

from .atom_mapper import AtomMapper, disp_key_to_index, disp_index_to_key


class ForceSymmetrizer:
    """
    Symmetrize atomic forces using crystal symmetry.

    Parameters
    ----------
    mapper : AtomMapper
        Provides ``symmetries`` (rotation_scc, atommap_sa) and ``disp_map_sd``.
    """

    def __init__(self, mapper: AtomMapper):
        self.mapper = mapper

    def symmetrize_equilibrium(self, F_av: np.ndarray) -> np.ndarray:
        """
        Average forces over all symmetry operations for an equilibrium config.

        Parameters
        ----------
        F_av : ndarray, shape (n_atoms, 3)

        Returns
        -------
        F_sym : ndarray, shape (n_atoms, 3)
            Symmetrized forces; ≈ 0 for a well-converged calculation.
        """
        sym = self.mapper.symmetries
        n_ops = len(sym.rotation_scc)
        F_sym = np.zeros_like(F_av)
        for s in range(n_ops):
            perm = sym.atommap_sa[s]        # perm[a] = b : atom a → b
            R = sym.rotation_scc[s]
            F_sym[perm] += F_av @ R         # scatter rotated forces into image slots
        F_sym /= n_ops
        return F_sym

    def symmetrize_displaced(self, disp_key: str, forces: dict) -> np.ndarray:
        """
        Symmetrize forces for a single displaced configuration.

        Parameters
        ----------
        disp_key : str
            Target displacement identifier, e.g. '0x-'.
        forces : dict[str, ndarray]
            Mapping from displacement key to force array (n_atoms, 3).
            Only the keys present in this dict are used; missing source
            displacements are silently skipped.

        Returns
        -------
        F_sym : ndarray, shape (n_atoms, 3)
            Symmetrized force array.  If no symmetry operations contribute,
            the original unsymmetrized array is returned and a warning is emitted.
        """
        sym = self.mapper.symmetries
        n_ops = len(sym.rotation_scc)
        d_tgt = disp_key_to_index(disp_key)
        F_sym = np.zeros_like(forces[disp_key])
        count = 0

        for s in range(n_ops):
            d_src = int(self.mapper.disp_map_sd[s, d_tgt])
            if d_src == -1:
                continue
            src_key = disp_index_to_key(d_src)
            if src_key not in forces:
                continue
            F_sym += forces[src_key] @ sym.rotation_scc[s]
            count += 1

        if count == 0:
            warnings.warn(
                f'No symmetry operations contributed to {disp_key!r}; '
                'returning unsymmetrized forces.'
            )
            return np.array(forces[disp_key])

        F_sym /= count
        return F_sym

    def symmetrize_all_displaced(self, disp_keys: set, forces: dict) -> dict:
        """
        Symmetrize forces for a set of displaced configurations.

        Parameters
        ----------
        disp_keys : set[str]
            Displacement identifiers to symmetrize.
        forces : dict[str, ndarray]
            Full force mapping (may contain more keys than ``disp_keys``).

        Returns
        -------
        dict[str, ndarray]
            Symmetrized force arrays keyed by displacement identifier.
        """
        return {k: self.symmetrize_displaced(k, forces) for k in disp_keys}
