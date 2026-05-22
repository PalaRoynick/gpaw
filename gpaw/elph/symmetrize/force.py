import warnings

import numpy as np
from ase.utils.filecache import MultiFileJSONCache

from .atom_mapper import AtomMapper, disp_key_to_index, disp_index_to_key

_SKIP_KEYS = frozenset({'info'})


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
            F_sym += F_av @ sym.rotation_scc[s]
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
            the initial unsymmetrized array is returned and a warning is shown.
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

    def symmetrize_cache(
        self, cache: MultiFileJSONCache, out_name: str
    ) -> MultiFileJSONCache:
        """
        Write a new cache with symmetrized forces, leaving the input untouched.

        All fields in each entry are copied verbatim;
        only ``'forces'`` is replaced with the symmetrized array.  The
        ``'info'`` entry is copied as-is.

        Parameters
        ----------
        cache : MultiFileJSONCache
            Source cache produced by the GPAW elph module.
        out_name : str
            Name for the output ``MultiFileJSONCache``.

        Returns
        -------
        MultiFileJSONCache
            New cache keyed identically to *cache* with symmetrized forces.
        """
        all_keys = {k for k in cache if k not in _SKIP_KEYS}
        disp_keys = all_keys - {'eq'}
        forces = {k: np.array(cache[k]['forces']) for k in all_keys}

        sym_forces = self.symmetrize_all_displaced(disp_keys, forces)
        if 'eq' in all_keys:
            sym_forces['eq'] = self.symmetrize_equilibrium(forces['eq'])

        out = MultiFileJSONCache(out_name)

        if 'info' in cache:
            with out.lock('info') as handle:
                if handle is not None:
                    handle.save(dict(cache['info']))

        for key in all_keys:
            data = dict(cache[key])
            data['forces'] = sym_forces[key]
            with out.lock(key) as handle:
                if handle is not None:
                    handle.save(data)

        return out
