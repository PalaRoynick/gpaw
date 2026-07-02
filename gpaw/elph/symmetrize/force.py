import numpy as np
from ase.utils.filecache import MultiFileJSONCache

from gpaw.new.symmetry import create_symmetries_object

_SKIP_KEYS = frozenset({'info'})


class ForceSymmetrizer:
    """
    Symmetrize atomic forces using crystal symmetry.

    Builds a :class:`gpaw.new.symmetry.Symmetries` object from ``atoms`` and
    analyses the symmetries that survive each single-atom displacement
    (``analyze_displacements``).  Equilibrium forces are symmetrized over the
    full point group (``Symmetries.symmetrize_forces``); displaced forces are
    symmetrized over the operations that survive that displacement
    (``Symmetries.symmetrize_displaced_forces``).

    Parameters
    ----------
    atoms : ase.Atoms
        The primitive / reference structure whose displacements are catalogued.
        Symmetry (and the displacement orbits) come from this cell, so pass the
        primitive cell -- not ``atoms * supercell`` -- to keep the full point
        group (a supercell drops operations and breaks phonon degeneracies).
        Forces may still be measured on a larger (supercell) atom set.
    """

    def __init__(self, atoms):
        self.atoms = atoms
        self.symmetries = create_symmetries_object(atoms)
        self.symmetries.analyze_displacements()

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
        return self.symmetries.symmetrize_forces(np.asarray(F_av))

    def symmetrize_displaced(self, disp_key: str, forces: dict) -> np.ndarray:
        """
        Symmetrize forces for a single displaced configuration.

        Parameters
        ----------
        disp_key : str
            Target displacement identifier, e.g. '0x-'.
        forces : dict[str, ndarray]
            Mapping from displacement key to force array (n_atoms, 3).

        Returns
        -------
        F_sym : ndarray, shape (n_atoms, 3)
            Symmetrized force array for ``disp_key``.
        """
        return self.symmetries.symmetrize_displaced_forces(
            [disp_key], forces)[disp_key]

    def symmetrize_all_displaced(self, disp_keys, forces: dict) -> dict:
        """
        Symmetrize forces for a set of displaced configurations.

        Parameters
        ----------
        disp_keys : iterable[str]
            Displacement identifiers to symmetrize.
        forces : dict[str, ndarray]
            Full force mapping (may contain more keys than ``disp_keys``).

        Returns
        -------
        dict[str, ndarray]
            Symmetrized force arrays keyed by displacement identifier.
        """
        return self.symmetries.symmetrize_displaced_forces(disp_keys, forces)

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
