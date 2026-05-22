import re
from functools import cached_property

import numpy as np

_AXIS_CHARS = ['x', 'y', 'z']
_SIGN_CHARS = ['+', '-']
_DISP_KEY_RE = re.compile(r'^(\d+)([xyz])([+-])$')


def encode_disp(atom: int, axis: int, sign_idx: int) -> int:
    """Encode (atom, axis, sign_idx) → flat displacement index
        d = atom*6 + axis*2 + sign_idx."""
    return atom * 6 + axis * 2 + sign_idx


def decode_disp(d: int) -> tuple:
    """Decode flat displacement index → (atom, axis, sign_idx)."""
    atom, rem = divmod(d, 6)
    axis, sign_idx = divmod(rem, 2)
    return atom, axis, sign_idx


def disp_key_to_index(key: str) -> int:
    """Convert cache key like '0x+' → flat displacement index."""
    m = _DISP_KEY_RE.match(key)
    if not m:
        raise ValueError(f'Invalid displacement key: {key!r}')
    atom, axis_char, sign_char = m.groups()
    axis = _AXIS_CHARS.index(axis_char)
    sign_idx = 0 if sign_char == '+' else 1
    return encode_disp(int(atom), axis, sign_idx)


def disp_index_to_key(d: int) -> str:
    """Convert flat displacement index → cache key like '0x+'."""
    atom, axis, sign_idx = decode_disp(d)
    return f'{atom}{_AXIS_CHARS[axis]}{_SIGN_CHARS[sign_idx]}'


class AtomMapper:
    """
    Build the displaced-displacement perm table from a Symmetries object.

    Parameters
    ----------
    symmetries
        A gpaw.new.symmetry.Symmetries instance (or any object exposing
        ``rotation_scc`` shape (n_ops, 3, 3) and
        ``atommap_sa`` shape (n_ops, n_atoms)).

    Attributes
    ----------
    disp_map_sd : ndarray, shape (n_ops, n_atoms * 6), dtype int
        ``disp_map_sd[s, d_target] = d_src`` — symmetry operation *s* maps
        source displacement *d_src* onto target displacement *d_target*.
        Sentinel −1 means no valid cardinal-axis image exists for that entry
        (only occurs for non-symmorphic operations; should not appear in
        practice for the symmorphic operations exposed by ``Symmetries``).
    """

    def __init__(self, symmetries):
        self.symmetries = symmetries

    @cached_property
    def disp_map_sd(self) -> np.ndarray:
        rotation_scc = self.symmetries.rotation_scc
        atommap_sa = self.symmetries.atommap_sa
        n_ops, n_atoms = atommap_sa.shape
        disp_map = np.full((n_ops, n_atoms * 6), -1, dtype=int)

        for s in range(n_ops):
            R = rotation_scc[s]
            for j in range(n_atoms):
                i = int(atommap_sa[s, j])
                for axis in range(3):
                    for si, sign in enumerate([+1, -1]):
                        e = np.zeros(3, dtype=int)
                        e[axis] = sign
                        e_new = e @ R   # row-vector convention: x' = x @ R
                        nz = np.nonzero(e_new)[0]
                        if len(nz) != 1:
                            continue    # not a cardinal axis; skip
                        new_axis = int(nz[0])
                        new_si = 0 if e_new[new_axis] > 0 else 1
                        d_src = encode_disp(j, axis, si)
                        d_tgt = encode_disp(i, new_axis, new_si)
                        disp_map[s, d_tgt] = d_src

        return disp_map
