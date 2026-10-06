"""Tiled storage of LCAO basis functions for BLAS-3 grid kernels.

The grid is split into tiles of ``TILE`` points.  For each tile, the basis
functions touching it are stored as a dense ``npts x nf`` matrix (zero where
a function is outside its cutoff sphere).  This lets the potential matrix and
the density be computed with one small BLAS-3 call per tile, instead of the
point-by-point loops of the LFC kernels (see c/lfc_tiled.cpp).
"""
from __future__ import annotations

import os

import numpy as np

import gpaw.cgpaw as cgpaw

TILE = (4, 4, 8)


def tiles_available() -> bool:
    return (hasattr(cgpaw, 'tiled_potential_matrix') and
            os.environ.get('GPAW_LFC_TILES', '1') != '0')


class TiledFunctions:
    def __init__(self, A_Wgm, M_W, G_B, W_B, n_c, nM, tile=TILE):
        n_c = np.asarray(n_c)
        tile_c = np.asarray(tile)
        ntile_c = -(-n_c // tile_c)
        npts = int(np.prod(tile_c))
        nm_W = np.array([A_gm.shape[1] for A_gm in A_Wgm], dtype=np.int64)

        # Grid points of each volume: its (enter, leave) boundary pairs,
        # in the same order as the rows of A_gm.
        enter = W_B >= 0
        eW = W_B[enter]
        eG = G_B[enter]
        lW = -1 - W_B[~enter]
        lG = G_B[~enter]
        e = np.lexsort((eG, eW))
        l = np.lexsort((lG, lW))
        W_i = eW[e].astype(np.int64)
        G1_i = eG[e].astype(np.int64)
        len_i = lG[l].astype(np.int64) - G1_i
        assert (lW[l] == W_i).all() and (len_i >= 0).all()
        npoints = int(len_i.sum())
        assert npoints == sum(len(A_gm) for A_gm in A_Wgm)
        G_g = (np.repeat(G1_i - np.cumsum(len_i) + len_i, len_i) +
               np.arange(npoints))
        W_g = np.repeat(W_i, len_i)

        # Tile and point-in-tile of each grid point:
        x_g, y_g, z_g = np.unravel_index(G_g, n_c)
        t_g = np.ravel_multi_index(
            (x_g // tile_c[0], y_g // tile_c[1], z_g // tile_c[2]), ntile_c)
        p_g = np.ravel_multi_index(
            (x_g % tile_c[0], y_g % tile_c[1], z_g % tile_c[2]), tile)
        del x_g, y_g, z_g

        # Column groups: unique (tile, M) pairs, sorted by tile and then
        # by M:
        M_W = np.asarray(M_W, dtype=np.int64)
        key_j, j_g = np.unique(t_g * nM + M_W[W_g], return_inverse=True)
        t_j = key_j // nM
        M_j = key_j % nM
        nm_M = np.zeros(nM, np.int64)
        nm_M[M_W] = nm_W
        nm_j = nm_M[M_j]

        ntiles = int(np.prod(ntile_c))
        nf_t = np.bincount(t_j, weights=nm_j,
                           minlength=ntiles).astype(np.int32)
        coloff_t = np.zeros(ntiles + 1, np.int64)
        np.cumsum(nf_t, out=coloff_t[1:])
        colstart_j = np.cumsum(nm_j) - nm_j  # global column position
        cols = (np.repeat(M_j - colstart_j, nm_j) +
                np.arange(int(nm_j.sum()))).astype(np.int32)

        phioff_t = np.zeros(ntiles + 1, np.int64)
        np.cumsum(npts * nf_t.astype(np.int64), out=phioff_t[1:])

        # Scatter the values into the tile matrices:
        Phi = np.zeros(phioff_t[-1])
        col_g = colstart_j[j_g] - coloff_t[t_g]
        base_g = phioff_t[t_g] + p_g * nf_t[t_g] + col_g
        start = 0
        for W, A_gm in enumerate(A_Wgm):
            ng, nm = A_gm.shape
            if ng == 0:
                continue
            base = base_g[start:start + ng]
            for m in range(nm):
                # += because periodic images of a function (several
                # volumes with the same M) may share grid points:
                Phi[base + m] += A_gm[:, m]
            start += ng

        # Grid point of each (tile, point-in-tile), -1 outside the grid:
        tx, ty, tz = np.unravel_index(np.arange(ntiles), ntile_c)
        lx, ly, lz = np.unravel_index(np.arange(npts), tile)
        gx = tx[:, None] * tile_c[0] + lx
        gy = ty[:, None] * tile_c[1] + ly
        gz = tz[:, None] * tile_c[2] + lz
        inside = (gx < n_c[0]) & (gy < n_c[1]) & (gz < n_c[2])
        G_tp = np.full((ntiles, npts), -1, np.int32)
        G_tp[inside] = np.ravel_multi_index(
            (gx[inside], gy[inside], gz[inside]), n_c)

        # The LFC kernel also fills the upper part of the diagonal blocks
        # (m1 < m2 within one volume); we mirror those entries to match:
        i_l, j_l = [], []
        for M0, nm in zip(M_W, nm_W):
            i, j = np.triu_indices(nm, 1)
            i_l.append(M0 + i)
            j_l.append(M0 + j)
        self.upper_i = np.concatenate(i_l, dtype=np.int64)
        self.upper_j = np.concatenate(j_l, dtype=np.int64)

        self.Phi = Phi
        self.phioff_t = phioff_t[:-1].copy()
        self.nf_t = nf_t
        self.cols = cols
        self.coloff_t = coloff_t[:-1].copy()
        self.G_tp = G_tp

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in [self.Phi, self.phioff_t, self.nf_t,
                                      self.cols, self.coloff_t, self.G_tp])

    def potential_matrix(self, vt_G, dv, Vt_MM):
        """Add lower triangle of the potential matrix to Vt_MM."""
        cgpaw.tiled_potential_matrix(
            np.ascontiguousarray(vt_G, dtype=float).ravel(), dv,
            self.Phi, self.phioff_t, self.nf_t,
            self.cols, self.coloff_t, self.G_tp, Vt_MM)
        Vt_MM[self.upper_i, self.upper_j] = Vt_MM[self.upper_j, self.upper_i]

    def construct_density(self, rho_MM, nt_G):
        assert nt_G.flags.c_contiguous
        cgpaw.tiled_construct_density(
            np.ascontiguousarray(rho_MM, dtype=float), nt_G,
            self.Phi, self.phioff_t, self.nf_t,
            self.cols, self.coloff_t, self.G_tp)
