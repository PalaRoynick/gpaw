/*  Copyright (C) 2026  CAMd
 *  Please see the accompanying LICENSE file for further information. */

// Tiled LCAO kernels.
//
// The basis functions are stored per tile of grid points as dense
// npts x nf matrices (point-major), where nf is the number of basis
// functions touching the tile.  This turns the potential-matrix and
// density construction into small BLAS-3 calls per tile.  See
// gpaw/lfc_tiles.py for the layout.

#include "python_utils.h"
#include "gpaw_utils.h"
#include "spline.h"
#include "bmgs/spherical_harmonics.h"
#include "bmgs/bmgs.h"

#ifndef GPAW_WITHOUT_BLAS

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef GPAW_NO_UNDERSCORE_BLAS
#  define dsyr2k_ dsyr2k
#  define dgemm_  dgemm
#  define zgemm_  zgemm
#endif

CLINKAGE void dsyr2k_(const char* uplo, const char* trans, int* n, int* k,
                      double* alpha, const double* a, int* lda,
                      const double* b, int* ldb, double* beta,
                      double* c, int* ldc);
CLINKAGE void dgemm_(const char* transa, const char* transb,
                     int* m, int* n, int* k, double* alpha,
                     const double* a, int* lda, const double* b, int* ldb,
                     double* beta, double* c, int* ldc);
CLINKAGE void zgemm_(const char* transa, const char* transb,
                     int* m, int* n, int* k, void* alpha,
                     const void* a, int* lda, const void* b, int* ldb,
                     void* beta, void* c, int* ldc);

static int max_work(const int* nf_t, int ntiles, int npts, int* nfmax)
{
    int m = 0;
    for (int t = 0; t < ntiles; t++)
        if (nf_t[t] > m)
            m = nf_t[t];
    *nfmax = m;
    return m * (m > npts ? m : npts);
}

// Gather the coefficient matrix of a tile (column-major nf x nf):
//
//   real rho:     r(a, b) = rho[M_a, M_b]
//   complex rho:  r(a, b) = Re(w_ab rho[M_a, M_b]) with
//                 w_ab = conj(ph_a) ph_b  (conj_first) or ph_a conj(ph_b),
//                 ph_a being the Bloch phase of the volume of column a.
static void gather_coefficients(double* r, int nf, const int* col_f,
                                PyArrayObject* rho_obj,
                                const int* colW_f,
                                const double_complex* ph_W,
                                bool conj_first)
{
    npy_intp ld = PyArray_DIMS(rho_obj)[1];
    if (ph_W == NULL) {
        const double* rho = (const double*)PyArray_DATA(rho_obj);
        for (int b = 0; b < nf; b++)
            for (int a = 0; a < nf; a++)
                r[a + b * nf] = rho[col_f[a] * ld + col_f[b]];
        return;
    }
    const double_complex* rho = (const double_complex*)PyArray_DATA(rho_obj);
    for (int b = 0; b < nf; b++) {
        double_complex phb = ph_W[colW_f[b]];
        for (int a = 0; a < nf; a++) {
            double_complex pha = ph_W[colW_f[a]];
            double_complex w = conj_first ? conj(pha) * phb : pha * conj(phb);
            r[a + b * nf] = creal(w * rho[col_f[a] * ld + col_f[b]]);
        }
    }
}

// Optional k-point data (a Python tuple, or None for real arithmetic):
//
//   colW:      volume of each (per-volume) column
//   ph_W:      Bloch phase of each volume
//   mcol:      index of each column in its tile's list of distinct M's
//   mM:        distinct M's of each tile
//   nfM_t:     number of distinct M's per tile
//   mcoloff_t: offset of each tile's list in mM
typedef struct {
    const int* colW;
    const double_complex* ph_W;
    const int* mcol;
    const int* mM;
    const int* nfM_t;
    const npy_int64* mcoloff_t;
    int nfMmax;
} KData;

static bool get_kdata(PyObject* obj, KData* kd, int ntiles)
{
    memset(kd, 0, sizeof(KData));
    if (obj == NULL || obj == Py_None)
        return true;
    if (!PyTuple_Check(obj) || PyTuple_Size(obj) != 6) {
        PyErr_SetString(PyExc_ValueError, "Bad k-point data.");
        return false;
    }
#define ITEM(i) PyArray_DATA((PyArrayObject*)PyTuple_GET_ITEM(obj, i))
    kd->colW = (const int*)ITEM(0);
    kd->ph_W = (const double_complex*)ITEM(1);
    kd->mcol = (const int*)ITEM(2);
    kd->mM = (const int*)ITEM(3);
    kd->nfM_t = (const int*)ITEM(4);
    kd->mcoloff_t = (const npy_int64*)ITEM(5);
#undef ITEM
    for (int t = 0; t < ntiles; t++)
        if (kd->nfM_t[t] > kd->nfMmax)
            kd->nfMmax = kd->nfM_t[t];
    return true;
}

// Bloch sums are worth it when a tile has many periodic images of the
// same functions (small cells): the GEMMs then run over the distinct M's
// instead of over all images.
static inline bool use_bloch_sum(const KData* kd, int t, int nf)
{
    return kd->ph_W != NULL && nf > 2 * kd->nfM_t[t];
}

// Psi (column-major nfM x npts) = sum over images of (conj) ph_W Phi
static void bloch_sum(double_complex* Psi, int nfM, int nf, int npts,
                      const double* Phi_pf, const int* mcol_f,
                      const int* colW_f, const double_complex* ph_W,
                      bool conj_phase)
{
    for (int i = 0; i < nfM * npts; i++)
        Psi[i] = 0.0;
    for (int f = 0; f < nf; f++) {
        double_complex ph = ph_W[colW_f[f]];
        if (conj_phase)
            ph = conj(ph);
        double_complex* Psi_p = Psi + mcol_f[f];
        for (int p = 0; p < npts; p++)
            Psi_p[p * nfM] += ph * Phi_pf[p * nf + f];
    }
}

// rs (column-major nfM x nfM) = rho[mM_j1, mM_j2]
static void gather_complex(double_complex* rs, int nfM, const int* mM_j,
                           PyArrayObject* rho_obj)
{
    npy_intp ld = PyArray_DIMS(rho_obj)[1];
    const double_complex* rho = (const double_complex*)PyArray_DATA(rho_obj);
    for (int j2 = 0; j2 < nfM; j2++)
        for (int j1 = 0; j1 < nfM; j1++)
            rs[j1 + j2 * nfM] = rho[mM_j[j1] * ld + mM_j[j2]];
}

// Add lower triangle of
//
//   V_MM[M1, M2] += sum_G Phi_M1(G) vt(G) dv Phi_M2(G),   M1 >= M2.
//
// Arguments: vt_G (flat), dv, Phi (flat), phi_offset_t (int64),
// nf_t (int32), cols (int32, sorted per tile), col_offset_t (int64),
// G_tp (int32, -1 for points outside the grid), V_MM (nM x ldV).
PyObject* tiled_potential_matrix(PyObject* self, PyObject* args)
{
    PyArrayObject *vt_obj, *Phi_obj, *phioff_obj, *nf_obj;
    PyArrayObject *cols_obj, *coloff_obj, *Gtp_obj, *V_obj;
    double dv;
    if (!PyArg_ParseTuple(args, "OdOOOOOOO", &vt_obj, &dv, &Phi_obj,
                          &phioff_obj, &nf_obj, &cols_obj, &coloff_obj,
                          &Gtp_obj, &V_obj))
        return NULL;

    const double* vt_G = (const double*)PyArray_DATA(vt_obj);
    const double* Phi = (const double*)PyArray_DATA(Phi_obj);
    const npy_int64* phioff_t = (const npy_int64*)PyArray_DATA(phioff_obj);
    const int* nf_t = (const int*)PyArray_DATA(nf_obj);
    const int* cols = (const int*)PyArray_DATA(cols_obj);
    const npy_int64* coloff_t = (const npy_int64*)PyArray_DATA(coloff_obj);
    const int* G_tp = (const int*)PyArray_DATA(Gtp_obj);
    double* V_MM = (double*)PyArray_DATA(V_obj);
    int ntiles = PyArray_DIMS(Gtp_obj)[0];
    int npts = PyArray_DIMS(Gtp_obj)[1];
    npy_intp ldV = PyArray_DIMS(V_obj)[1];

    int nfmax;
    max_work(nf_t, ntiles, npts, &nfmax);
    double* X = (double*)malloc(sizeof(double) * nfmax * npts);
    double* B = (double*)malloc(sizeof(double) * nfmax * nfmax);
    if (X == NULL || B == NULL) {
        free(X);
        free(B);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    for (int t = 0; t < ntiles; t++) {
        int nf = nf_t[t];
        if (nf == 0)
            continue;
        const double* Phi_pf = Phi + phioff_t[t];
        const int* G_p = G_tp + (npy_intp)t * npts;
        const int* col_f = cols + coloff_t[t];

        // X = 0.5 * vt * dv * Phi
        for (int p = 0; p < npts; p++) {
            double v = G_p[p] >= 0 ? 0.5 * dv * vt_G[G_p[p]] : 0.0;
            for (int f = 0; f < nf; f++)
                X[p * nf + f] = v * Phi_pf[p * nf + f];
        }

        // Column-major view: Phi and X are nf x npts.
        // B = Phi X^T + X Phi^T = Phi^T diag(vt dv) Phi (lower part).
        double one = 1.0;
        double zero = 0.0;
        dsyr2k_("L", "N", &nf, &npts, &one, Phi_pf, &nf, X, &nf,
                &zero, B, &nf);

        // B(r, c) for r >= c lives at B[r + c * nf]; cols are sorted,
        // so col_f[r] >= col_f[c]:
        for (int c = 0; c < nf; c++) {
            double* V_c = V_MM + col_f[c];
            for (int r = c; r < nf; r++)
                V_c[col_f[r] * ldV] += B[r + c * nf];
        }
    }
    Py_END_ALLOW_THREADS

    free(X);
    free(B);
    Py_RETURN_NONE;
}

// nt_G[G] += sum_{M1 M2} Phi_M1(G) rho_MM[M1, M2] Phi_M2(G)
//
// rho_MM must be symmetric.
PyObject* tiled_construct_density(PyObject* self, PyObject* args)
{
    PyArrayObject *rho_obj, *nt_obj, *Phi_obj, *phioff_obj, *nf_obj;
    PyArrayObject *cols_obj, *coloff_obj, *Gtp_obj;
    PyObject *kd_obj = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOO|O", &rho_obj, &nt_obj, &Phi_obj,
                          &phioff_obj, &nf_obj, &cols_obj, &coloff_obj,
                          &Gtp_obj, &kd_obj))
        return NULL;
    KData kd;
    if (!get_kdata(kd_obj, &kd, PyArray_DIMS(Gtp_obj)[0]))
        return NULL;

    double* nt_G = (double*)PyArray_DATA(nt_obj);
    const double* Phi = (const double*)PyArray_DATA(Phi_obj);
    const npy_int64* phioff_t = (const npy_int64*)PyArray_DATA(phioff_obj);
    const int* nf_t = (const int*)PyArray_DATA(nf_obj);
    const int* cols = (const int*)PyArray_DATA(cols_obj);
    const npy_int64* coloff_t = (const npy_int64*)PyArray_DATA(coloff_obj);
    const int* G_tp = (const int*)PyArray_DATA(Gtp_obj);
    int ntiles = PyArray_DIMS(Gtp_obj)[0];
    int npts = PyArray_DIMS(Gtp_obj)[1];

    int nfmax;
    max_work(nf_t, ntiles, npts, &nfmax);
    double* r = (double*)malloc(sizeof(double) * nfmax * nfmax);
    double* Y = (double*)malloc(sizeof(double) * nfmax * npts);
    int nfM = kd.nfMmax;
    double_complex* Psi = GPAW_MALLOC(double_complex, nfM * npts + 1);
    double_complex* PsiC = GPAW_MALLOC(double_complex, nfM * npts + 1);
    double_complex* Z = GPAW_MALLOC(double_complex, nfM * npts + 1);
    double_complex* rs = GPAW_MALLOC(double_complex, nfM * nfM + 1);

    Py_BEGIN_ALLOW_THREADS
    for (int t = 0; t < ntiles; t++) {
        int nf = nf_t[t];
        if (nf == 0)
            continue;
        const double* Phi_pf = Phi + phioff_t[t];
        const int* G_p = G_tp + (npy_intp)t * npts;
        const int* col_f = cols + coloff_t[t];

        if (use_bloch_sum(&kd, t, nf)) {
            // n = Re sum_j Psi_j (rho conj(Psi))_j with Psi = sum ph Phi
            int nfM = kd.nfM_t[t];
            bloch_sum(Psi, nfM, nf, npts, Phi_pf, kd.mcol + coloff_t[t],
                      kd.colW + coloff_t[t], kd.ph_W, false);
            for (int i = 0; i < nfM * npts; i++)
                PsiC[i] = conj(Psi[i]);
            gather_complex(rs, nfM, kd.mM + kd.mcoloff_t[t], rho_obj);
            double_complex one = 1.0;
            double_complex zero = 0.0;
            zgemm_("N", "N", &nfM, &npts, &nfM, &one, rs, &nfM, PsiC, &nfM,
                   &zero, Z, &nfM);
            for (int p = 0; p < npts; p++) {
                int G = G_p[p];
                if (G < 0)
                    continue;
                double n = 0.0;
                for (int j = 0; j < nfM; j++)
                    n += creal(Psi[j + p * nfM] * Z[j + p * nfM]);
                nt_G[G] += n;
            }
            continue;
        }

        gather_coefficients(r, nf, col_f, rho_obj,
                            kd.colW ? kd.colW + coloff_t[t] : NULL,
                            kd.ph_W, false);

        // Column-major: Y (nf x npts) = r (nf x nf) . Phi (nf x npts)
        double one = 1.0;
        double zero = 0.0;
        dgemm_("N", "N", &nf, &npts, &nf, &one, r, &nf, Phi_pf, &nf,
               &zero, Y, &nf);

        for (int p = 0; p < npts; p++) {
            int G = G_p[p];
            if (G < 0)
                continue;
            double n = 0.0;
            for (int f = 0; f < nf; f++)
                n += Y[p * nf + f] * Phi_pf[p * nf + f];
            nt_G[G] += n;
        }
    }
    Py_END_ALLOW_THREADS

    free(r);
    free(Y);
    free(Psi);
    free(PsiC);
    free(Z);
    free(rs);
    Py_RETURN_NONE;
}

// Displacement-resolved potential matrices (k-points), per-volume columns:
//
//   Vt_xMM[x, M1, M2] += sum_G Phi_W1(G) vt(G) dv Phi_W2(G)
//
// for all pairs of volumes with x = x_W[W2] - x_W[W1] >= 0 (like the LFC
// kernel calculate_potential_matrices).
PyObject* tiled_potential_matrices(PyObject* self, PyObject* args)
{
    PyArrayObject *vt_obj, *Phi_obj, *phioff_obj, *nf_obj, *cols_obj;
    PyArrayObject *colW_obj, *coloff_obj, *Gtp_obj, *x_W_obj, *V_obj;
    double dv;
    if (!PyArg_ParseTuple(args, "OdOOOOOOOOO", &vt_obj, &dv, &Phi_obj,
                          &phioff_obj, &nf_obj, &cols_obj, &colW_obj,
                          &coloff_obj, &Gtp_obj, &x_W_obj, &V_obj))
        return NULL;

    const double* vt_G = (const double*)PyArray_DATA(vt_obj);
    const double* Phi = (const double*)PyArray_DATA(Phi_obj);
    const npy_int64* phioff_t = (const npy_int64*)PyArray_DATA(phioff_obj);
    const int* nf_t = (const int*)PyArray_DATA(nf_obj);
    const int* cols = (const int*)PyArray_DATA(cols_obj);
    const int* colW = (const int*)PyArray_DATA(colW_obj);
    const npy_int64* coloff_t = (const npy_int64*)PyArray_DATA(coloff_obj);
    const int* G_tp = (const int*)PyArray_DATA(Gtp_obj);
    const int* x_W = (const int*)PyArray_DATA(x_W_obj);
    double* V_xMM = (double*)PyArray_DATA(V_obj);
    int ntiles = PyArray_DIMS(Gtp_obj)[0];
    int npts = PyArray_DIMS(Gtp_obj)[1];
    npy_intp nx = PyArray_DIMS(V_obj)[0];
    npy_intp nM1 = PyArray_DIMS(V_obj)[1];
    npy_intp nM2 = PyArray_DIMS(V_obj)[2];

    int nfmax;
    max_work(nf_t, ntiles, npts, &nfmax);
    double* X = (double*)malloc(sizeof(double) * nfmax * npts);
    double* B = (double*)malloc(sizeof(double) * nfmax * nfmax);
    if (X == NULL || B == NULL) {
        free(X);
        free(B);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    for (int t = 0; t < ntiles; t++) {
        int nf = nf_t[t];
        if (nf == 0)
            continue;
        const double* Phi_pf = Phi + phioff_t[t];
        const int* G_p = G_tp + (npy_intp)t * npts;
        const int* col_f = cols + coloff_t[t];
        const int* W_f = colW + coloff_t[t];

        for (int p = 0; p < npts; p++) {
            double v = G_p[p] >= 0 ? 0.5 * dv * vt_G[G_p[p]] : 0.0;
            for (int f = 0; f < nf; f++)
                X[p * nf + f] = v * Phi_pf[p * nf + f];
        }
        double one = 1.0;
        double zero = 0.0;
        dsyr2k_("L", "N", &nf, &npts, &one, Phi_pf, &nf, X, &nf,
                &zero, B, &nf);

        for (int c = 0; c < nf; c++) {
            int Mc = col_f[c];
            int xc = x_W[W_f[c]];
            for (int r = c; r < nf; r++) {
                double value = B[r + c * nf];
                int Mr = col_f[r];
                int xr = x_W[W_f[r]];
                // (W1, W2) = (r, c):
                int x = xc - xr;
                if (x >= 0 && x < nx)
                    V_xMM[(x * nM1 + Mr) * nM2 + Mc] += value;
                // (W1, W2) = (c, r):
                x = xr - xc;
                if (r != c && x >= 0 && x < nx)
                    V_xMM[(x * nM1 + Mc) * nM2 + Mr] += value;
            }
        }
    }
    Py_END_ALLOW_THREADS

    free(X);
    free(B);
    Py_RETURN_NONE;
}

// Potential term of the LCAO forces (all three directions):
//
//   F_vM[c, M1] += sum_G dPhi_M1(G)/dr_c vt(G) dv sum_M2 rhoT[M1, M2] Phi_M2(G)
//
// The derivatives are evaluated per (tile, volume) pair on the grid
// points given by the pair's bit mask (the points of that volume in the
// tile), so periodic images of a function are handled like in the LFC
// kernel.
PyObject* tiled_force_contribution(PyObject* self, PyObject* args)
{
    PyArrayObject *vt_obj, *rho_obj, *F_obj, *Phi_obj, *phioff_obj;
    PyArrayObject *nf_obj, *cols_obj, *coloff_obj, *Gtp_obj;
    PyArrayObject *pairoff_obj, *pairW_obj, *paircol_obj, *pairmask_obj;
    PyArrayObject *M_W_obj, *pos_Wv_obj, *spline_M_obj;
    PyArrayObject *h_cv_obj, *n_c_obj, *beg_c_obj;
    double dv;
    PyObject *kd_obj = NULL;
    if (!PyArg_ParseTuple(args, "OdOOOOOOOOOOOOOOOOOO|O",
                          &vt_obj, &dv, &rho_obj, &F_obj,
                          &Phi_obj, &phioff_obj, &nf_obj, &cols_obj,
                          &coloff_obj, &Gtp_obj,
                          &pairoff_obj, &pairW_obj, &paircol_obj,
                          &pairmask_obj, &M_W_obj, &pos_Wv_obj,
                          &spline_M_obj, &h_cv_obj, &n_c_obj, &beg_c_obj,
                          &kd_obj))
        return NULL;
    KData kd;
    if (!get_kdata(kd_obj, &kd, PyArray_DIMS(Gtp_obj)[0]))
        return NULL;

    const double* vt_G = (const double*)PyArray_DATA(vt_obj);
    double* F_vM = (double*)PyArray_DATA(F_obj);
    const double* Phi = (const double*)PyArray_DATA(Phi_obj);
    const npy_int64* phioff_t = (const npy_int64*)PyArray_DATA(phioff_obj);
    const int* nf_t = (const int*)PyArray_DATA(nf_obj);
    const int* cols = (const int*)PyArray_DATA(cols_obj);
    const npy_int64* coloff_t = (const npy_int64*)PyArray_DATA(coloff_obj);
    const int* G_tp = (const int*)PyArray_DATA(Gtp_obj);
    const npy_int64* pairoff_t = (const npy_int64*)PyArray_DATA(pairoff_obj);
    const int* pair_W = (const int*)PyArray_DATA(pairW_obj);
    const int* pair_col = (const int*)PyArray_DATA(paircol_obj);
    const uint64_t* pair_mask = (const uint64_t*)PyArray_DATA(pairmask_obj);
    const int* M_W = (const int*)PyArray_DATA(M_W_obj);
    const double (*pos_Wv)[3] = (const double (*)[3])PyArray_DATA(pos_Wv_obj);
    const SplineObject** spline_M =
        (const SplineObject**)PyArray_DATA(spline_M_obj);
    const double* h_cv = (const double*)PyArray_DATA(h_cv_obj);
    const long* n_c = (const long*)PyArray_DATA(n_c_obj);
    const long* beg_c = (const long*)PyArray_DATA(beg_c_obj);
    int ntiles = PyArray_DIMS(Gtp_obj)[0];
    int npts = PyArray_DIMS(Gtp_obj)[1];
    npy_intp nM = PyArray_DIMS(rho_obj)[1];
    if (npts > 128) {
        PyErr_SetString(PyExc_ValueError, "At most 128 points per tile.");
        return NULL;
    }

    int nfmax;
    max_work(nf_t, ntiles, npts, &nfmax);
    double* r = (double*)malloc(sizeof(double) * nfmax * nfmax);
    double* Y = (double*)malloc(sizeof(double) * nfmax * npts);
    if (r == NULL || Y == NULL) {
        free(r);
        free(Y);
        return PyErr_NoMemory();
    }
    int nfMmax = kd.nfMmax;
    double_complex* Psi = GPAW_MALLOC(double_complex, nfMmax * npts + 1);
    double_complex* Z = GPAW_MALLOC(double_complex, nfMmax * npts + 1);
    double_complex* rs = GPAW_MALLOC(double_complex, nfMmax * nfMmax + 1);

    Py_BEGIN_ALLOW_THREADS
    for (int t = 0; t < ntiles; t++) {
        int nf = nf_t[t];
        if (nf == 0)
            continue;
        const double* Phi_pf = Phi + phioff_t[t];
        const int* G_p = G_tp + (npy_intp)t * npts;
        const int* col_f = cols + coloff_t[t];

        if (use_bloch_sum(&kd, t, nf)) {
            // Y(a, p) = Re(conj(ph_a) (rhoT Psi)(M_a, p)), Psi = sum ph Phi
            int nfM = kd.nfM_t[t];
            const int* mcol_f = kd.mcol + coloff_t[t];
            const int* colW_f = kd.colW + coloff_t[t];
            bloch_sum(Psi, nfM, nf, npts, Phi_pf, mcol_f, colW_f,
                      kd.ph_W, false);
            gather_complex(rs, nfM, kd.mM + kd.mcoloff_t[t], rho_obj);
            double_complex one = 1.0;
            double_complex zero = 0.0;
            zgemm_("N", "N", &nfM, &npts, &nfM, &one, rs, &nfM, Psi, &nfM,
                   &zero, Z, &nfM);
            for (int a = 0; a < nf; a++) {
                double_complex phc = conj(kd.ph_W[colW_f[a]]);
                for (int p = 0; p < npts; p++)
                    Y[a + p * nf] = creal(phc * Z[mcol_f[a] + p * nfM]);
            }
        }
        else {
            // r(f1, f2) = rhoT[col f1, col f2] (column-major)
            gather_coefficients(r, nf, col_f, rho_obj,
                                kd.colW ? kd.colW + coloff_t[t] : NULL,
                                kd.ph_W, true);

            // Y (nf x npts) = r . Phi
            double one = 1.0;
            double zero = 0.0;
            dgemm_("N", "N", &nf, &npts, &nf, &one, r, &nf, Phi_pf, &nf,
                   &zero, Y, &nf);
        }

        for (npy_int64 j = pairoff_t[t]; j < pairoff_t[t + 1]; j++) {
            int W = pair_W[j];
            int M0 = M_W[W];
            int f0 = pair_col[j];
            const bmgsspline* spline = &spline_M[M0]->spline;
            int l = spline->l;
            int nm = 2 * l + 1;
            const double* pos_v = pos_Wv[W];
            const uint64_t* mask = pair_mask + 2 * j;
            double fdYdc_m[2 * 4 + 1];  // l <= 4
            double rlYdfdr_m[2 * 4 + 1];
            for (int p = 0; p < npts; p++) {
                if (!((mask[p >> 6] >> (p & 63)) & 1))
                    continue;
                int G = G_p[p];
                long iz = G % n_c[2] + beg_c[2];
                long iy = (G / n_c[2]) % n_c[1] + beg_c[1];
                long ix = G / (n_c[2] * n_c[1]) + beg_c[0];
                double R_c[3];
                for (int v = 0; v < 3; v++)
                    R_c[v] = (h_cv[v] * ix + h_cv[3 + v] * iy +
                              h_cv[6 + v] * iz - pos_v[v]);
                double x = R_c[0];
                double y = R_c[1];
                double z = R_c[2];
                double r2 = x * x + y * y + z * z;
                double rr = sqrt(r2);
                double f, dfdr;
                bmgs_get_value_and_derivative(spline, rr, &f, &dfdr);
                double vtdv = vt_G[G] * dv;
                const double* Y_f = Y + p * nf + f0;
                for (int c = 0; c < 3; c++) {
                    double Rcinvr = rr > 1e-15 ? R_c[c] / rr : 0.0;
                    switch (c) {
                    case 0:
                        spherical_harmonics_derivative_x(l, f, x, y, z, r2,
                                                         fdYdc_m);
                        break;
                    case 1:
                        spherical_harmonics_derivative_y(l, f, x, y, z, r2,
                                                         fdYdc_m);
                        break;
                    case 2:
                        spherical_harmonics_derivative_z(l, f, x, y, z, r2,
                                                         fdYdc_m);
                        break;
                    }
                    spherical_harmonics(l, dfdr * Rcinvr, x, y, z, r2,
                                        rlYdfdr_m);
                    double* F_M = F_vM + c * nM + M0;
                    for (int m = 0; m < nm; m++)
                        F_M[m] += (vtdv * (fdYdc_m[m] + rlYdfdr_m[m]) *
                                   Y_f[m]);
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    free(r);
    free(Y);
    free(Psi);
    free(Z);
    free(rs);
    Py_RETURN_NONE;
}

#endif  // GPAW_WITHOUT_BLAS
