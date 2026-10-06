/*  Copyright (C) 2026  CAMd
 *  Please see the accompanying LICENSE file for further information. */

// Tiled LCAO kernels (gamma point, real arithmetic).
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
#endif

CLINKAGE void dsyr2k_(const char* uplo, const char* trans, int* n, int* k,
                      double* alpha, const double* a, int* lda,
                      const double* b, int* ldb, double* beta,
                      double* c, int* ldc);
CLINKAGE void dgemm_(const char* transa, const char* transb,
                     int* m, int* n, int* k, double* alpha,
                     const double* a, int* lda, const double* b, int* ldb,
                     double* beta, double* c, int* ldc);

static int max_work(const int* nf_t, int ntiles, int npts, int* nfmax)
{
    int m = 0;
    for (int t = 0; t < ntiles; t++)
        if (nf_t[t] > m)
            m = nf_t[t];
    *nfmax = m;
    return m * (m > npts ? m : npts);
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
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &rho_obj, &nt_obj, &Phi_obj,
                          &phioff_obj, &nf_obj, &cols_obj, &coloff_obj,
                          &Gtp_obj))
        return NULL;

    const double* rho_MM = (const double*)PyArray_DATA(rho_obj);
    double* nt_G = (double*)PyArray_DATA(nt_obj);
    const double* Phi = (const double*)PyArray_DATA(Phi_obj);
    const npy_int64* phioff_t = (const npy_int64*)PyArray_DATA(phioff_obj);
    const int* nf_t = (const int*)PyArray_DATA(nf_obj);
    const int* cols = (const int*)PyArray_DATA(cols_obj);
    const npy_int64* coloff_t = (const npy_int64*)PyArray_DATA(coloff_obj);
    const int* G_tp = (const int*)PyArray_DATA(Gtp_obj);
    int ntiles = PyArray_DIMS(Gtp_obj)[0];
    int npts = PyArray_DIMS(Gtp_obj)[1];
    npy_intp ldrho = PyArray_DIMS(rho_obj)[1];

    int nfmax;
    max_work(nf_t, ntiles, npts, &nfmax);
    double* r = (double*)malloc(sizeof(double) * nfmax * nfmax);
    double* Y = (double*)malloc(sizeof(double) * nfmax * npts);
    if (r == NULL || Y == NULL) {
        free(r);
        free(Y);
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

        for (int c = 0; c < nf; c++) {
            const double* rho_c = rho_MM + col_f[c] * ldrho;
            for (int rr = 0; rr < nf; rr++)
                r[rr + c * nf] = rho_c[col_f[rr]];
        }

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
    if (!PyArg_ParseTuple(args, "OdOOOOOOOOOOOOOOOOOO",
                          &vt_obj, &dv, &rho_obj, &F_obj,
                          &Phi_obj, &phioff_obj, &nf_obj, &cols_obj,
                          &coloff_obj, &Gtp_obj,
                          &pairoff_obj, &pairW_obj, &paircol_obj,
                          &pairmask_obj, &M_W_obj, &pos_Wv_obj,
                          &spline_M_obj, &h_cv_obj, &n_c_obj, &beg_c_obj))
        return NULL;

    const double* vt_G = (const double*)PyArray_DATA(vt_obj);
    const double* rho_MM = (const double*)PyArray_DATA(rho_obj);
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

    Py_BEGIN_ALLOW_THREADS
    for (int t = 0; t < ntiles; t++) {
        int nf = nf_t[t];
        if (nf == 0)
            continue;
        const double* Phi_pf = Phi + phioff_t[t];
        const int* G_p = G_tp + (npy_intp)t * npts;
        const int* col_f = cols + coloff_t[t];

        // r(f1, f2) = rhoT[col f1, col f2] (column-major)
        for (int f2 = 0; f2 < nf; f2++)
            for (int f1 = 0; f1 < nf; f1++)
                r[f1 + f2 * nf] = rho_MM[col_f[f1] * nM + col_f[f2]];

        // Y (nf x npts) = r . Phi
        double one = 1.0;
        double zero = 0.0;
        dgemm_("N", "N", &nf, &npts, &nf, &one, r, &nf, Phi_pf, &nf,
               &zero, Y, &nf);

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
    Py_RETURN_NONE;
}

#endif  // GPAW_WITHOUT_BLAS
