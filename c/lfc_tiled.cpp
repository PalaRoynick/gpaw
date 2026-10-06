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

#ifndef GPAW_WITHOUT_BLAS

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

#endif  // GPAW_WITHOUT_BLAS
