#include "pw.h"
#include "python_utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

//#include "gpu/cpp/pyarray_utils.hpp"
//#include "gpu/cpp/pybind_cupy_type_caster.hpp"

#include "gpaw_complex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace py = pybind11;

inline bool is_c_contiguous(pybind11::array arr) { return arr.flags() & pybind11::array::c_style; }

#ifndef GPAW_WITHOUT_BLAS
#ifdef GPAW_NO_UNDERSCORE_BLAS
#  define ssyr2k_ ssyr2k
#  define sgemm_  sgemm
#  define dgemm_  dgemm
#  define dsyr2k_ dsyr2k
#  define cher2k_ cher2k
#  define cgemm_  cgemm
#  define zher2k_ zher2k
#  define zgemm_  zgemm
#endif

extern "C" {
void ssyr2k_(const char* uplo, const char* trans, int* n, int* k,
             const float* alpha, const float* a, int* lda,
             const float* b, int* ldb, const float* beta,
             float* c, int* ldc);
void sgemm_(const char* transa, const char* transb, int* m, int* n,
            int* k, const float* alpha, const float* a, int* lda,
            const float* b, int* ldb, const float* beta,
            float* c, int* ldc);
void dsyr2k_(const char* uplo, const char* trans, int* n, int* k,
             const double* alpha, const double* a, int* lda,
             const double* b, int* ldb, const double* beta,
             double* c, int* ldc);
void dgemm_(const char* transa, const char* transb, int* m, int* n,
            int* k, const double* alpha, const double* a, int* lda,
            const double* b, int* ldb, const double* beta,
            double* c, int* ldc);
void cher2k_(const char* uplo, const char* trans, int* n, int* k,
             void* alpha, void* a, int* lda,
             void* b, int* ldb, const float* beta,
             void* c, int* ldc);
void cgemm_(const char* transa, const char* transb, int* m, int* n,
            int* k, void* alpha, void* a, int* lda,
            void* b, int* ldb, void* beta,
            void* c, int* ldc);
void zher2k_(const char* uplo, const char* trans, int* n, int* k,
             void* alpha, void* a, int* lda,
             void* b, int* ldb, const double* beta,
             void* c, int* ldc);
void zgemm_(const char* transa, const char* transb, int* m, int* n,
            int* k, void* alpha, void* a, int* lda,
            void* b, int* ldb, void* beta,
            void* c, int* ldc);
}
#endif

template <typename T> void _pw_insert(int nG,
                int nQ,
                const std::complex<T>* c_G,
                const npy_int32* Q_G,
                T scale,
                std::complex<T>* tmp_Q)
// Does the same as these two lines of Python:
//
//     tmp_Q[:] = 0.0
//     tmp_Q.ravel()[Q_G] = c_G * scale
{
    int Q1 = 0;
    for (int G = 0; G < nG; G++) {
        int Q2 = Q_G[G];
        for (; Q1 < Q2; Q1++)
            tmp_Q[Q1] = 0.0;
        tmp_Q[Q1++] = c_G[G] * scale;
        }
    for (; Q1 < nQ; Q1++)
        tmp_Q[Q1] = 0.0;
}

template <typename T>
void _add_to_density_real(double f,
                          const py::buffer_info& psit_info,
                          double* nt_R)
{
    const auto* psit_base = static_cast<const char*>(psit_info.ptr);
    const ssize_t ndim = static_cast<ssize_t>(psit_info.ndim);
    std::vector<ssize_t> index(ndim, 0);
    const ssize_t n = std::accumulate(psit_info.shape.begin(),
                                      psit_info.shape.end(),
                                      ssize_t{1},
                                      std::multiplies<ssize_t>());

    for (ssize_t i = 0; i < n; i++) {
        ssize_t offset = 0;
        for (ssize_t d = 0; d < ndim; d++) {
            offset += index[d] * psit_info.strides[d];
        }
        T value = *reinterpret_cast<const T*>(psit_base + offset);
        nt_R[i] += f * value * value;

        for (ssize_t d = ndim - 1; d >= 0; d--) {
            index[d]++;
            if (index[d] < psit_info.shape[d]) {
                break;
            }
            index[d] = 0;
        }
    }
}

template <typename T>
void _add_to_density_complex(double f,
                             const std::complex<T>* psit_R,
                             double* nt_R,
                             ssize_t n)
{
    for (ssize_t i = 0; i < n; i++) {
        nt_R[i] += f * std::norm(psit_R[i]);
    }
}

template <typename T> void _pw_precond(int nG,
                const T* G2_G,
                const std::complex<T>* R_G,
                T ekin,
                std::complex<T>* out_G)
{
    for (int G = 0; G < nG; G++) {
        T x = G2_G[G] / (ekin * T(3.0));
        T a = T(27.0) + x * (T(18.0) + x * (T(12.0) + x * T(8.0)));
        T xx = x * x;
        out_G[G] = (-T(4.0) / (T(3.0) * ekin) *
                    a / (a + T(16.0) * xx * xx) * R_G[G]);
    }
}

template <typename T> void _pwlfc_expand_complex(int nG,
                int nJ,
                int nL,
                int nsplines,
                const T* f_Gs,
                const T* GK_Gv,
                const T* pos_av,
                const std::complex<T>* eikR_a,
                const T* Y_GL,
                const npy_int32* l_s,
                const npy_int32* a_J,
                const npy_int32* s_J,
                bool cc,
                std::complex<T>* f_GI)
{
    const std::complex<T> imag_powers[4] = {
        std::complex<T>(1.0, 0.0),
        std::complex<T>(0.0, -1.0),
        std::complex<T>(-1.0, 0.0),
        std::complex<T>(0.0, 1.0)
    };

    std::complex<T> emiGR(0.0, 0.0);

    for (int G = 0; G < nG; G++) {
        int a_old = -1;
        for (int J = 0; J < nJ; J++) {
            int s = s_J[J];
            int l = l_s[s];

            if (a_J[J] != a_old) {
                a_old = a_J[J];
                T f0 = (GK_Gv[0] * pos_av[0 + 3 * a_old] +
                        GK_Gv[1] * pos_av[1 + 3 * a_old] +
                        GK_Gv[2] * pos_av[2 + 3 * a_old]);
                emiGR = std::complex<T>(std::cos(f0), -std::sin(f0)) * eikR_a[a_old];
            }

            std::complex<T> f1 = emiGR * f_Gs[s] * imag_powers[l % 4];
            for (int m = 0; m < 2 * l + 1; m++) {
                std::complex<T> f = f1 * Y_GL[l * l + m];
                *f_GI++ = cc ? std::conj(f) : f;
            }
        }
        f_Gs += nsplines;
        GK_Gv += 3;
        Y_GL += nL;
    }
}

template <typename T> void _pwlfc_expand_real(int nG,
                int nJ,
                int nL,
                int nsplines,
                const T* f_Gs,
                const T* GK_Gv,
                const T* pos_av,
                const std::complex<T>* eikR_a,
                const T* Y_GL,
                const npy_int32* l_s,
                const npy_int32* a_J,
                const npy_int32* s_J,
                bool cc,
                T* f_GI,
                int nI)
{
    const std::complex<T> imag_powers[4] = {
        std::complex<T>(1.0, 0.0),
        std::complex<T>(0.0, -1.0),
        std::complex<T>(-1.0, 0.0),
        std::complex<T>(0.0, 1.0)
    };

    std::complex<T> emiGR(0.0, 0.0);

    for (int G = 0; G < nG; G++) {
        int a_old = -1;
        for (int J = 0; J < nJ; J++) {
            int s = s_J[J];
            int l = l_s[s];

            if (a_J[J] != a_old) {
                a_old = a_J[J];
                T f0 = (GK_Gv[0] * pos_av[0 + 3 * a_old] +
                        GK_Gv[1] * pos_av[1 + 3 * a_old] +
                        GK_Gv[2] * pos_av[2 + 3 * a_old]);
                emiGR = std::complex<T>(std::cos(f0), -std::sin(f0)) * eikR_a[a_old];
            }

            std::complex<T> f1 = emiGR * f_Gs[s] * imag_powers[l % 4];
            for (int m = 0; m < 2 * l + 1; m++) {
                std::complex<T> f = f1 * Y_GL[l * l + m];
                f_GI[0] = std::real(f);
                f_GI[nI] = cc ? -std::imag(f) : std::imag(f);
                f_GI++;
            }
        }
        f_Gs += nsplines;
        GK_Gv += 3;
        Y_GL += nL;
        f_GI += nI;
    }
}

template <typename T> void _pwlfc_expand_old_complex(int nG,
                int nJ,
                int nL,
                int natoms,
                int nsplines,
                const T* f_Gs,
                const std::complex<T>* emiGR_Ga,
                const T* Y_GL,
                const npy_int32* l_s,
                const npy_int32* a_J,
                const npy_int32* s_J,
                bool cc,
                std::complex<T>* f_GI)
{
    const std::complex<T> imag_powers[4] = {
        std::complex<T>(1.0, 0.0),
        std::complex<T>(0.0, -1.0),
        std::complex<T>(-1.0, 0.0),
        std::complex<T>(0.0, 1.0)
    };

    for (int G = 0; G < nG; G++) {
        for (int J = 0; J < nJ; J++) {
            int s = s_J[J];
            int l = l_s[s];
            std::complex<T> f1 = emiGR_Ga[a_J[J]] * f_Gs[s] * imag_powers[l % 4];
            for (int m = 0; m < 2 * l + 1; m++) {
                std::complex<T> f = f1 * Y_GL[l * l + m];
                *f_GI++ = cc ? std::conj(f) : f;
            }
        }
        f_Gs += nsplines;
        emiGR_Ga += natoms;
        Y_GL += nL;
    }
}

template <typename T> void _pwlfc_expand_old_real(int nG,
                int nJ,
                int nL,
                int natoms,
                int nsplines,
                const T* f_Gs,
                const std::complex<T>* emiGR_Ga,
                const T* Y_GL,
                const npy_int32* l_s,
                const npy_int32* a_J,
                const npy_int32* s_J,
                bool cc,
                T* f_GI,
                int nI)
{
    const std::complex<T> imag_powers[4] = {
        std::complex<T>(1.0, 0.0),
        std::complex<T>(0.0, -1.0),
        std::complex<T>(-1.0, 0.0),
        std::complex<T>(0.0, 1.0)
    };

    for (int G = 0; G < nG; G++) {
        for (int J = 0; J < nJ; J++) {
            int s = s_J[J];
            int l = l_s[s];
            std::complex<T> f1 = emiGR_Ga[a_J[J]] * f_Gs[s] * imag_powers[l % 4];
            for (int m = 0; m < 2 * l + 1; m++) {
                std::complex<T> f = f1 * Y_GL[l * l + m];
                f_GI[0] = std::real(f);
                f_GI[nI] = cc ? -std::imag(f) : std::imag(f);
                f_GI++;
            }
        }
        f_Gs += nsplines;
        emiGR_Ga += natoms;
        Y_GL += nL;
        f_GI += nI;
    }
}


void cpp_pw_insert(py::array c_G,
                   py::array Q_G,
                   double scale,
                   py::array tmp_Q)
{
    assert(c_G.ndim() == 1);
    assert(Q_G.ndim() == 1);
    assert(tmp_Q.ndim() == 3);
    assert(is_c_contiguous(c_G));
    assert(is_c_contiguous(Q_G));
    assert(is_c_contiguous(tmp_Q));
    int nG = c_G.shape()[0];
    int nQ = tmp_Q.size();

    bool is_double = c_G.dtype().is(py::dtype::of<std::complex<double>>());
    bool is_float = c_G.dtype().is(py::dtype::of<std::complex<float>>());

    if (is_double)
    {
        _pw_insert<double>(nG, nQ,
                   static_cast<const std::complex<double>*>(c_G.data()),
                   static_cast<const npy_int32*>(Q_G.data()),
                   scale,
                   static_cast<std::complex<double>*>(tmp_Q.mutable_data()));
    }
    else if (is_float)
    {
        _pw_insert<float>(nG, nQ,
                   static_cast<const std::complex<float>*>(c_G.data()),
                   static_cast<const npy_int32*>(Q_G.data()),
                   static_cast<float>(scale),
                   static_cast<std::complex<float>*>(tmp_Q.mutable_data()));
    }
    else
    {
        throw std::invalid_argument("Supports only double and float arrays.");
    }
//    nG =
//    _pw_insert(nG, nQ, c_G, Q_G, scale, tmp_Q)
}

void cpp_add_to_density(double f,
                        py::array psit_R,
                        py::array nt_R)
{
    assert(nt_R.dtype().is(py::dtype::of<double>()));
    assert(psit_R.size() == nt_R.size());

    double* nt = static_cast<double*>(nt_R.mutable_data());
    auto nt_info = nt_R.request();

    bool is_double = psit_R.dtype().is(py::dtype::of<double>());
    bool is_float = psit_R.dtype().is(py::dtype::of<float>());
    bool is_zdouble = psit_R.dtype().is(py::dtype::of<std::complex<double>>());
    bool is_zfloat = psit_R.dtype().is(py::dtype::of<std::complex<float>>());

    if (is_double)
    {
        auto psit_info = psit_R.request();
        assert(psit_info.ndim == nt_info.ndim);
        _add_to_density_real<double>(f, psit_info, nt);
    }
    else if (is_float)
    {
        auto psit_info = psit_R.request();
        assert(psit_info.ndim == nt_info.ndim);
        _add_to_density_real<float>(f, psit_info, nt);
    }
    else if (is_zdouble)
    {
        assert(is_c_contiguous(psit_R));
        _add_to_density_complex<double>(
            f,
            static_cast<const std::complex<double>*>(psit_R.data()),
            nt,
            psit_R.size());
    }
    else if (is_zfloat)
    {
        assert(is_c_contiguous(psit_R));
        _add_to_density_complex<float>(
            f,
            static_cast<const std::complex<float>*>(psit_R.data()),
            nt,
            psit_R.size());
    }
    else
    {
        throw std::invalid_argument(
            "add_to_density supports float32/float64/complex64/complex128 "
            "wave functions and float64 density output.");
    }
}

#ifndef GPAW_WITHOUT_BLAS
void cpp_mmm(py::object alpha_obj,
             py::array M1,
             const std::string& trans1,
             py::array M2,
             const std::string& trans2,
             py::object beta_obj,
             py::array M3)
{
    assert(M1.ndim() == 2);
    assert(M2.ndim() == 2);
    assert(M3.ndim() == 2);
    assert(M1.size() == 0 || M1.strides()[1] == M3.itemsize());
    assert(M2.size() == 0 || M2.strides()[1] == M3.itemsize());
    assert(M3.size() == 0 || M3.strides()[1] == M3.itemsize());
    assert(M1.dtype().is(M2.dtype()));
    assert(M1.dtype().is(M3.dtype()));
    assert(!trans1.empty());
    assert(!trans2.empty());

    char t1 = trans1[0];
    char t2 = trans2[0];

    int bytes = M3.itemsize();
    int m = M3.shape()[1];
    int n = M3.shape()[0];
    int lda = M2.strides()[0] / bytes;
    int ldb = M1.strides()[0] / bytes;
    int ldc = std::max(std::max(1, m), static_cast<int>(M3.strides()[0] / bytes));

    int k;
    if (t2 == 'N' || t2 == 'n') {
        k = M2.shape()[0];
        lda = std::max(std::max(1, m), lda);
    } else {
        k = M2.shape()[1];
        lda = std::max(std::max(1, k), lda);
    }

    if (t1 == 'N' || t1 == 'n') {
        ldb = std::max(std::max(1, k), ldb);
    } else {
        ldb = std::max(std::max(1, n), ldb);
    }

    if (M3.dtype().is(py::dtype::of<double>()))
    {
        double alpha = py::cast<double>(alpha_obj);
        double beta = py::cast<double>(beta_obj);
        dgemm_(&t2, &t1, &m, &n, &k,
               &alpha,
               static_cast<const double*>(M2.data()), &lda,
               static_cast<const double*>(M1.data()), &ldb,
               &beta,
               static_cast<double*>(M3.mutable_data()), &ldc);
    }
    else if (M3.dtype().is(py::dtype::of<float>()))
    {
        float alpha = py::cast<float>(alpha_obj);
        float beta = py::cast<float>(beta_obj);
        sgemm_(&t2, &t1, &m, &n, &k,
               &alpha,
               static_cast<const float*>(M2.data()), &lda,
               static_cast<const float*>(M1.data()), &ldb,
               &beta,
               static_cast<float*>(M3.mutable_data()), &ldc);
    }
    else if (M3.dtype().is(py::dtype::of<std::complex<double>>()))
    {
        std::complex<double> alpha = py::cast<std::complex<double>>(alpha_obj);
        std::complex<double> beta = py::cast<std::complex<double>>(beta_obj);
        zgemm_(&t2, &t1, &m, &n, &k,
               &alpha,
               const_cast<void*>(static_cast<const void*>(M2.data())), &lda,
               const_cast<void*>(static_cast<const void*>(M1.data())), &ldb,
               &beta,
               static_cast<void*>(M3.mutable_data()), &ldc);
    }
    else if (M3.dtype().is(py::dtype::of<std::complex<float>>()))
    {
        std::complex<float> alpha = py::cast<std::complex<float>>(alpha_obj);
        std::complex<float> beta = py::cast<std::complex<float>>(beta_obj);
        cgemm_(&t2, &t1, &m, &n, &k,
               &alpha,
               const_cast<void*>(static_cast<const void*>(M2.data())), &lda,
               const_cast<void*>(static_cast<const void*>(M1.data())), &ldb,
               &beta,
               static_cast<void*>(M3.mutable_data()), &ldc);
    }
    else
    {
        throw std::invalid_argument("Unsupported dtype for mmm.");
    }
}
#endif

#ifndef GPAW_WITHOUT_BLAS
void cpp_r2k(py::object alpha_obj,
             py::array a,
             py::array b,
             double beta,
             py::array c,
             const std::string& trans = "c")
{
    assert(a.ndim() == 2);
    assert(b.ndim() == 2);
    assert(c.ndim() == 2);
    assert(a.dtype().is(b.dtype()));
    assert(a.dtype().is(c.dtype()));
    assert(!trans.empty());

    char t = trans[0];
    int n = c.shape()[0];
    int k;
    int lda;
    if (t == 'c' || t == 'C') {
        k = a.shape()[1];
        lda = std::max(1, static_cast<int>(a.strides()[0] / a.itemsize()));
    } else {
        k = a.shape()[0];
        lda = std::max(n, 1);
    }
    int ldc = std::max(1, static_cast<int>(c.strides()[0] / c.itemsize()));

    if (c.dtype().is(py::dtype::of<double>()))
    {
        double alpha = py::cast<double>(alpha_obj);
        dsyr2k_("u", &t, &n, &k,
                &alpha,
                static_cast<const double*>(a.data()), &lda,
                static_cast<const double*>(b.data()), &lda,
                &beta,
                static_cast<double*>(c.mutable_data()), &ldc);
    }
    else if (c.dtype().is(py::dtype::of<float>()))
    {
        float alpha = py::cast<float>(alpha_obj);
        float beta_f = static_cast<float>(beta);
        ssyr2k_("u", &t, &n, &k,
                &alpha,
                static_cast<const float*>(a.data()), &lda,
                static_cast<const float*>(b.data()), &lda,
                &beta_f,
                static_cast<float*>(c.mutable_data()), &ldc);
    }
    else if (c.dtype().is(py::dtype::of<std::complex<double>>()))
    {
        std::complex<double> alpha = py::cast<std::complex<double>>(alpha_obj);
        zher2k_("u", &t, &n, &k,
                &alpha,
                const_cast<void*>(static_cast<const void*>(a.data())), &lda,
                const_cast<void*>(static_cast<const void*>(b.data())), &lda,
                &beta,
                static_cast<void*>(c.mutable_data()), &ldc);
    }
    else if (c.dtype().is(py::dtype::of<std::complex<float>>()))
    {
        std::complex<float> alpha = py::cast<std::complex<float>>(alpha_obj);
        float beta_f = static_cast<float>(beta);
        cher2k_("u", &t, &n, &k,
                &alpha,
                const_cast<void*>(static_cast<const void*>(a.data())), &lda,
                const_cast<void*>(static_cast<const void*>(b.data())), &lda,
                &beta_f,
                static_cast<void*>(c.mutable_data()), &ldc);
    }
    else
    {
        throw std::invalid_argument("Unsupported dtype for r2k.");
    }
}
#endif

void cpp_pwlfc_expand_old(py::array f_Gs,
                          py::array emiGR_Ga,
                          py::array Y_GL,
                          py::array l_s,
                          py::array a_J,
                          py::array s_J,
                          bool cc,
                          py::array f_GI)
{
    assert(f_Gs.ndim() == 2);
    assert(emiGR_Ga.ndim() == 2);
    assert(Y_GL.ndim() == 2);
    assert(l_s.ndim() == 1);
    assert(a_J.ndim() == 1);
    assert(s_J.ndim() == 1);
    assert(f_GI.ndim() == 2);
    assert(is_c_contiguous(f_Gs));
    assert(is_c_contiguous(emiGR_Ga));
    assert(is_c_contiguous(Y_GL));
    assert(is_c_contiguous(l_s));
    assert(is_c_contiguous(a_J));
    assert(is_c_contiguous(s_J));
    assert(is_c_contiguous(f_GI));
    assert(l_s.dtype().is(py::dtype::of<npy_int32>()));
    assert(a_J.dtype().is(py::dtype::of<npy_int32>()));
    assert(s_J.dtype().is(py::dtype::of<npy_int32>()));

    int nG = emiGR_Ga.shape()[0];
    int nJ = a_J.shape()[0];
    int nL = Y_GL.shape()[1];
    int natoms = emiGR_Ga.shape()[1];
    int nsplines = f_Gs.shape()[1];

    assert(f_Gs.shape()[0] == nG);
    assert(Y_GL.shape()[0] == nG);
    assert(s_J.shape()[0] == nJ);

    bool is_double_complex = (
        f_Gs.dtype().is(py::dtype::of<double>()) &&
        emiGR_Ga.dtype().is(py::dtype::of<std::complex<double>>()) &&
        Y_GL.dtype().is(py::dtype::of<double>()) &&
        f_GI.dtype().is(py::dtype::of<std::complex<double>>()));

    bool is_float_complex = (
        f_Gs.dtype().is(py::dtype::of<float>()) &&
        emiGR_Ga.dtype().is(py::dtype::of<std::complex<float>>()) &&
        Y_GL.dtype().is(py::dtype::of<float>()) &&
        f_GI.dtype().is(py::dtype::of<std::complex<float>>()));

    bool is_double_real = (
        f_Gs.dtype().is(py::dtype::of<double>()) &&
        emiGR_Ga.dtype().is(py::dtype::of<std::complex<double>>()) &&
        Y_GL.dtype().is(py::dtype::of<double>()) &&
        f_GI.dtype().is(py::dtype::of<double>()));

    bool is_float_real = (
        f_Gs.dtype().is(py::dtype::of<float>()) &&
        emiGR_Ga.dtype().is(py::dtype::of<std::complex<float>>()) &&
        Y_GL.dtype().is(py::dtype::of<float>()) &&
        f_GI.dtype().is(py::dtype::of<float>()));

    if (is_double_complex)
    {
        assert(f_GI.shape()[0] == nG);
        _pwlfc_expand_old_complex<double>(
            nG, nJ, nL, natoms, nsplines,
            static_cast<const double*>(f_Gs.data()),
            static_cast<const std::complex<double>*>(emiGR_Ga.data()),
            static_cast<const double*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<std::complex<double>*>(f_GI.mutable_data()));
    }
    else if (is_float_complex)
    {
        assert(f_GI.shape()[0] == nG);
        _pwlfc_expand_old_complex<float>(
            nG, nJ, nL, natoms, nsplines,
            static_cast<const float*>(f_Gs.data()),
            static_cast<const std::complex<float>*>(emiGR_Ga.data()),
            static_cast<const float*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<std::complex<float>*>(f_GI.mutable_data()));
    }
    else if (is_double_real)
    {
        assert(f_GI.shape()[0] == 2 * nG);
        _pwlfc_expand_old_real<double>(
            nG, nJ, nL, natoms, nsplines,
            static_cast<const double*>(f_Gs.data()),
            static_cast<const std::complex<double>*>(emiGR_Ga.data()),
            static_cast<const double*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<double*>(f_GI.mutable_data()),
            f_GI.shape()[1]);
    }
    else if (is_float_real)
    {
        assert(f_GI.shape()[0] == 2 * nG);
        _pwlfc_expand_old_real<float>(
            nG, nJ, nL, natoms, nsplines,
            static_cast<const float*>(f_Gs.data()),
            static_cast<const std::complex<float>*>(emiGR_Ga.data()),
            static_cast<const float*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<float*>(f_GI.mutable_data()),
            f_GI.shape()[1]);
    }
    else
    {
        throw std::invalid_argument(
            "Unsupported dtype combination for pwlfc_expand_old.");
    }
}

void cpp_pwlfc_expand(py::array f_Gs,
                      py::array GK_Gv,
                      py::array pos_av,
                      py::array eikR_a,
                      py::array Y_GL,
                      py::array l_s,
                      py::array a_J,
                      py::array s_J,
                      bool cc,
                      py::array f_GI)
{
    assert(f_Gs.ndim() == 2);
    assert(GK_Gv.ndim() == 2);
    assert(pos_av.ndim() == 2);
    assert(eikR_a.ndim() == 1);
    assert(Y_GL.ndim() == 2);
    assert(l_s.ndim() == 1);
    assert(a_J.ndim() == 1);
    assert(s_J.ndim() == 1);
    assert(f_GI.ndim() == 2);
    assert(GK_Gv.shape()[1] == 3);
    assert(pos_av.shape()[1] == 3);
    assert(is_c_contiguous(f_Gs));
    assert(is_c_contiguous(GK_Gv));
    assert(is_c_contiguous(pos_av));
    assert(is_c_contiguous(eikR_a));
    assert(is_c_contiguous(Y_GL));
    assert(is_c_contiguous(l_s));
    assert(is_c_contiguous(a_J));
    assert(is_c_contiguous(s_J));
    assert(is_c_contiguous(f_GI));
    assert(l_s.dtype().is(py::dtype::of<npy_int32>()));
    assert(a_J.dtype().is(py::dtype::of<npy_int32>()));
    assert(s_J.dtype().is(py::dtype::of<npy_int32>()));

    int nG = GK_Gv.shape()[0];
    int nJ = a_J.shape()[0];
    int nL = Y_GL.shape()[1];
    int nsplines = f_Gs.shape()[1];

    assert(f_Gs.shape()[0] == nG);
    assert(Y_GL.shape()[0] == nG);
    assert(s_J.shape()[0] == nJ);

    bool is_double_complex = (
        f_Gs.dtype().is(py::dtype::of<double>()) &&
        GK_Gv.dtype().is(py::dtype::of<double>()) &&
        pos_av.dtype().is(py::dtype::of<double>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<double>>()) &&
        Y_GL.dtype().is(py::dtype::of<double>()) &&
        f_GI.dtype().is(py::dtype::of<std::complex<double>>()));

    bool is_float_complex = (
        f_Gs.dtype().is(py::dtype::of<float>()) &&
        GK_Gv.dtype().is(py::dtype::of<float>()) &&
        pos_av.dtype().is(py::dtype::of<float>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<float>>()) &&
        Y_GL.dtype().is(py::dtype::of<float>()) &&
        f_GI.dtype().is(py::dtype::of<std::complex<float>>()));

    bool is_double_real = (
        f_Gs.dtype().is(py::dtype::of<double>()) &&
        GK_Gv.dtype().is(py::dtype::of<double>()) &&
        pos_av.dtype().is(py::dtype::of<double>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<double>>()) &&
        Y_GL.dtype().is(py::dtype::of<double>()) &&
        f_GI.dtype().is(py::dtype::of<double>()));

    bool is_float_real = (
        f_Gs.dtype().is(py::dtype::of<float>()) &&
        GK_Gv.dtype().is(py::dtype::of<float>()) &&
        pos_av.dtype().is(py::dtype::of<float>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<float>>()) &&
        Y_GL.dtype().is(py::dtype::of<float>()) &&
        f_GI.dtype().is(py::dtype::of<float>()));

    if (is_double_complex)
    {
        assert(f_GI.shape()[0] == nG);
        _pwlfc_expand_complex<double>(
            nG, nJ, nL, nsplines,
            static_cast<const double*>(f_Gs.data()),
            static_cast<const double*>(GK_Gv.data()),
            static_cast<const double*>(pos_av.data()),
            static_cast<const std::complex<double>*>(eikR_a.data()),
            static_cast<const double*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<std::complex<double>*>(f_GI.mutable_data()));
    }
    else if (is_float_complex)
    {
        assert(f_GI.shape()[0] == nG);
        _pwlfc_expand_complex<float>(
            nG, nJ, nL, nsplines,
            static_cast<const float*>(f_Gs.data()),
            static_cast<const float*>(GK_Gv.data()),
            static_cast<const float*>(pos_av.data()),
            static_cast<const std::complex<float>*>(eikR_a.data()),
            static_cast<const float*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<std::complex<float>*>(f_GI.mutable_data()));
    }
    else if (is_double_real)
    {
        assert(f_GI.shape()[0] == 2 * nG);
        _pwlfc_expand_real<double>(
            nG, nJ, nL, nsplines,
            static_cast<const double*>(f_Gs.data()),
            static_cast<const double*>(GK_Gv.data()),
            static_cast<const double*>(pos_av.data()),
            static_cast<const std::complex<double>*>(eikR_a.data()),
            static_cast<const double*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<double*>(f_GI.mutable_data()),
            f_GI.shape()[1]);
    }
    else if (is_float_real)
    {
        assert(f_GI.shape()[0] == 2 * nG);
        _pwlfc_expand_real<float>(
            nG, nJ, nL, nsplines,
            static_cast<const float*>(f_Gs.data()),
            static_cast<const float*>(GK_Gv.data()),
            static_cast<const float*>(pos_av.data()),
            static_cast<const std::complex<float>*>(eikR_a.data()),
            static_cast<const float*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<float*>(f_GI.mutable_data()),
            f_GI.shape()[1]);
    }
    else
    {
        throw std::invalid_argument(
            "Unsupported dtype combination for pwlfc_expand.");
    }
}

void cpp_pw_precond(py::array G2_G,
                    py::array R_G,
                    double ekin,
                    py::array out_G)
{
    assert(G2_G.ndim() == 1);
    assert(R_G.ndim() == 1);
    assert(out_G.ndim() == 1);
    assert(is_c_contiguous(G2_G));
    assert(is_c_contiguous(R_G));
    assert(is_c_contiguous(out_G));

    int nG = G2_G.shape()[0];
    assert(R_G.shape()[0] == nG);
    assert(out_G.shape()[0] == nG);

    bool is_double = (G2_G.dtype().is(py::dtype::of<double>()) &&
                      R_G.dtype().is(py::dtype::of<std::complex<double>>()) &&
                      out_G.dtype().is(py::dtype::of<std::complex<double>>()));
    bool is_float = (G2_G.dtype().is(py::dtype::of<float>()) &&
                     R_G.dtype().is(py::dtype::of<std::complex<float>>()) &&
                     out_G.dtype().is(py::dtype::of<std::complex<float>>()));

    if (is_double)
    {
        _pw_precond<double>(nG,
            static_cast<const double*>(G2_G.data()),
            static_cast<const std::complex<double>*>(R_G.data()),
            ekin,
            static_cast<std::complex<double>*>(out_G.mutable_data()));
    }
    else if (is_float)
    {
        _pw_precond<float>(nG,
            static_cast<const float*>(G2_G.data()),
            static_cast<const std::complex<float>*>(R_G.data()),
            static_cast<float>(ekin),
            static_cast<std::complex<float>*>(out_G.mutable_data()));
    }
    else
    {
        throw std::invalid_argument(
            "Supports only matching float/complex64 or double/complex128 arrays.");
    }
}

py::array cpp_pawexxvv(py::array M_pp,
                       py::array D_ii)
{
    assert(M_pp.ndim() == 2);
    assert(D_ii.ndim() == 2);
    assert(is_c_contiguous(M_pp));
    assert(is_c_contiguous(D_ii));

    int NI = D_ii.shape()[0];
    int NP = M_pp.shape()[0];
    assert(D_ii.shape()[1] == NI);
    assert(M_pp.shape()[1] == NP);

    bool is_double = (M_pp.dtype().is(py::dtype::of<double>()) &&
                      D_ii.dtype().is(py::dtype::of<double>()));
    bool is_float = (M_pp.dtype().is(py::dtype::of<float>()) &&
                     D_ii.dtype().is(py::dtype::of<float>()));

    if (is_double)
    {
        py::array_t<double> V_ii({NI, NI});
        auto* M = static_cast<const double*>(M_pp.data());
        auto* D = static_cast<const double*>(D_ii.data());
        auto* V = static_cast<double*>(V_ii.mutable_data());

        for (int i1 = 0; i1 < NI; i1++)
        {
            for (int i2 = 0; i2 < NI; i2++)
            {
                double value = 0.0;
                for (int i3 = 0; i3 < NI; i3++)
                {
                    int p13 = (std::min(i3, i1) *
                               (2 * NI - 1 - std::min(i3, i1)) / 2) +
                              std::max(i3, i1);
                    for (int i4 = 0; i4 < NI; i4++)
                    {
                        int p24 = (std::min(i2, i4) *
                                   (2 * NI - 1 - std::min(i2, i4)) / 2) +
                                  std::max(i2, i4);
                        value += M[p13 * NP + p24] * D[i3 * NI + i4];
                    }
                }
                V[i1 * NI + i2] = value;
            }
        }

        return V_ii;
    }
    else if (is_float)
    {
        py::array_t<float> V_ii({NI, NI});
        auto* M = static_cast<const float*>(M_pp.data());
        auto* D = static_cast<const float*>(D_ii.data());
        auto* V = static_cast<float*>(V_ii.mutable_data());

        for (int i1 = 0; i1 < NI; i1++)
        {
            for (int i2 = 0; i2 < NI; i2++)
            {
                float value = 0.0f;
                for (int i3 = 0; i3 < NI; i3++)
                {
                    int p13 = (std::min(i3, i1) *
                               (2 * NI - 1 - std::min(i3, i1)) / 2) +
                              std::max(i3, i1);
                    for (int i4 = 0; i4 < NI; i4++)
                    {
                        int p24 = (std::min(i2, i4) *
                                   (2 * NI - 1 - std::min(i2, i4)) / 2) +
                                  std::max(i2, i4);
                        value += M[p13 * NP + p24] * D[i3 * NI + i4];
                    }
                }
                V[i1 * NI + i2] = value;
            }
        }

        return V_ii;
    }

    throw std::invalid_argument("Supports only matching float32 or float64 arrays.");
}

template <typename T> void _plane_wave_grid(const long* beg,
                const long* end,
                const T* h,
                const T* vk,
                const T* vr0,
                std::complex<T>* pw)
{
    T kr[3];
    T kr0[3];
    int n[3];
    int ij;
    for (int c = 0; c < 3; c++) {
        n[c] = end[c] - beg[c];
        kr0[c] = vk[c] * vr0[c];
    }
    for (int i = 0; i < n[0]; i++) {
        kr[0] = vk[0] * h[0] * (beg[0] + i) - kr0[0];
        for (int j = 0; j < n[1]; j++) {
            kr[1] = kr[0] + vk[1] * h[1] * (beg[1] + j) - kr0[1];
            ij = (i * n[1] + j) * n[2];
            for (int k = 0; k < n[2]; k++) {
                kr[2] = kr[1] + vk[2] * h[2] * (beg[2] + k) - kr0[2];
                pw[ij + k] = std::complex<T>(std::cos(kr[2]), std::sin(kr[2]));
            }
        }
    }
}

void cpp_plane_wave_grid(py::array beg_c,
                         py::array end_c,
                         py::array h_c,
                         py::array k_c,
                         py::array r0_c,
                         py::array pw_g)
{
    assert(beg_c.ndim() == 1);
    assert(end_c.ndim() == 1);
    assert(h_c.ndim() == 1);
    assert(k_c.ndim() == 1);
    assert(r0_c.ndim() == 1);
    assert(pw_g.ndim() == 3);
    assert(beg_c.shape()[0] == 3);
    assert(end_c.shape()[0] == 3);
    assert(h_c.shape()[0] == 3);
    assert(k_c.shape()[0] == 3);
    assert(r0_c.shape()[0] == 3);
    assert(is_c_contiguous(beg_c));
    assert(is_c_contiguous(end_c));
    assert(is_c_contiguous(h_c));
    assert(is_c_contiguous(k_c));
    assert(is_c_contiguous(r0_c));
    assert(is_c_contiguous(pw_g));

    bool is_double = (beg_c.dtype().is(py::dtype::of<long>()) &&
                      end_c.dtype().is(py::dtype::of<long>()) &&
                      h_c.dtype().is(py::dtype::of<double>()) &&
                      k_c.dtype().is(py::dtype::of<double>()) &&
                      r0_c.dtype().is(py::dtype::of<double>()) &&
                      pw_g.dtype().is(py::dtype::of<std::complex<double>>()));
    bool is_float = (beg_c.dtype().is(py::dtype::of<long>()) &&
                     end_c.dtype().is(py::dtype::of<long>()) &&
                     h_c.dtype().is(py::dtype::of<float>()) &&
                     k_c.dtype().is(py::dtype::of<float>()) &&
                     r0_c.dtype().is(py::dtype::of<float>()) &&
                     pw_g.dtype().is(py::dtype::of<std::complex<float>>()));

    if (is_double)
    {
        _plane_wave_grid<double>(static_cast<const long*>(beg_c.data()),
                                 static_cast<const long*>(end_c.data()),
                                 static_cast<const double*>(h_c.data()),
                                 static_cast<const double*>(k_c.data()),
                                 static_cast<const double*>(r0_c.data()),
                                 static_cast<std::complex<double>*>(pw_g.mutable_data()));
    }
    else if (is_float)
    {
        _plane_wave_grid<float>(static_cast<const long*>(beg_c.data()),
                                static_cast<const long*>(end_c.data()),
                                static_cast<const float*>(h_c.data()),
                                static_cast<const float*>(k_c.data()),
                                static_cast<const float*>(r0_c.data()),
                                static_cast<std::complex<float>*>(pw_g.mutable_data()));
    }
    else
    {
        throw std::invalid_argument(
            "Supports long index arrays and matching float/complex64 or double/complex128 arrays.");
    }
}

bool bind_pw_submodule(PyObject* module)
{
    if (!module || module == Py_None)
    {
        return false;
    }

    py::module_ m = py::reinterpret_borrow<py::module_>(module);
    if (m.is_none())
    {
        return false;
    }

    py::module_ submodule = m.def_submodule("pw", "PW specific C++ bindings for GPAW. (EXPERIMENTAL)");

    // Bind Cupy conversion testers (call from Python-side test)

    submodule.def("pw_insert",
        &cpp_pw_insert,
        py::arg("c_G"), py::arg("Q_G"), py::arg("scale"), py::arg("tmp_Q"),
        R"(Docstring for pw_insert.)"
    );
    submodule.def("add_to_density",
        &cpp_add_to_density,
        py::arg("f"), py::arg("psit_R"), py::arg("nt_R"),
        R"(Docstring for add_to_density.)"
    );
#ifndef GPAW_WITHOUT_BLAS
    submodule.def("mmm",
        &cpp_mmm,
        py::arg("alpha"), py::arg("a"), py::arg("opa"),
        py::arg("b"), py::arg("opb"), py::arg("beta"), py::arg("c"),
        R"(Docstring for mmm.)"
    );
    submodule.def("r2k",
        &cpp_r2k,
        py::arg("alpha"), py::arg("a"), py::arg("b"),
        py::arg("beta"), py::arg("c"), py::arg("trans") = "c",
        R"(Docstring for r2k.)"
    );
#endif
    submodule.def("pw_precond",
        &cpp_pw_precond,
        py::arg("G2_G"), py::arg("R_G"), py::arg("ekin"), py::arg("out_G"),
        R"(Docstring for pw_precond.)"
    );
    submodule.def("pwlfc_expand",
        &cpp_pwlfc_expand,
        py::arg("f_Gs"), py::arg("GK_Gv"), py::arg("pos_av"),
        py::arg("eikR_a"), py::arg("Y_GL"), py::arg("l_s"),
        py::arg("a_J"), py::arg("s_J"), py::arg("cc"), py::arg("f_GI"),
        R"(Docstring for pwlfc_expand.)"
    );
    submodule.def("pwlfc_expand_old",
        &cpp_pwlfc_expand_old,
        py::arg("f_Gs"), py::arg("emiGR_Ga"), py::arg("Y_GL"),
        py::arg("l_s"), py::arg("a_J"), py::arg("s_J"),
        py::arg("cc"), py::arg("f_GI"),
        R"(Docstring for pwlfc_expand_old.)"
    );
    submodule.def("pawexxvv",
        &cpp_pawexxvv,
        py::arg("M_pp"), py::arg("D_ii"),
        R"(Docstring for pawexxvv.)"
    );
    submodule.def("plane_wave_grid",
        &cpp_plane_wave_grid,
        py::arg("beg_c"), py::arg("end_c"), py::arg("h_c"),
        py::arg("k_c"), py::arg("r0_c"), py::arg("pw_g"),
        R"(Docstring for plane_wave_grid.)"
    );

    return true;
}
