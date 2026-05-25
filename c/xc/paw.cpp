#include "python_utils.h"
#include "extensions.h"
#include "gpaw_utils.h"

#include <array>
#include <climits>
#include <stdexcept>


#ifndef GPAW_WITHOUT_BLAS
#  ifdef GPAW_NO_UNDERSCORE_BLAS
#    define dgemm_ dgemm
#  endif

CLINKAGE_BEGIN
void dgemm_(const char* transa, const char* transb, int *m, int * n,
            int *k, double *alpha, double *a, int *lda,
            double *b, int *ldb, double *beta,
            double *c, int *ldc);
CLINKAGE_END
#endif


static inline int idx3(int i, int j, int k, int nj, int nk)
{
    return (i * nj + j) * nk + k;
}


static inline int idx4(int i, int j, int k, int l, int nj, int nk, int nl)
{
    return ((i * nj + j) * nk + k) * nl + l;
}


static void row_dgemm(char transa,
                      char transb,
                      int m,
                      int n,
                      int k,
                      double alpha,
                      const double* a,
                      int lda,
                      const double* b,
                      int ldb,
                      double beta,
                      double* c,
                      int ldc)
{
#ifndef GPAW_WITHOUT_BLAS
    int fm = n;
    int fn = m;
    int fk = k;
    int fa = ldb;
    int fb = lda;
    int fc = ldc;
    if (transb == 'N' || transb == 'n')
        fa = MAX(fa, n);
    else
        fa = MAX(fa, k);
    if (transa == 'N' || transa == 'n')
        fb = MAX(fb, k);
    else
        fb = MAX(fb, m);
    fc = MAX(fc, n);

    dgemm_(&transb, &transa, &fm, &fn, &fk,
           &alpha, (double*)b, &fa, (double*)a, &fb, &beta, c, &fc);
#else
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double value = 0.0;
            for (int l = 0; l < k; l++) {
                double avalue;
                double bvalue;
                if (transa == 'N' || transa == 'n')
                    avalue = a[i * lda + l];
                else
                    avalue = a[l * lda + i];
                if (transb == 'N' || transb == 'n')
                    bvalue = b[l * ldb + j];
                else
                    bvalue = b[j * ldb + l];
                value += avalue * bvalue;
            }
            c[i * ldc + j] = alpha * value + beta * c[i * ldc + j];
        }
    }
#endif
}


template<int N, int M = 1>
class SelectedDim;


template<int N>
class DoubleArray {
public:
    DoubleArray(PyObject* object)
        : array_(nullptr), data_(nullptr), c_contiguous_(0)
    {
        for (int i = 0; i < N; i++) {
            dims_[i] = 1;
            strides_[i] = 0;
        }

        if (!PyArray_Check(object)) {
            PyErr_SetString(PyExc_TypeError, "expected a NumPy array");
            return;
        }

        PyArrayObject* array = (PyArrayObject*)object;
        if (PyArray_TYPE(array) != NPY_DOUBLE) {
            PyErr_SetString(PyExc_ValueError, "expected a float64 array");
            return;
        }

        array_ = array;
        data_ = DOUBLEP(array);
        if (PyArray_NDIM(array_) != N) {
            PyErr_SetString(PyExc_ValueError,
                            "unexpected number of array dimensions");
            array_ = nullptr;
            data_ = nullptr;
            return;
        }
        c_contiguous_ = PyArray_ISCARRAY(array_);
        for (int i = 0; i < N; i++)
            dims_[i] = PyArray_DIM(array_, i);
        for (int i = 0; i < N; i++)
            strides_[i] = PyArray_STRIDE(array_, i);
    }

    DoubleArray(double* data,
                const npy_intp* dims,
                const npy_intp* strides)
        : array_(nullptr), data_(data), c_contiguous_(1)
    {
        for (int i = 0; i < N; i++) {
            dims_[i] = dims[i];
            strides_[i] = strides[i];
        }
        npy_intp expected = sizeof(double);
        for (int i = N - 1; i >= 0; i--) {
            if (strides_[i] != expected)
                c_contiguous_ = 0;
            expected *= dims_[i];
        }
    }

    bool ok() const
    {
        return data_ != nullptr;
    }

    int ndim() const
    {
        return N;
    }

    int dim(int axis) const
    {
        return dims_[axis];
    }

    npy_intp stride(int axis) const
    {
        return strides_[axis] / sizeof(double);
    }

    npy_intp byte_stride(int axis) const
    {
        return strides_[axis];
    }

    int c_contiguous() const
    {
        return c_contiguous_;
    }

    npy_intp size() const
    {
        npy_intp result = 1;
        for (int i = 0; i < N; i++)
            result *= dims_[i];
        return result;
    }

    double* data() const
    {
        return data_;
    }

    static DoubleArray<N> view(double* data,
                               const std::array<npy_intp, N>& dims,
                               const std::array<npy_intp, N>& strides)
    {
        std::array<npy_intp, N> byte_strides;
        for (int i = 0; i < N; i++)
            byte_strides[i] = strides[i] * (npy_intp)sizeof(double);
        return DoubleArray<N>(data, dims.data(), byte_strides.data());
    }

    template<int M>
    DoubleArray<M> view(npy_intp offset,
                        const std::array<npy_intp, M>& dims,
                        const std::array<npy_intp, M>& strides) const
    {
        return DoubleArray<M>::view(data_ + offset, dims, strides);
    }

    DoubleArray<N> transpose() const
    {
        static_assert(N == 2, "transpose view is implemented for 2D arrays");
        return DoubleArray<N>::view(data_,
                                    {dim(1), dim(0)},
                                    {stride(1), stride(0)});
    }

    void matmul(const DoubleArray& a,
                const DoubleArray& b,
                double alpha = 1.0,
                double beta = 0.0) const
    {
        static_assert(N == 2, "matmul output must be a 2D array");
        if (a.dim(1) != b.dim(0) ||
            dim(0) != a.dim(0) ||
            dim(1) != b.dim(1))
            throw std::invalid_argument("matmul dimension mismatch");

        const char transa = a.blas_transpose();
        const char transb = b.blas_transpose();
        const npy_intp lda = a.blas_leading_dimension(transa);
        const npy_intp ldb = b.blas_leading_dimension(transb);
        if (stride(1) != 1)
            throw std::invalid_argument("matmul output must be row-major");
        if (stride(0) < MAX(1, dim(1)))
            throw std::invalid_argument("matmul output row stride is too small");
        if (lda > INT_MAX || ldb > INT_MAX || stride(0) > INT_MAX)
            throw std::invalid_argument("matmul row stride is too large");

        row_dgemm(transa, transb, a.dim(0), b.dim(1), a.dim(1),
                  alpha, a.data(), (int)lda,
                  b.data(), (int)ldb,
                  beta, data_, (int)stride(0));
    }

    void add_scaled(const DoubleArray<N>& source, double scale) const
    {
        for (int i = 0; i < N; i++)
            if (dim(i) != source.dim(i))
                throw std::invalid_argument("add_scaled dimension mismatch");
        if (!c_contiguous() || !source.c_contiguous())
            throw std::invalid_argument("add_scaled requires contiguous arrays");

        const npy_intp n = size();
        for (npy_intp i = 0; i < n; i++)
            data_[i] += scale * source.data_[i];
    }

    void transpose_from(const DoubleArray& source,
                        int i0,
                        int i1 = -1,
                        int i2 = -1) const
    {
        if (source.ndim() != ndim())
            throw std::invalid_argument("transpose rank mismatch");

        if (ndim() == 1) {
            if (i0 != 0 || dim(0) != source.dim(0))
                throw std::invalid_argument("invalid 1D transpose");
            for (int a = 0; a < dim(0); a++)
                data_[a * stride(0)] = source.data_[a * source.stride(0)];
            return;
        }

        if (ndim() == 2) {
            if (i0 < 0 || i0 > 1 || i1 < 0 || i1 > 1 || i0 == i1 ||
                dim(0) != source.dim(i0) ||
                dim(1) != source.dim(i1))
                throw std::invalid_argument("invalid 2D transpose");
            for (int a = 0; a < dim(0); a++) {
                for (int b = 0; b < dim(1); b++) {
                    int source_indices[2];
                    source_indices[i0] = a;
                    source_indices[i1] = b;
                    data_[a * stride(0) + b * stride(1)] =
                        source.data_[source_indices[0] * source.stride(0) +
                                     source_indices[1] * source.stride(1)];
                }
            }
            return;
        }

        if (ndim() == 3) {
            if (i0 < 0 || i0 > 2 || i1 < 0 || i1 > 2 ||
                i2 < 0 || i2 > 2 || i0 == i1 || i0 == i2 || i1 == i2 ||
                dim(0) != source.dim(i0) ||
                dim(1) != source.dim(i1) ||
                dim(2) != source.dim(i2))
                throw std::invalid_argument("invalid 3D transpose");
            for (int a = 0; a < dim(0); a++) {
                for (int b = 0; b < dim(1); b++) {
                    for (int c = 0; c < dim(2); c++) {
                        int source_indices[3];
                        source_indices[i0] = a;
                        source_indices[i1] = b;
                        source_indices[i2] = c;
                        data_[a * stride(0) + b * stride(1) +
                              c * stride(2)] =
                            source.data_[
                                source_indices[0] * source.stride(0) +
                                source_indices[1] * source.stride(1) +
                                source_indices[2] * source.stride(2)];
                    }
                }
            }
            return;
        }

        throw std::invalid_argument("unsupported transpose rank");
    }

    template<typename... Dims>
    SelectedDim<N, 1 + sizeof...(Dims)> select(int dim,
                                               Dims... dims) const;

private:
    PyArrayObject* array_;
    double* data_;
    npy_intp dims_[N];
    npy_intp strides_[N];
    int c_contiguous_;

    char blas_transpose() const
    {
        static_assert(N == 2, "BLAS layout is implemented for 2D arrays");
        if (stride(1) == 1)
            return 'N';
        if (stride(0) == 1)
            return 'T';
        throw std::invalid_argument("matmul requires row-major arrays or "
                                    "their transposed views");
    }

    npy_intp blas_leading_dimension(char transpose) const
    {
        static_assert(N == 2, "BLAS layout is implemented for 2D arrays");
        if (transpose == 'N') {
            if (stride(0) < MAX(1, dim(1)))
                throw std::invalid_argument(
                    "matmul left/right row stride is too small");
            return stride(0);
        }

        if (stride(1) < MAX(1, dim(0)))
            throw std::invalid_argument(
                "matmul transposed row stride is too small");
        return stride(1);
    }

    template<int ON, int CM>
    friend class SelectedDim;
};


template<int N, int M>
class SelectedDim {
public:
    SelectedDim(const DoubleArray<N>& array, const std::array<int, M>& dims)
        : array_(array), dims_(dims)
    {
    }

    const DoubleArray<N>& array() const
    {
        return array_;
    }

    int dim() const
    {
        static_assert(M == 1, "dim() is only valid for one selected axis");
        return dims_[0];
    }

    int dim(int axis) const
    {
        return dims_[axis];
    }

private:
    const DoubleArray<N> array_;
    const std::array<int, M> dims_;
};


template<int N>
template<typename... Dims>
SelectedDim<N, 1 + sizeof...(Dims)>
DoubleArray<N>::select(int dim, Dims... dims) const
{
    return SelectedDim<N, 1 + sizeof...(Dims)>(*this, {dim, dims...});
}


void contract(const SelectedDim<2, 1>& left,
              const SelectedDim<2, 1>& right,
              const DoubleArray<2>& out)
{
    const DoubleArray<2>& a = left.array();
    const DoubleArray<2>& b = right.array();

    if (left.dim() != 1 || right.dim() != 0)
        throw std::invalid_argument("unsupported contraction dimensions");
    if (a.dim(1) != b.dim(0) ||
        out.dim(0) != a.dim(0) ||
        out.dim(1) != b.dim(1))
        throw std::invalid_argument("contract dimension mismatch");

    out.matmul(a, b);
}


void contract(const SelectedDim<2, 1>& left,
              const SelectedDim<3, 1>& right,
              const DoubleArray<3>& out)
{
    const DoubleArray<2>& a = left.array();
    const DoubleArray<3>& b = right.array();

    if (left.dim() != 1 || (right.dim() != 1 && right.dim() != 0))
        throw std::invalid_argument(
            "unsupported 2D-3D contraction dimensions");

    if (right.dim() == 1) {
        if (a.dim(1) != b.dim(1) ||
            out.dim(0) != a.dim(0) ||
            out.dim(1) != b.dim(0) ||
            out.dim(2) != b.dim(2))
            throw std::invalid_argument("2D-3D contract dimension mismatch");

        for (int i = 0; i < b.dim(0); i++) {
            const DoubleArray<2> b_i =
                b.view<2>(i * b.stride(0),
                          {b.dim(1), b.dim(2)},
                          {b.stride(1), b.stride(2)});
            const DoubleArray<2> out_i =
                out.view<2>(i * out.stride(1),
                            {out.dim(0), out.dim(2)},
                            {out.stride(0), out.stride(2)});
            out_i.matmul(a, b_i);
        }
        return;
    }

    if (a.dim(1) != b.dim(0) ||
        out.dim(0) != a.dim(0) ||
        out.dim(1) != b.dim(1) ||
        out.dim(2) != b.dim(2))
        throw std::invalid_argument("2D-3D contract dimension mismatch");
    if (b.stride(2) != 1 || b.stride(1) != b.dim(2) ||
        out.stride(2) != 1 || out.stride(1) != out.dim(2))
        throw std::invalid_argument(
            "2D-3D contract requires contiguous trailing dimensions");

    const DoubleArray<2> b_pqL =
        b.view<2>(0,
                  {b.dim(0), b.dim(1) * b.dim(2)},
                  {b.stride(0), 1});
    const DoubleArray<2> out_sqL =
        out.view<2>(0,
                    {out.dim(0), out.dim(1) * out.dim(2)},
                    {out.stride(0), 1});
    out_sqL.matmul(a, b_pqL);
}


void contract(const SelectedDim<3, 1>& left,
              const SelectedDim<3, 1>& right,
              const DoubleArray<4>& out)
{
    const DoubleArray<3>& a = left.array();
    const DoubleArray<3>& b = right.array();

    if (left.dim() != 2 || right.dim() != 1)
        throw std::invalid_argument("unsupported 3D-3D contraction dimensions");
    if (a.dim(2) != b.dim(1) ||
        out.dim(0) != a.dim(0) ||
        out.dim(1) != a.dim(1) ||
        out.dim(2) != b.dim(0) ||
        out.dim(3) != b.dim(2))
        throw std::invalid_argument("3D-3D contract dimension mismatch");

    for (int i = 0; i < a.dim(0); i++) {
        const DoubleArray<2> a_i =
            a.view<2>(i * a.stride(0),
                      {a.dim(1), a.dim(2)},
                      {a.stride(1), a.stride(2)});
        for (int j = 0; j < b.dim(0); j++) {
            const DoubleArray<2> b_j =
                b.view<2>(j * b.stride(0),
                          {b.dim(1), b.dim(2)},
                          {b.stride(1), b.stride(2)});
            const DoubleArray<2> out_ij =
                out.view<2>(i * out.stride(0) + j * out.stride(2),
                            {out.dim(1), out.dim(3)},
                            {out.stride(1), out.stride(3)});
            out_ij.matmul(a_i, b_j);
        }
    }
}


double contract(const SelectedDim<1, 1>& left,
                const SelectedDim<1, 1>& middle,
                const SelectedDim<2, 2>& right)
{
    const DoubleArray<1>& a = left.array();
    const DoubleArray<1>& b = middle.array();
    const DoubleArray<2>& c = right.array();

    if (left.dim() != 0 || middle.dim() != 0 ||
        right.dim(0) != 0 || right.dim(1) != 1)
        throw std::invalid_argument("unsupported scalar contraction dimensions");
    if (a.dim(0) != c.dim(0) || b.dim(0) != c.dim(1))
        throw std::invalid_argument("scalar contract dimension mismatch");

    double value = 0.0;
    for (int i = 0; i < c.dim(0); i++) {
        double row_value = 0.0;
        for (int j = 0; j < c.dim(1); j++)
            row_value += b.data()[j * b.stride(0)] *
                c.data()[i * c.stride(0) + j * c.stride(1)];
        value += a.data()[i * a.stride(0)] * row_value;
    }
    return value;
}


template<int N>
static int require_array(const DoubleArray<N>& array, int ndim)
{
    if (!array.ok())
        return 0;
    if (N != ndim) {
        PyErr_SetString(PyExc_ValueError,
                        "unexpected number of array dimensions");
        return 0;
    }
    if (!array.c_contiguous()) {
        PyErr_SetString(PyExc_ValueError,
                        "expected C-contiguous arrays");
        return 0;
    }
    return 1;
}


CLINKAGE PyObject* test_paw_double_array_matmul(PyObject* self,
                                                PyObject* args)
{
    PyObject* a_obj;
    PyObject* b_obj;
    PyObject* out_obj;

    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &out_obj))
        return NULL;

    DoubleArray<2> a(a_obj);
    DoubleArray<2> b(b_obj);
    DoubleArray<2> out(out_obj);
    if (!a.ok() || !b.ok() || !out.ok())
        return NULL;

    try {
        out.matmul(a, b);
    }
    catch (const std::exception& err) {
        PyErr_SetString(PyExc_ValueError, err.what());
        return NULL;
    }

    Py_RETURN_NONE;
}


static void derivative_term(const double* dr_g,
                            const double* in_g,
                            double* out_g,
                            int ngrids)
{
    out_g[0] = (in_g[1] - in_g[0]) / dr_g[0];
    for (int g = 1; g < ngrids - 1; g++)
        out_g[g] = 0.5 * (in_g[g + 1] - in_g[g - 1]) / dr_g[g];
    out_g[ngrids - 1] = (in_g[ngrids - 1] - in_g[ngrids - 2]) /
        dr_g[ngrids - 1];
}


static void derivative2_term(const double* dr_g,
                             const double* dv_g,
                             const double* dedsigma_g,
                             const double* a_g,
                             double scale,
                             double* out_g,
                             int ngrids,
                             int add)
{
    double c0 = scale * dv_g[0] * dedsigma_g[0] * a_g[0] / dr_g[0];
    double c1 = scale * dv_g[1] * dedsigma_g[1] * a_g[1] / dr_g[1];

    if (add)
        out_g[0] += 0.5 * c1 + c0;
    else
        out_g[0] = 0.5 * c1 + c0;

    for (int g = 1; g < ngrids - 2; g++) {
        double cm1 = scale * dv_g[g - 1] * dedsigma_g[g - 1] *
            a_g[g - 1] / dr_g[g - 1];
        double cp1 = scale * dv_g[g + 1] * dedsigma_g[g + 1] *
            a_g[g + 1] / dr_g[g + 1];
        double value = 0.5 * (cp1 - cm1);
        if (add)
            out_g[g] += value;
        else
            out_g[g] = value;
    }

    double cm3 = scale * dv_g[ngrids - 3] * dedsigma_g[ngrids - 3] *
        a_g[ngrids - 3] / dr_g[ngrids - 3];
    double cm2 = scale * dv_g[ngrids - 2] * dedsigma_g[ngrids - 2] *
        a_g[ngrids - 2] / dr_g[ngrids - 2];
    double cm1 = scale * dv_g[ngrids - 1] * dedsigma_g[ngrids - 1] *
        a_g[ngrids - 1] / dr_g[ngrids - 1];

    if (add)
        out_g[ngrids - 2] += cm1 - 0.5 * cm3;
    else
        out_g[ngrids - 2] = cm1 - 0.5 * cm3;

    if (add)
        out_g[ngrids - 1] += -cm1 - 0.5 * cm2;
    else
        out_g[ngrids - 1] = -cm1 - 0.5 * cm2;
}


static DoubleArray<3> spin_middle_view(const DoubleArray<3>& array)
{
    return array.view<3>(0,
                         {array.dim(1), array.dim(0), array.dim(2)},
                         {array.stride(1), array.stride(0),
                          array.stride(2)});
}


static void project_density_and_derivative(const DoubleArray<3>& n_sLg,
                                           const DoubleArray<3>& dndr_sLg,
                                           const DoubleArray<2>& Y_nL,
                                           const DoubleArray<3>& n_sng,
                                           const DoubleArray<3>& a_sng)
{
    const DoubleArray<3> n_nsg = spin_middle_view(n_sng);
    const DoubleArray<3> a_nsg = spin_middle_view(a_sng);

    contract(Y_nL.select(1), n_sLg.select(1), n_nsg);
    contract(Y_nL.select(1), dndr_sLg.select(1), a_nsg);
}


static void transpose_angular_gradient(const DoubleArray<3>& rnablaY_nLv,
                                       const DoubleArray<3>& rnablaY_vnL)
{
    rnablaY_vnL.transpose_from(rnablaY_nLv, 2, 0, 1);
}


static void project_angular_gradient(const DoubleArray<3>& n_sLg,
                                     const DoubleArray<3>& rnablaY_vnL,
                                     const DoubleArray<4>& b_vnsg)
{
    contract(rnablaY_vnL.select(2), n_sLg.select(1), b_vnsg);
}


static void calculate_angular_sigma(const DoubleArray<4>& b_vnsg,
                                    const DoubleArray<3>& sigma_xng)
{
    const int nspins = b_vnsg.dim(2);
    const int nlebedev = b_vnsg.dim(1);
    const int ngrids = b_vnsg.dim(3);

    for (int n = 0; n < nlebedev; n++) {
        for (int g = 0; g < ngrids; g++) {
            for (int s = 0; s < nspins; s++) {
                double sigma = 0.0;
                for (int v = 0; v < 3; v++) {
                    const double b =
                        b_vnsg.data()[v * b_vnsg.stride(0) +
                                      n * b_vnsg.stride(1) +
                                      s * b_vnsg.stride(2) +
                                      g * b_vnsg.stride(3)];
                    sigma += b * b;
                }
                sigma_xng.data()[(2 * s) * sigma_xng.stride(0) +
                                 n * sigma_xng.stride(1) +
                                 g * sigma_xng.stride(2)] = sigma;
            }
            if (nspins == 2) {
                double sigma = 0.0;
                for (int v = 0; v < 3; v++) {
                    sigma +=
                        b_vnsg.data()[v * b_vnsg.stride(0) +
                                      n * b_vnsg.stride(1) +
                                      g * b_vnsg.stride(3)] *
                        b_vnsg.data()[v * b_vnsg.stride(0) +
                                      n * b_vnsg.stride(1) +
                                      b_vnsg.stride(2) +
                                      g * b_vnsg.stride(3)];
                }
                sigma_xng.data()[sigma_xng.stride(0) +
                                 n * sigma_xng.stride(1) +
                                 g * sigma_xng.stride(2)] = sigma;
            }
        }
    }
}


static void scale_angular_sigma(const DoubleArray<1>& r_g,
                                const DoubleArray<3>& sigma_xng)
{
    for (int n = 0; n < sigma_xng.dim(1); n++) {
        for (int x = 0; x < sigma_xng.dim(0); x++) {
            for (int g = 1; g < sigma_xng.dim(2); g++) {
                const double r = r_g.data()[g * r_g.stride(0)];
                sigma_xng.data()[x * sigma_xng.stride(0) +
                                 n * sigma_xng.stride(1) +
                                 g * sigma_xng.stride(2)] /= r * r;
            }
            sigma_xng.data()[x * sigma_xng.stride(0) +
                             n * sigma_xng.stride(1)] =
                sigma_xng.data()[x * sigma_xng.stride(0) +
                                 n * sigma_xng.stride(1) +
                                 sigma_xng.stride(2)];
        }
    }
}


static void add_radial_sigma(const DoubleArray<3>& a_sng,
                             const DoubleArray<3>& sigma_xng)
{
    const int nspins = a_sng.dim(0);

    for (int n = 0; n < a_sng.dim(1); n++) {
        for (int s = 0; s < nspins; s++) {
            for (int g = 0; g < a_sng.dim(2); g++) {
                const double a = a_sng.data()[s * a_sng.stride(0) +
                                              n * a_sng.stride(1) +
                                              g * a_sng.stride(2)];
                sigma_xng.data()[(2 * s) * sigma_xng.stride(0) +
                                 n * sigma_xng.stride(1) +
                                 g * sigma_xng.stride(2)] += a * a;
            }
        }
        if (nspins == 2) {
            for (int g = 0; g < a_sng.dim(2); g++) {
                sigma_xng.data()[sigma_xng.stride(0) +
                                 n * sigma_xng.stride(1) +
                                 g * sigma_xng.stride(2)] +=
                    a_sng.data()[n * a_sng.stride(1) +
                                 g * a_sng.stride(2)] *
                    a_sng.data()[a_sng.stride(0) +
                                 n * a_sng.stride(1) +
                                 g * a_sng.stride(2)];
            }
        }
    }
}


static void project_density_compute(const DoubleArray<3>& n_sLg,
                                    const DoubleArray<3>& dndr_sLg,
                                    const DoubleArray<2>& Y_nL,
                                    const DoubleArray<3>& rnablaY_nLv,
                                    const DoubleArray<1>& r_g,
                                    const DoubleArray<3>& n_sng,
                                    const DoubleArray<3>& a_sng,
                                    const DoubleArray<4>& b_vnsg,
                                    const DoubleArray<3>& sigma_xng,
                                    const DoubleArray<3>& rnablaY_vnL)
{
    project_density_and_derivative(n_sLg, dndr_sLg, Y_nL, n_sng, a_sng);
    transpose_angular_gradient(rnablaY_nLv, rnablaY_vnL);
    project_angular_gradient(n_sLg, rnablaY_vnL, b_vnsg);
    calculate_angular_sigma(b_vnsg, sigma_xng);
    scale_angular_sigma(r_g, sigma_xng);
    add_radial_sigma(a_sng, sigma_xng);
}


static void radial_gradient_correction_compute(const double* dr_g,
                                               const double* dv_g,
                                               const double* dedsigma_xng,
                                               const double* a_sng,
                                               double* vv_sng,
                                               int nspins,
                                               int nlebedev,
                                               int ngrids)
{
    for (int s = 0; s < nspins; s++) {
        for (int n = 0; n < nlebedev; n++) {
            const double* dedsigma_g =
                dedsigma_xng + (2 * s * nlebedev + n) * ngrids;
            const double* a_g = a_sng + (s * nlebedev + n) * ngrids;
            double* vv_g = vv_sng + (s * nlebedev + n) * ngrids;
            derivative2_term(dr_g, dv_g, dedsigma_g, a_g, -2.0,
                             vv_g, ngrids, 0);
        }
    }

    if (nspins == 2) {
        for (int n = 0; n < nlebedev; n++) {
            const double* dedsigma_g = dedsigma_xng + nlebedev * ngrids +
                n * ngrids;
            const double* a1_g = a_sng + (nlebedev + n) * ngrids;
            const double* a0_g = a_sng + n * ngrids;
            double* vv0_g = vv_sng + n * ngrids;
            double* vv1_g = vv_sng + (nlebedev + n) * ngrids;
            derivative2_term(dr_g, dv_g, dedsigma_g, a1_g, -1.0,
                             vv0_g, ngrids, 1);
            derivative2_term(dr_g, dv_g, dedsigma_g, a0_g, -1.0,
                             vv1_g, ngrids, 1);
        }
    }

    for (int s = 0; s < nspins; s++) {
        for (int n = 0; n < nlebedev; n++) {
            double* vv_g = vv_sng + (s * nlebedev + n) * ngrids;
            for (int g = 1; g < ngrids; g++)
                vv_g[g] /= dv_g[g];
            vv_g[0] = vv_g[1];
        }
    }
}


static void back_project_density_compute(double* dEdD_sqL,
                                         const double* dv_g,
                                         const double* dedn_sng,
                                         const double* vv_sng,
                                         const double* n_qg,
                                         const double* weight_n,
                                         const double* Y_nL,
                                         int nspins,
                                         int nq,
                                         int Lmax,
                                         int nlebedev,
                                         int ngrids)
{
    double* work_ng = (double*)malloc(
        (size_t)nlebedev * ngrids * sizeof(double));
    double* work_nq = (double*)malloc(
        (size_t)nlebedev * nq * sizeof(double));
    double* weighted_Y_nL = (double*)malloc(
        (size_t)nlebedev * Lmax * sizeof(double));
    if (work_ng != NULL && work_nq != NULL && weighted_Y_nL != NULL) {
        for (int n = 0; n < nlebedev; n++) {
            for (int L = 0; L < Lmax; L++)
                weighted_Y_nL[n * Lmax + L] =
                    weight_n[n] * Y_nL[n * Lmax + L];
        }

        for (int s = 0; s < nspins; s++) {
            for (int n = 0; n < nlebedev; n++) {
                for (int g = 0; g < ngrids; g++) {
                    double dedn =
                        dedn_sng[idx3(s, n, g, nlebedev, ngrids)];
                    if (vv_sng != NULL)
                        dedn += vv_sng[idx3(s, n, g, nlebedev, ngrids)];
                    work_ng[n * ngrids + g] = dv_g[g] * dedn;
                }
            }
            row_dgemm('N', 'T', nlebedev, nq, ngrids,
                      1.0, work_ng, ngrids,
                      n_qg, ngrids,
                      0.0, work_nq, nq);
            row_dgemm('T', 'N', nq, Lmax, nlebedev,
                      1.0, work_nq, nq,
                      weighted_Y_nL, Lmax,
                      1.0, dEdD_sqL + idx3(s, 0, 0, nq, Lmax),
                      Lmax);
        }
        free(work_ng);
        free(work_nq);
        free(weighted_Y_nL);
        return;
    }
    free(work_ng);
    free(work_nq);
    free(weighted_Y_nL);

    for (int s = 0; s < nspins; s++) {
        for (int q = 0; q < nq; q++) {
            for (int L = 0; L < Lmax; L++) {
                double sum = 0.0;
                for (int n = 0; n < nlebedev; n++) {
                    const double wY = weight_n[n] * Y_nL[n * Lmax + L];
                    for (int g = 0; g < ngrids; g++) {
                        double dedn = dedn_sng[idx3(s, n, g,
                                                     nlebedev, ngrids)];
                        if (vv_sng != NULL)
                            dedn += vv_sng[idx3(s, n, g,
                                                nlebedev, ngrids)];
                        sum += dv_g[g] * dedn * n_qg[q * ngrids + g] * wY;
                    }
                }
                dEdD_sqL[idx3(s, q, L, nq, Lmax)] += sum;
            }
        }
    }
}


static void back_project_gradient_compute(double* dEdD_sqL,
                                          const double* dedsigma_xng,
                                          const double* b_vnsg,
                                          const double* n_qg,
                                          const double* dr_g,
                                          const double* weight_n,
                                          const double* rnablaY_nLv,
                                          int nspins,
                                          int nq,
                                          int Lmax,
                                          int nlebedev,
                                          int ngrids)
{
    const double prefactor = 8.0 * M_PI;
    double* work_ng = (double*)malloc(
        (size_t)nlebedev * ngrids * sizeof(double));
    double* work_nq = (double*)malloc(
        (size_t)nlebedev * nq * sizeof(double));
    double* weighted_R_nL = (double*)malloc(
        (size_t)nlebedev * Lmax * sizeof(double));
    if (work_ng != NULL && work_nq != NULL && weighted_R_nL != NULL) {
        for (int s = 0; s < nspins; s++) {
            for (int v = 0; v < 3; v++) {
                for (int n = 0; n < nlebedev; n++) {
                    for (int L = 0; L < Lmax; L++) {
                        weighted_R_nL[n * Lmax + L] = weight_n[n] *
                            rnablaY_nLv[idx3(n, L, v, Lmax, 3)];
                    }
                    for (int g = 0; g < ngrids; g++) {
                        double value =
                            dedsigma_xng[idx3(2 * s, n, g,
                                              nlebedev, ngrids)] *
                            b_vnsg[idx4(v, n, s, g,
                                        nlebedev, nspins, ngrids)];
                        if (nspins == 2) {
                            value += 0.5 *
                                dedsigma_xng[idx3(1, n, g,
                                                  nlebedev, ngrids)] *
                                b_vnsg[idx4(v, n, 1 - s, g,
                                            nlebedev, nspins, ngrids)];
                        }
                        const double radial_scale = dr_g == NULL ?
                            1.0 : dr_g[g];
                        work_ng[n * ngrids + g] = radial_scale * value;
                    }
                }
                row_dgemm('N', 'T', nlebedev, nq, ngrids,
                          1.0, work_ng, ngrids,
                          n_qg, ngrids,
                          0.0, work_nq, nq);
                row_dgemm('T', 'N', nq, Lmax, nlebedev,
                          prefactor, work_nq, nq,
                          weighted_R_nL, Lmax,
                          1.0, dEdD_sqL + idx3(s, 0, 0, nq, Lmax),
                          Lmax);
            }
        }
        free(work_ng);
        free(work_nq);
        free(weighted_R_nL);
        return;
    }
    free(work_ng);
    free(work_nq);
    free(weighted_R_nL);

    for (int s = 0; s < nspins; s++) {
        for (int q = 0; q < nq; q++) {
            for (int L = 0; L < Lmax; L++) {
                double sum = 0.0;
                for (int n = 0; n < nlebedev; n++) {
                    const double weight = weight_n[n];
                    for (int v = 0; v < 3; v++) {
                        double B = 0.0;
                        for (int g = 0; g < ngrids; g++) {
                            double value =
                                dedsigma_xng[idx3(2 * s, n, g,
                                                  nlebedev, ngrids)] *
                                b_vnsg[idx4(v, n, s, g,
                                            nlebedev, nspins, ngrids)];
                            if (nspins == 2) {
                                value += 0.5 *
                                    dedsigma_xng[idx3(1, n, g,
                                                      nlebedev, ngrids)] *
                                    b_vnsg[idx4(v, n, 1 - s, g,
                                                nlebedev, nspins, ngrids)];
                            }
                            const double radial_scale = dr_g == NULL ?
                                1.0 : dr_g[g];
                            B += value * radial_scale * n_qg[q * ngrids + g];
                        }
                        sum += weight *
                            rnablaY_nLv[idx3(n, L, v, Lmax, 3)] * B;
                    }
                }
                dEdD_sqL[idx3(s, q, L, nq, Lmax)] += prefactor * sum;
            }
        }
    }
}


static double integrate_energy_compute(const DoubleArray<2>& e_ng,
                                       const DoubleArray<1>& dv_g,
                                       const DoubleArray<1>& weight_n)
{
    return contract(weight_n.select(0), dv_g.select(0), e_ng.select(0, 1));
}


static int expansion_shape_check(const DoubleArray<3>& dEdD_sqL,
                                 const DoubleArray<2>& e_ng,
                                 const DoubleArray<3>& dedn_sng,
                                 const DoubleArray<3>& dedsigma_xng,
                                 const DoubleArray<3>& a_sng,
                                 const DoubleArray<4>& b_vnsg,
                                 const DoubleArray<2>& n_qg,
                                 const DoubleArray<1>& dr_g,
                                 const DoubleArray<1>& dv_g,
                                 const DoubleArray<1>& weight_n,
                                 const DoubleArray<2>& Y_nL,
                                 const DoubleArray<3>& rnablaY_nLv)
{
    const int nspins = dEdD_sqL.dim(0);
    const int nq = dEdD_sqL.dim(1);
    const int Lmax = dEdD_sqL.dim(2);
    const int nlebedev = e_ng.dim(0);
    const int ngrids = e_ng.dim(1);

    if (nspins != 1 && nspins != 2) {
        PyErr_SetString(PyExc_ValueError, "spin dimension must be 1 or 2");
        return 0;
    }
    if (dedn_sng.dim(0) != nspins ||
        dedn_sng.dim(1) != nlebedev ||
        dedn_sng.dim(2) != ngrids ||
        dedsigma_xng.dim(0) != 2 * nspins - 1 ||
        dedsigma_xng.dim(1) != nlebedev ||
        dedsigma_xng.dim(2) != ngrids ||
        a_sng.dim(0) != nspins ||
        a_sng.dim(1) != nlebedev ||
        a_sng.dim(2) != ngrids ||
        b_vnsg.dim(0) != 3 ||
        b_vnsg.dim(1) != nlebedev ||
        b_vnsg.dim(2) != nspins ||
        b_vnsg.dim(3) != ngrids ||
        n_qg.dim(0) != nq ||
        n_qg.dim(1) != ngrids ||
        dr_g.dim(0) != ngrids ||
        dv_g.dim(0) != ngrids ||
        weight_n.dim(0) != nlebedev ||
        Y_nL.dim(0) != nlebedev ||
        Y_nL.dim(1) != Lmax ||
        rnablaY_nLv.dim(0) != nlebedev ||
        rnablaY_nLv.dim(1) != Lmax ||
        rnablaY_nLv.dim(2) != 3) {
        PyErr_SetString(PyExc_ValueError,
                        "incompatible PAW GGA expansion array shapes");
        return 0;
    }
    return 1;
}


CLINKAGE PyObject* paw_gga_project_density(PyObject* self, PyObject* args)
{
    PyObject* n_sLg_obj;
    PyObject* dndr_sLg_obj;
    PyObject* Y_nL_obj;
    PyObject* rnablaY_nLv_obj;
    PyObject* r_g_obj;
    PyObject* n_sng_obj;
    PyObject* a_sng_obj;
    PyObject* b_vnsg_obj;
    PyObject* sigma_xng_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOOOO",
                          &n_sLg_obj,
                          &dndr_sLg_obj,
                          &Y_nL_obj,
                          &rnablaY_nLv_obj,
                          &r_g_obj,
                          &n_sng_obj,
                          &a_sng_obj,
                          &b_vnsg_obj,
                          &sigma_xng_obj))
        return NULL;

    DoubleArray<3> n_sLg(n_sLg_obj);
    DoubleArray<3> dndr_sLg(dndr_sLg_obj);
    DoubleArray<2> Y_nL(Y_nL_obj);
    DoubleArray<3> rnablaY_nLv(rnablaY_nLv_obj);
    DoubleArray<1> r_g(r_g_obj);
    DoubleArray<3> n_sng(n_sng_obj);
    DoubleArray<3> a_sng(a_sng_obj);
    DoubleArray<4> b_vnsg(b_vnsg_obj);
    DoubleArray<3> sigma_xng(sigma_xng_obj);
    if (!require_array(n_sLg, 3) ||
        !require_array(dndr_sLg, 3) ||
        !require_array(Y_nL, 2) ||
        !require_array(rnablaY_nLv, 3) ||
        !require_array(r_g, 1) ||
        !require_array(n_sng, 3) ||
        !require_array(a_sng, 3) ||
        !require_array(b_vnsg, 4) ||
        !require_array(sigma_xng, 3))
        return NULL;

    const int nspins = n_sLg.dim(0);
    const int Lmax = n_sLg.dim(1);
    const int ngrids = n_sLg.dim(2);
    const int nlebedev = Y_nL.dim(0);

    if (nspins != 1 && nspins != 2) {
        PyErr_SetString(PyExc_ValueError, "spin dimension must be 1 or 2");
        return NULL;
    }
    if (dndr_sLg.dim(0) != nspins ||
        dndr_sLg.dim(1) != Lmax ||
        dndr_sLg.dim(2) != ngrids ||
        Y_nL.dim(1) != Lmax ||
        rnablaY_nLv.dim(0) != nlebedev ||
        rnablaY_nLv.dim(1) != Lmax ||
        rnablaY_nLv.dim(2) != 3 ||
        r_g.dim(0) != ngrids ||
        n_sng.dim(0) != nspins ||
        n_sng.dim(1) != nlebedev ||
        n_sng.dim(2) != ngrids ||
        a_sng.dim(0) != nspins ||
        a_sng.dim(1) != nlebedev ||
        a_sng.dim(2) != ngrids ||
        b_vnsg.dim(0) != 3 ||
        b_vnsg.dim(1) != nlebedev ||
        b_vnsg.dim(2) != nspins ||
        b_vnsg.dim(3) != ngrids ||
        sigma_xng.dim(0) != 2 * nspins - 1 ||
        sigma_xng.dim(1) != nlebedev ||
        sigma_xng.dim(2) != ngrids) {
        PyErr_SetString(PyExc_ValueError,
                        "incompatible PAW GGA projection array shapes");
        return NULL;
    }

    const size_t rnablaY_vnL_size = (size_t)3 * nlebedev * Lmax;
    double* rnablaY_vnL_data =
        (double*)malloc(rnablaY_vnL_size * sizeof(double));
    if (rnablaY_vnL_data == NULL)
        return PyErr_NoMemory();
    const DoubleArray<3> rnablaY_vnL =
        DoubleArray<3>::view(rnablaY_vnL_data,
                             {3, nlebedev, Lmax},
                             {(npy_intp)nlebedev * Lmax, Lmax, 1});

    Py_BEGIN_ALLOW_THREADS
    project_density_compute(n_sLg, dndr_sLg, Y_nL, rnablaY_nLv, r_g,
                            n_sng, a_sng, b_vnsg, sigma_xng,
                            rnablaY_vnL);
    Py_END_ALLOW_THREADS

    free(rnablaY_vnL_data);
    Py_RETURN_NONE;
}


CLINKAGE PyObject* paw_gga_radial_gradient_correction(PyObject* self,
                                                      PyObject* args)
{
    PyObject* dr_g_obj;
    PyObject* dv_g_obj;
    PyObject* dedsigma_xng_obj;
    PyObject* a_sng_obj;
    PyObject* vv_sng_obj;

    if (!PyArg_ParseTuple(args, "OOOOO",
                          &dr_g_obj,
                          &dv_g_obj,
                          &dedsigma_xng_obj,
                          &a_sng_obj,
                          &vv_sng_obj))
        return NULL;

    DoubleArray<1> dr_g(dr_g_obj);
    DoubleArray<1> dv_g(dv_g_obj);
    DoubleArray<3> dedsigma_xng(dedsigma_xng_obj);
    DoubleArray<3> a_sng(a_sng_obj);
    DoubleArray<3> vv_sng(vv_sng_obj);
    if (!require_array(dr_g, 1) ||
        !require_array(dv_g, 1) ||
        !require_array(dedsigma_xng, 3) ||
        !require_array(a_sng, 3) ||
        !require_array(vv_sng, 3))
        return NULL;

    const int nspins = a_sng.dim(0);
    const int nlebedev = a_sng.dim(1);
    const int ngrids = a_sng.dim(2);
    const int ngradient = dedsigma_xng.dim(0);

    if (nspins != 1 && nspins != 2) {
        PyErr_SetString(PyExc_ValueError, "a_sng spin dimension must be 1 or 2");
        return NULL;
    }
    if (ngrids < 3) {
        PyErr_SetString(PyExc_ValueError, "radial grid must have at least 3 points");
        return NULL;
    }
    if (dr_g.dim(0) != ngrids ||
        dv_g.dim(0) != ngrids ||
        dedsigma_xng.dim(1) != nlebedev ||
        dedsigma_xng.dim(2) != ngrids ||
        vv_sng.dim(0) != nspins ||
        vv_sng.dim(1) != nlebedev ||
        vv_sng.dim(2) != ngrids ||
        ngradient != 2 * nspins - 1) {
        PyErr_SetString(PyExc_ValueError, "incompatible PAW GGA array shapes");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    radial_gradient_correction_compute(dr_g.data(), dv_g.data(),
                                       dedsigma_xng.data(), a_sng.data(),
                                       vv_sng.data(), nspins, nlebedev,
                                       ngrids);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}


CLINKAGE PyObject* paw_gga_back_project_density(PyObject* self,
                                                PyObject* args)
{
    PyObject* dEdD_sqL_obj;
    PyObject* dv_g_obj;
    PyObject* dedn_sng_obj;
    PyObject* vv_sng_arg;
    PyObject* n_qg_obj;
    PyObject* weight_n_obj;
    PyObject* Y_nL_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &dEdD_sqL_obj,
                          &dv_g_obj,
                          &dedn_sng_obj,
                          &vv_sng_arg,
                          &n_qg_obj,
                          &weight_n_obj,
                          &Y_nL_obj))
        return NULL;

    DoubleArray<3> dEdD_sqL(dEdD_sqL_obj);
    DoubleArray<1> dv_g(dv_g_obj);
    DoubleArray<3> dedn_sng(dedn_sng_obj);
    DoubleArray<2> n_qg(n_qg_obj);
    DoubleArray<1> weight_n(weight_n_obj);
    DoubleArray<2> Y_nL(Y_nL_obj);
    if (!require_array(dEdD_sqL, 3) ||
        !require_array(dv_g, 1) ||
        !require_array(dedn_sng, 3) ||
        !require_array(n_qg, 2) ||
        !require_array(weight_n, 1) ||
        !require_array(Y_nL, 2))
        return NULL;

    DoubleArray<3>* vv_sng = NULL;
    if (vv_sng_arg != Py_None) {
        vv_sng = new DoubleArray<3>(vv_sng_arg);
        if (!require_array(*vv_sng, 3)) {
            delete vv_sng;
            return NULL;
        }
    }

    const int nspins = dEdD_sqL.dim(0);
    const int nq = dEdD_sqL.dim(1);
    const int Lmax = dEdD_sqL.dim(2);
    const int nlebedev = dedn_sng.dim(1);
    const int ngrids = dedn_sng.dim(2);

    if (dedn_sng.dim(0) != nspins ||
        (vv_sng != NULL &&
         (vv_sng->dim(0) != nspins ||
          vv_sng->dim(1) != nlebedev ||
          vv_sng->dim(2) != ngrids)) ||
        dv_g.dim(0) != ngrids ||
        n_qg.dim(0) != nq ||
        n_qg.dim(1) != ngrids ||
        weight_n.dim(0) != nlebedev ||
        Y_nL.dim(0) != nlebedev ||
        Y_nL.dim(1) != Lmax) {
        delete vv_sng;
        PyErr_SetString(PyExc_ValueError,
                        "incompatible PAW GGA density back-projection shapes");
        return NULL;
    }

    const double* vv_sng_data = vv_sng == NULL ? NULL : vv_sng->data();
    Py_BEGIN_ALLOW_THREADS
    back_project_density_compute(dEdD_sqL.data(),
                                 dv_g.data(),
                                 dedn_sng.data(),
                                 vv_sng_data,
                                 n_qg.data(),
                                 weight_n.data(),
                                 Y_nL.data(),
                                 nspins, nq, Lmax, nlebedev, ngrids);
    Py_END_ALLOW_THREADS
    delete vv_sng;

    Py_RETURN_NONE;
}


CLINKAGE PyObject* paw_gga_back_project_gradient(PyObject* self,
                                                 PyObject* args)
{
    PyObject* dEdD_sqL_obj;
    PyObject* dedsigma_xng_obj;
    PyObject* b_vnsg_obj;
    PyObject* n_qg_obj;
    PyObject* weight_n_obj;
    PyObject* rnablaY_nLv_obj;

    if (!PyArg_ParseTuple(args, "OOOOOO",
                          &dEdD_sqL_obj,
                          &dedsigma_xng_obj,
                          &b_vnsg_obj,
                          &n_qg_obj,
                          &weight_n_obj,
                          &rnablaY_nLv_obj))
        return NULL;

    DoubleArray<3> dEdD_sqL(dEdD_sqL_obj);
    DoubleArray<3> dedsigma_xng(dedsigma_xng_obj);
    DoubleArray<4> b_vnsg(b_vnsg_obj);
    DoubleArray<2> n_qg(n_qg_obj);
    DoubleArray<1> weight_n(weight_n_obj);
    DoubleArray<3> rnablaY_nLv(rnablaY_nLv_obj);
    if (!require_array(dEdD_sqL, 3) ||
        !require_array(dedsigma_xng, 3) ||
        !require_array(b_vnsg, 4) ||
        !require_array(n_qg, 2) ||
        !require_array(weight_n, 1) ||
        !require_array(rnablaY_nLv, 3))
        return NULL;

    const int nspins = dEdD_sqL.dim(0);
    const int nq = dEdD_sqL.dim(1);
    const int Lmax = dEdD_sqL.dim(2);
    const int ngradient = dedsigma_xng.dim(0);
    const int nlebedev = dedsigma_xng.dim(1);
    const int ngrids = dedsigma_xng.dim(2);

    if (nspins != 1 && nspins != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "dEdD_sqL spin dimension must be 1 or 2");
        return NULL;
    }
    if (b_vnsg.dim(0) != 3 ||
        b_vnsg.dim(1) != nlebedev ||
        b_vnsg.dim(2) != nspins ||
        b_vnsg.dim(3) != ngrids ||
        n_qg.dim(0) != nq ||
        n_qg.dim(1) != ngrids ||
        weight_n.dim(0) != nlebedev ||
        rnablaY_nLv.dim(0) != nlebedev ||
        rnablaY_nLv.dim(1) != Lmax ||
        rnablaY_nLv.dim(2) != 3 ||
        ngradient != 2 * nspins - 1) {
        PyErr_SetString(PyExc_ValueError,
                        "incompatible PAW GGA back-projection array shapes");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    back_project_gradient_compute(dEdD_sqL.data(),
                                  dedsigma_xng.data(),
                                  b_vnsg.data(),
                                  n_qg.data(),
                                  NULL,
                                  weight_n.data(),
                                  rnablaY_nLv.data(),
                                  nspins, nq, Lmax, nlebedev, ngrids);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}


CLINKAGE PyObject* paw_gga_finish_lebedev(PyObject* self, PyObject* args)
{
    PyObject* dEdD_sqL_obj;
    PyObject* e_ng_obj;
    PyObject* dedn_sng_obj;
    PyObject* dedsigma_xng_obj;
    PyObject* a_sng_obj;
    PyObject* b_vnsg_obj;
    PyObject* n_qg_obj;
    PyObject* dr_g_obj;
    PyObject* dv_g_obj;
    PyObject* weight_n_obj;
    PyObject* Y_nL_obj;
    PyObject* rnablaY_nLv_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOO",
                          &dEdD_sqL_obj,
                          &e_ng_obj,
                          &dedn_sng_obj,
                          &dedsigma_xng_obj,
                          &a_sng_obj,
                          &b_vnsg_obj,
                          &n_qg_obj,
                          &dr_g_obj,
                          &dv_g_obj,
                          &weight_n_obj,
                          &Y_nL_obj,
                          &rnablaY_nLv_obj))
        return NULL;

    DoubleArray<3> dEdD_sqL(dEdD_sqL_obj);
    DoubleArray<2> e_ng(e_ng_obj);
    DoubleArray<3> dedn_sng(dedn_sng_obj);
    DoubleArray<3> dedsigma_xng(dedsigma_xng_obj);
    DoubleArray<3> a_sng(a_sng_obj);
    DoubleArray<4> b_vnsg(b_vnsg_obj);
    DoubleArray<2> n_qg(n_qg_obj);
    DoubleArray<1> dr_g(dr_g_obj);
    DoubleArray<1> dv_g(dv_g_obj);
    DoubleArray<1> weight_n(weight_n_obj);
    DoubleArray<2> Y_nL(Y_nL_obj);
    DoubleArray<3> rnablaY_nLv(rnablaY_nLv_obj);
    if (!require_array(dEdD_sqL, 3) ||
        !require_array(e_ng, 2) ||
        !require_array(dedn_sng, 3) ||
        !require_array(dedsigma_xng, 3) ||
        !require_array(a_sng, 3) ||
        !require_array(b_vnsg, 4) ||
        !require_array(n_qg, 2) ||
        !require_array(dr_g, 1) ||
        !require_array(dv_g, 1) ||
        !require_array(weight_n, 1) ||
        !require_array(Y_nL, 2) ||
        !require_array(rnablaY_nLv, 3))
        return NULL;

    if (!expansion_shape_check(dEdD_sqL, e_ng, dedn_sng, dedsigma_xng,
                               a_sng, b_vnsg, n_qg, dr_g, dv_g,
                               weight_n, Y_nL, rnablaY_nLv))
        return NULL;

    const int nspins = dEdD_sqL.dim(0);
    const int nq = dEdD_sqL.dim(1);
    const int Lmax = dEdD_sqL.dim(2);
    const int nlebedev = e_ng.dim(0);
    const int ngrids = e_ng.dim(1);

    const size_t vv_size = (size_t)nspins * nlebedev * ngrids;
    double* vv_sng = (double*)malloc(vv_size * sizeof(double));
    if (vv_sng == NULL)
        return PyErr_NoMemory();

    double E;
    Py_BEGIN_ALLOW_THREADS
    radial_gradient_correction_compute(dr_g.data(),
                                       dv_g.data(),
                                       dedsigma_xng.data(),
                                       a_sng.data(),
                                       vv_sng,
                                       nspins, nlebedev, ngrids);
    back_project_density_compute(dEdD_sqL.data(),
                                 dv_g.data(),
                                 dedn_sng.data(),
                                 vv_sng,
                                 n_qg.data(),
                                 weight_n.data(),
                                 Y_nL.data(),
                                 nspins, nq, Lmax, nlebedev, ngrids);
    back_project_gradient_compute(dEdD_sqL.data(),
                                  dedsigma_xng.data(),
                                  b_vnsg.data(),
                                  n_qg.data(),
                                  dr_g.data(),
                                  weight_n.data(),
                                  rnablaY_nLv.data(),
                                  nspins, nq, Lmax, nlebedev, ngrids);
    E = integrate_energy_compute(e_ng, dv_g, weight_n);
    Py_END_ALLOW_THREADS

    free(vv_sng);
    return PyFloat_FromDouble(E);
}


static int call_kernel(PyObject* kernel_calculate,
                       PyArrayObject* e_ng_obj,
                       PyArrayObject* n_sng_obj,
                       PyArrayObject* dedn_sng_obj,
                       PyArrayObject* sigma_xng_obj,
                       PyArrayObject* dedsigma_xng_obj)
{
    PyObject* result = PyObject_CallFunctionObjArgs(
        kernel_calculate,
        (PyObject*)e_ng_obj,
        (PyObject*)n_sng_obj,
        (PyObject*)dedn_sng_obj,
        (PyObject*)sigma_xng_obj,
        (PyObject*)dedsigma_xng_obj,
        NULL);
    if (result == NULL)
        return 0;
    Py_DECREF(result);
    return 1;
}


static int calculate_expansion(PyObject* kernel_calculate,
                               const double* D_sLq,
                               const double* n_qg,
                               const double* nc0_sg,
                               const double* B_pqL,
                               const double* dr_g,
                               const double* dv_g,
                               const double* r_g,
                               const double* Y_nL,
                               const double* weight_n,
                               const double* rnablaY_nLv,
                               double* dEdD_sp,
                               double dEdD_scale,
                               double* E,
                               int nspins,
                               int np,
                               int nq,
                               int Lmax,
                               int ngrids,
                               int nlebedev)
{
    npy_intp n_sng_dims[3] = {nspins, nlebedev, ngrids};
    npy_intp sigma_dims[3] = {2 * nspins - 1, nlebedev, ngrids};
    npy_intp e_dims[2] = {nlebedev, ngrids};
    npy_intp dEdD_dims[3] = {nspins, nq, Lmax};
    double* n_sLg = NULL;
    double* dndr_sLg = NULL;
    double* rnablaY_vnL_data = NULL;
    double* vv_sng = NULL;
    double* n_sng = NULL;
    double* a_sng = NULL;
    double* b_vnsg = NULL;
    double* sigma_xng = NULL;
    double* e_ng = NULL;
    double* dedn_sng = NULL;
    double* dedsigma_xng = NULL;
    double* dEdD_sqL = NULL;
    size_t n_sLg_size = (size_t)nspins * Lmax * ngrids;
    size_t rnablaY_vnL_size = (size_t)3 * nlebedev * Lmax;
    size_t vv_size = (size_t)nspins * nlebedev * ngrids;

    PyArrayObject* n_sng_obj = (PyArrayObject*)PyArray_SimpleNew(
        3, n_sng_dims, NPY_DOUBLE);
    PyArrayObject* a_sng_obj = (PyArrayObject*)PyArray_SimpleNew(
        3, n_sng_dims, NPY_DOUBLE);
    npy_intp b_vnsg_dims[4] = {3, nlebedev, nspins, ngrids};
    PyArrayObject* b_vnsg_obj = (PyArrayObject*)PyArray_SimpleNew(
        4, b_vnsg_dims, NPY_DOUBLE);
    PyArrayObject* sigma_xng_obj = (PyArrayObject*)PyArray_SimpleNew(
        3, sigma_dims, NPY_DOUBLE);
    PyArrayObject* e_ng_obj = (PyArrayObject*)PyArray_SimpleNew(
        2, e_dims, NPY_DOUBLE);
    PyArrayObject* dedn_sng_obj = (PyArrayObject*)PyArray_ZEROS(
        3, n_sng_dims, NPY_DOUBLE, 0);
    PyArrayObject* dedsigma_xng_obj = (PyArrayObject*)PyArray_ZEROS(
        3, sigma_dims, NPY_DOUBLE, 0);
    PyArrayObject* dEdD_sqL_obj = (PyArrayObject*)PyArray_ZEROS(
        3, dEdD_dims, NPY_DOUBLE, 0);

    if (n_sng_obj == NULL || a_sng_obj == NULL || b_vnsg_obj == NULL ||
        sigma_xng_obj == NULL || e_ng_obj == NULL ||
        dedn_sng_obj == NULL || dedsigma_xng_obj == NULL ||
        dEdD_sqL_obj == NULL)
        goto fail;

    n_sng = DOUBLEP(n_sng_obj);
    a_sng = DOUBLEP(a_sng_obj);
    b_vnsg = DOUBLEP(b_vnsg_obj);
    sigma_xng = DOUBLEP(sigma_xng_obj);
    e_ng = DOUBLEP(e_ng_obj);
    dedn_sng = DOUBLEP(dedn_sng_obj);
    dedsigma_xng = DOUBLEP(dedsigma_xng_obj);
    dEdD_sqL = DOUBLEP(dEdD_sqL_obj);

    n_sLg = (double*)malloc(n_sLg_size * sizeof(double));
    dndr_sLg = (double*)malloc(n_sLg_size * sizeof(double));
    rnablaY_vnL_data = (double*)malloc(rnablaY_vnL_size * sizeof(double));
    if (n_sLg == NULL || dndr_sLg == NULL || rnablaY_vnL_data == NULL) {
        PyErr_NoMemory();
        free(n_sLg);
        free(dndr_sLg);
        free(rnablaY_vnL_data);
        goto fail;
    }

    {
        const DoubleArray<3> n_sLg_view =
            DoubleArray<3>::view(n_sLg,
                                 {nspins, Lmax, ngrids},
                                 {(npy_intp)Lmax * ngrids, ngrids, 1});
        const DoubleArray<3> dndr_sLg_view =
            DoubleArray<3>::view(dndr_sLg,
                                 {nspins, Lmax, ngrids},
                                 {(npy_intp)Lmax * ngrids, ngrids, 1});
        const DoubleArray<2> Y_nL_view =
            DoubleArray<2>::view((double*)Y_nL,
                                 {nlebedev, Lmax},
                                 {Lmax, 1});
        const DoubleArray<3> rnablaY_nLv_view =
            DoubleArray<3>::view((double*)rnablaY_nLv,
                                 {nlebedev, Lmax, 3},
                                 {(npy_intp)Lmax * 3, 3, 1});
        const DoubleArray<1> r_g_view =
            DoubleArray<1>::view((double*)r_g, {ngrids}, {1});
        const DoubleArray<3> n_sng_view =
            DoubleArray<3>::view(n_sng,
                                 {nspins, nlebedev, ngrids},
                                 {(npy_intp)nlebedev * ngrids, ngrids, 1});
        const DoubleArray<3> a_sng_view =
            DoubleArray<3>::view(a_sng,
                                 {nspins, nlebedev, ngrids},
                                 {(npy_intp)nlebedev * ngrids, ngrids, 1});
        const DoubleArray<4> b_vnsg_view =
            DoubleArray<4>::view(
                b_vnsg,
                {3, nlebedev, nspins, ngrids},
                {(npy_intp)nlebedev * nspins * ngrids,
                 (npy_intp)nspins * ngrids, ngrids, 1});
        const DoubleArray<3> sigma_xng_view =
            DoubleArray<3>::view(sigma_xng,
                                 {2 * nspins - 1, nlebedev, ngrids},
                                 {(npy_intp)nlebedev * ngrids, ngrids, 1});
        const DoubleArray<3> rnablaY_vnL =
            DoubleArray<3>::view(rnablaY_vnL_data,
                                 {3, nlebedev, Lmax},
                                 {(npy_intp)nlebedev * Lmax, Lmax, 1});

        Py_BEGIN_ALLOW_THREADS
        for (int s = 0; s < nspins; s++) {
            row_dgemm('N', 'N', Lmax, ngrids, nq,
                      1.0, D_sLq + idx3(s, 0, 0, Lmax, nq), nq,
                      n_qg, ngrids,
                      0.0, n_sLg + idx3(s, 0, 0, Lmax, ngrids),
                      ngrids);
            for (int L = 0; L < Lmax; L++) {
                if (L == 0)
                    for (int g = 0; g < ngrids; g++)
                        n_sLg[idx3(s, L, g, Lmax, ngrids)] +=
                            nc0_sg[s * ngrids + g];
                derivative_term(dr_g,
                                n_sLg + idx3(s, L, 0, Lmax, ngrids),
                                dndr_sLg + idx3(s, L, 0, Lmax, ngrids),
                                ngrids);
            }
        }

        project_density_compute(n_sLg_view, dndr_sLg_view, Y_nL_view,
                                rnablaY_nLv_view, r_g_view,
                                n_sng_view, a_sng_view, b_vnsg_view,
                                sigma_xng_view, rnablaY_vnL);
        Py_END_ALLOW_THREADS
    }

    free(n_sLg);
    free(dndr_sLg);
    free(rnablaY_vnL_data);
    n_sLg = NULL;
    dndr_sLg = NULL;
    rnablaY_vnL_data = NULL;

    if (!call_kernel(kernel_calculate, e_ng_obj, n_sng_obj, dedn_sng_obj,
                     sigma_xng_obj, dedsigma_xng_obj))
        goto fail;

    vv_sng = (double*)malloc(vv_size * sizeof(double));
    if (vv_sng == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    {
        const DoubleArray<2> e_ng_view =
            DoubleArray<2>::view(e_ng,
                                 {nlebedev, ngrids},
                                 {(npy_intp)ngrids, 1});
        const DoubleArray<1> dv_g_view =
            DoubleArray<1>::view((double*)dv_g, {ngrids}, {1});
        const DoubleArray<1> weight_n_view =
            DoubleArray<1>::view((double*)weight_n, {nlebedev}, {1});

        Py_BEGIN_ALLOW_THREADS
        radial_gradient_correction_compute(dr_g, dv_g, dedsigma_xng,
                                           a_sng, vv_sng,
                                           nspins, nlebedev, ngrids);
        back_project_density_compute(dEdD_sqL, dv_g,
                                     dedn_sng, vv_sng, n_qg,
                                     weight_n, Y_nL, nspins, nq, Lmax,
                                     nlebedev, ngrids);
        back_project_gradient_compute(dEdD_sqL,
                                      dedsigma_xng,
                                      b_vnsg,
                                      n_qg, dr_g, weight_n, rnablaY_nLv,
                                      nspins, nq, Lmax, nlebedev, ngrids);
        *E = integrate_energy_compute(e_ng_view, dv_g_view, weight_n_view);
        Py_END_ALLOW_THREADS
    }

    free(vv_sng);
    vv_sng = NULL;

    if (dEdD_sp != NULL) {
        double* dEdD_tmp_sp = (double*)malloc(
            (size_t)nspins * np * sizeof(double));
        if (dEdD_tmp_sp == NULL) {
            PyErr_NoMemory();
            goto fail;
        }

        const DoubleArray<3> dEdD_sqL_view =
            DoubleArray<3>::view(dEdD_sqL,
                                 {nspins, nq, Lmax},
                                 {(npy_intp)nq * Lmax, Lmax, 1});
        const DoubleArray<2> dEdD_sqL_flat =
            dEdD_sqL_view.view<2>(0,
                                  {nspins, nq * Lmax},
                                  {(npy_intp)nq * Lmax, 1});
        const DoubleArray<2> B_pqL_flat =
            DoubleArray<2>::view((double*)B_pqL,
                                 {np, nq * Lmax},
                                 {(npy_intp)nq * Lmax, 1});
        const DoubleArray<2> dEdD_tmp =
            DoubleArray<2>::view(dEdD_tmp_sp,
                                 {nspins, np},
                                 {np, 1});
        const DoubleArray<2> dEdD_sp_view =
            DoubleArray<2>::view(dEdD_sp,
                                 {nspins, np},
                                 {np, 1});

        Py_BEGIN_ALLOW_THREADS
        dEdD_tmp.matmul(dEdD_sqL_flat, B_pqL_flat.transpose());
        dEdD_sp_view.add_scaled(dEdD_tmp, dEdD_scale);
        Py_END_ALLOW_THREADS
        free(dEdD_tmp_sp);
    }

    Py_DECREF(n_sng_obj);
    Py_DECREF(a_sng_obj);
    Py_DECREF(b_vnsg_obj);
    Py_DECREF(sigma_xng_obj);
    Py_DECREF(e_ng_obj);
    Py_DECREF(dedn_sng_obj);
    Py_DECREF(dedsigma_xng_obj);
    Py_DECREF(dEdD_sqL_obj);
    return 1;

fail:
    free(n_sLg);
    free(dndr_sLg);
    free(rnablaY_vnL_data);
    free(vv_sng);
    Py_XDECREF(n_sng_obj);
    Py_XDECREF(a_sng_obj);
    Py_XDECREF(b_vnsg_obj);
    Py_XDECREF(sigma_xng_obj);
    Py_XDECREF(e_ng_obj);
    Py_XDECREF(dedn_sng_obj);
    Py_XDECREF(dedsigma_xng_obj);
    Py_XDECREF(dEdD_sqL_obj);
    return 0;
}


CLINKAGE PyObject* paw_gga_single_atom_correction(PyObject* self,
                                                  PyObject* args)
{
    PyObject* kernel_calculate;
    PyObject* D_sp_obj;
    PyObject* dEdD_sp_arg;
    PyObject* B_pqL_obj;
    PyObject* n_qg_obj;
    PyObject* nt_qg_obj;
    PyObject* nc0_sg_obj;
    PyObject* nct0_sg_obj;
    PyObject* dr_g_obj;
    PyObject* dv_g_obj;
    PyObject* r_g_obj;
    PyObject* Y_nL_obj;
    PyObject* weight_n_obj;
    PyObject* rnablaY_nLv_obj;
    double e_xc0;
    int addcoredensity;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOdi",
                          &kernel_calculate,
                          &D_sp_obj,
                          &dEdD_sp_arg,
                          &B_pqL_obj,
                          &n_qg_obj,
                          &nt_qg_obj,
                          &nc0_sg_obj,
                          &nct0_sg_obj,
                          &dr_g_obj,
                          &dv_g_obj,
                          &r_g_obj,
                          &Y_nL_obj,
                          &weight_n_obj,
                          &rnablaY_nLv_obj,
                          &e_xc0,
                          &addcoredensity))
        return NULL;

    if (!PyCallable_Check(kernel_calculate)) {
        PyErr_SetString(PyExc_TypeError, "kernel_calculate must be callable");
        return NULL;
    }

    DoubleArray<2> D_sp(D_sp_obj);
    DoubleArray<3> B_pqL(B_pqL_obj);
    DoubleArray<2> n_qg(n_qg_obj);
    DoubleArray<2> nt_qg(nt_qg_obj);
    DoubleArray<2> nc0_sg(nc0_sg_obj);
    DoubleArray<2> nct0_sg(nct0_sg_obj);
    DoubleArray<1> dr_g(dr_g_obj);
    DoubleArray<1> dv_g(dv_g_obj);
    DoubleArray<1> r_g(r_g_obj);
    DoubleArray<2> Y_nL(Y_nL_obj);
    DoubleArray<1> weight_n(weight_n_obj);
    DoubleArray<3> rnablaY_nLv(rnablaY_nLv_obj);
    if (!require_array(D_sp, 2) ||
        !require_array(B_pqL, 3) ||
        !require_array(n_qg, 2) ||
        !require_array(nt_qg, 2) ||
        !require_array(nc0_sg, 2) ||
        !require_array(nct0_sg, 2) ||
        !require_array(dr_g, 1) ||
        !require_array(dv_g, 1) ||
        !require_array(r_g, 1) ||
        !require_array(Y_nL, 2) ||
        !require_array(weight_n, 1) ||
        !require_array(rnablaY_nLv, 3))
        return NULL;

    DoubleArray<2>* dEdD_sp_array = NULL;
    if (dEdD_sp_arg != Py_None) {
        dEdD_sp_array = new DoubleArray<2>(dEdD_sp_arg);
        if (!require_array(*dEdD_sp_array, 2)) {
            delete dEdD_sp_array;
            return NULL;
        }
    }

    const int nspins = D_sp.dim(0);
    const int np = D_sp.dim(1);
    const int nq = B_pqL.dim(1);
    const int Lmax = B_pqL.dim(2);
    const int ngrids = n_qg.dim(1);
    const int nlebedev = Y_nL.dim(0);

    if (nspins != 1 && nspins != 2) {
        delete dEdD_sp_array;
        PyErr_SetString(PyExc_ValueError, "D_sp spin dimension must be 1 or 2");
        return NULL;
    }
    if (B_pqL.dim(0) != np ||
        n_qg.dim(0) != nq ||
        nt_qg.dim(0) != nq ||
        nt_qg.dim(1) != ngrids ||
        nc0_sg.dim(0) != nspins ||
        nc0_sg.dim(1) != ngrids ||
        nct0_sg.dim(0) != nspins ||
        nct0_sg.dim(1) != ngrids ||
        dr_g.dim(0) != ngrids ||
        dv_g.dim(0) != ngrids ||
        r_g.dim(0) != ngrids ||
        Y_nL.dim(1) != Lmax ||
        weight_n.dim(0) != nlebedev ||
        rnablaY_nLv.dim(0) != nlebedev ||
        rnablaY_nLv.dim(1) != Lmax ||
        rnablaY_nLv.dim(2) != 3 ||
        (dEdD_sp_array != NULL &&
         (dEdD_sp_array->dim(0) != nspins ||
          dEdD_sp_array->dim(1) != np))) {
        delete dEdD_sp_array;
        PyErr_SetString(PyExc_ValueError,
                        "incompatible PAW GGA single-atom array shapes");
        return NULL;
    }

    const size_t D_size = (size_t)nspins * Lmax * nq;
    double* D_sLq = (double*)malloc(D_size * sizeof(double));
    if (D_sLq == NULL) {
        delete dEdD_sp_array;
        return PyErr_NoMemory();
    }

    const double* B_pqL_data = B_pqL.data();

    double* D_sqL = (double*)malloc(D_size * sizeof(double));
    if (D_sqL == NULL) {
        free(D_sLq);
        delete dEdD_sp_array;
        return PyErr_NoMemory();
    }

    const DoubleArray<3> D_sqL_view =
        DoubleArray<3>::view(D_sqL,
                             {nspins, nq, Lmax},
                             {(npy_intp)nq * Lmax, Lmax, 1});
    const DoubleArray<3> D_sLq_view =
        DoubleArray<3>::view(D_sLq,
                             {nspins, Lmax, nq},
                             {(npy_intp)Lmax * nq, nq, 1});

    Py_BEGIN_ALLOW_THREADS
    contract(D_sp.select(1), B_pqL.select(0), D_sqL_view);
    D_sLq_view.transpose_from(D_sqL_view, 0, 2, 1);
    Py_END_ALLOW_THREADS
    free(D_sqL);

    double* dEdD_sp = dEdD_sp_array == NULL ? NULL : dEdD_sp_array->data();
    double E = 0.0;
    double Et = 0.0;
    if (!calculate_expansion(kernel_calculate, D_sLq, n_qg.data(),
                             nc0_sg.data(), B_pqL_data,
                             dr_g.data(), dv_g.data(),
                             r_g.data(), Y_nL.data(),
                             weight_n.data(),
                             rnablaY_nLv.data(),
                             dEdD_sp, 1.0, &E, nspins, np, nq, Lmax,
                             ngrids, nlebedev)) {
        free(D_sLq);
        delete dEdD_sp_array;
        return NULL;
    }

    if (!calculate_expansion(kernel_calculate, D_sLq, nt_qg.data(),
                             nct0_sg.data(), B_pqL_data,
                             dr_g.data(), dv_g.data(),
                             r_g.data(), Y_nL.data(),
                             weight_n.data(),
                             rnablaY_nLv.data(),
                             dEdD_sp, -1.0, &Et, nspins, np, nq, Lmax,
                             ngrids, nlebedev)) {
        free(D_sLq);
        delete dEdD_sp_array;
        return NULL;
    }

    free(D_sLq);
    delete dEdD_sp_array;
    double energy = E - Et;
    if (addcoredensity)
        energy -= e_xc0;
    return PyFloat_FromDouble(energy);
}
