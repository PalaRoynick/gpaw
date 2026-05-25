#include "python_utils.h"
#include "extensions.h"
#include "gpaw_utils.h"

#include <array>
#include <climits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


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


class PythonError : public std::runtime_error {
public:
    PythonError(PyObject* type, const std::string& message)
        : std::runtime_error(message), type_(type)
    {
    }

    PyObject* type() const
    {
        return type_;
    }

private:
    PyObject* type_;
};


class PropagatePythonError : public std::runtime_error {
public:
    PropagatePythonError()
        : std::runtime_error("propagating existing Python exception")
    {
    }
};


static void set_python_exception_from_cpp()
{
    try {
        throw;
    } catch (const PropagatePythonError&) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError,
                            "C++ code requested Python exception propagation");
    } catch (const PythonError& err) {
        PyErr_SetString(err.type(), err.what());
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch (const std::invalid_argument& err) {
        PyErr_SetString(PyExc_ValueError, err.what());
    } catch (const std::exception& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown C++ exception");
    }
}


template<typename Func>
static PyObject* translate_exceptions(Func&& func)
{
    try {
        return func();
    } catch (...) {
        set_python_exception_from_cpp();
        return NULL;
    }
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
class DoubleArrayView {
public:
    DoubleArrayView(PyObject* object)
        : array_(nullptr), data_(nullptr), c_contiguous_(0)
    {
        for (int i = 0; i < N; i++) {
            dims_[i] = 1;
            strides_[i] = 0;
        }

        if (!PyArray_Check(object))
            throw PythonError(PyExc_TypeError, "expected a NumPy array");

        PyArrayObject* array = (PyArrayObject*)object;
        if (PyArray_TYPE(array) != NPY_DOUBLE)
            throw std::invalid_argument("expected a float64 array");

        array_ = array;
        data_ = DOUBLEP(array);
        if (PyArray_NDIM(array_) != N) {
            array_ = nullptr;
            data_ = nullptr;
            throw std::invalid_argument(
                "unexpected number of array dimensions");
        }
        c_contiguous_ = PyArray_ISCARRAY(array_);
        for (int i = 0; i < N; i++)
            dims_[i] = PyArray_DIM(array_, i);
        for (int i = 0; i < N; i++)
            strides_[i] = PyArray_STRIDE(array_, i);
    }

    DoubleArrayView(double* data,
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

    static DoubleArrayView<N> view(double* data,
                               const std::array<npy_intp, N>& dims,
                               const std::array<npy_intp, N>& strides)
    {
        std::array<npy_intp, N> byte_strides;
        for (int i = 0; i < N; i++)
            byte_strides[i] = strides[i] * (npy_intp)sizeof(double);
        return DoubleArrayView<N>(data, dims.data(), byte_strides.data());
    }

    static DoubleArrayView<N> contiguous(double* data,
                                     const std::array<npy_intp, N>& dims)
    {
        std::array<npy_intp, N> strides;
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * dims[i + 1];
        return DoubleArrayView<N>::view(data, dims, strides);
    }

    template<int M>
    DoubleArrayView<M> view(npy_intp offset,
                        const std::array<npy_intp, M>& dims,
                        const std::array<npy_intp, M>& strides) const
    {
        return DoubleArrayView<M>::view(data_ + offset, dims, strides);
    }

    DoubleArrayView<N> transpose() const
    {
        static_assert(N == 2, "transpose view is implemented for 2D arrays");
        return DoubleArrayView<N>::view(data_,
                                    {dim(1), dim(0)},
                                    {stride(1), stride(0)});
    }

    void matmul(const DoubleArrayView& a,
                const DoubleArrayView& b,
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

    void add_scaled(const DoubleArrayView<N>& source, double scale) const
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

    void transpose_from(const DoubleArrayView& source,
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
    SelectedDim(const DoubleArrayView<N>& array, const std::array<int, M>& dims)
        : array_(array), dims_(dims)
    {
    }

    const DoubleArrayView<N>& array() const
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
    const DoubleArrayView<N> array_;
    const std::array<int, M> dims_;
};


template<int N>
template<typename... Dims>
SelectedDim<N, 1 + sizeof...(Dims)>
DoubleArrayView<N>::select(int dim, Dims... dims) const
{
    return SelectedDim<N, 1 + sizeof...(Dims)>(*this, {dim, dims...});
}


void contract(const SelectedDim<2, 1>& left,
              const SelectedDim<2, 1>& right,
              const DoubleArrayView<2>& out)
{
    const DoubleArrayView<2>& a = left.array();
    const DoubleArrayView<2>& b = right.array();

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
              const DoubleArrayView<3>& out)
{
    const DoubleArrayView<2>& a = left.array();
    const DoubleArrayView<3>& b = right.array();

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
            const DoubleArrayView<2> b_i =
                b.view<2>(i * b.stride(0),
                          {b.dim(1), b.dim(2)},
                          {b.stride(1), b.stride(2)});
            const DoubleArrayView<2> out_i =
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

    const DoubleArrayView<2> b_pqL =
        b.view<2>(0,
                  {b.dim(0), b.dim(1) * b.dim(2)},
                  {b.stride(0), 1});
    const DoubleArrayView<2> out_sqL =
        out.view<2>(0,
                    {out.dim(0), out.dim(1) * out.dim(2)},
                    {out.stride(0), 1});
    out_sqL.matmul(a, b_pqL);
}


void contract(const SelectedDim<3, 1>& left,
              const SelectedDim<3, 1>& right,
              const DoubleArrayView<4>& out)
{
    const DoubleArrayView<3>& a = left.array();
    const DoubleArrayView<3>& b = right.array();

    if (left.dim() != 2 || right.dim() != 1)
        throw std::invalid_argument("unsupported 3D-3D contraction dimensions");
    if (a.dim(2) != b.dim(1) ||
        out.dim(0) != a.dim(0) ||
        out.dim(1) != a.dim(1) ||
        out.dim(2) != b.dim(0) ||
        out.dim(3) != b.dim(2))
        throw std::invalid_argument("3D-3D contract dimension mismatch");

    for (int i = 0; i < a.dim(0); i++) {
        const DoubleArrayView<2> a_i =
            a.view<2>(i * a.stride(0),
                      {a.dim(1), a.dim(2)},
                      {a.stride(1), a.stride(2)});
        for (int j = 0; j < b.dim(0); j++) {
            const DoubleArrayView<2> b_j =
                b.view<2>(j * b.stride(0),
                          {b.dim(1), b.dim(2)},
                          {b.stride(1), b.stride(2)});
            const DoubleArrayView<2> out_ij =
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
    const DoubleArrayView<1>& a = left.array();
    const DoubleArrayView<1>& b = middle.array();
    const DoubleArrayView<2>& c = right.array();

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


// RAII wrapper around a malloc'd double buffer.
class Buffer {
public:
    explicit Buffer(size_t n)
        : n_(n), data_(n ? (double*)malloc(n * sizeof(double)) : nullptr)
    {
        if (n && !data_)
            throw std::bad_alloc();
    }
    ~Buffer() { free(data_); }
    double* data() const { return data_; }
    size_t size() const { return n_; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
private:
    size_t n_;
    double* data_;
};


template<int N>
class DoubleArray : private Buffer, public DoubleArrayView<N> {
    static size_t count(const std::array<npy_intp, N>& dims)
    {
        size_t n = 1;
        for (int i = 0; i < N; i++)
            n *= (size_t)dims[i];
        return n;
    }

    static std::array<npy_intp, N>
    byte_strides(const std::array<npy_intp, N>& dims)
    {
        std::array<npy_intp, N> strides;
        strides[N - 1] = sizeof(double);
        for (int i = N - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * dims[i + 1];
        return strides;
    }

public:
    explicit DoubleArray(const std::array<npy_intp, N>& dims)
        : Buffer(count(dims)),
          DoubleArrayView<N>(Buffer::data(), dims.data(),
                             byte_strides(dims).data())
    {
    }

    double* data() const
    {
        return DoubleArrayView<N>::data();
    }

    npy_intp size() const
    {
        return DoubleArrayView<N>::size();
    }

    void zero() const
    {
        memset(this->data(), 0, this->size() * sizeof(double));
    }
};


// Tracks named dimensions discovered from argument arrays and allocates
// temporary arrays from compact dimension strings such as "sng".
class DimSpace {
    std::map<char, npy_intp> known_dims_;

    static const char* dim_suffix(const char* name)
    {
        const char* last = nullptr;
        for (const char* p = name; *p; ++p)
            if (*p == '_') last = p;
        return last ? last + 1 : name;
    }

public:
    template<int N>
    void register_array(const char* name, const DoubleArrayView<N>& arr)
    {
        const char* suffix = dim_suffix(name);
        int len = 0;
        while (suffix[len]) ++len;
        if (len != N)
            return;
        for (int i = 0; i < N; i++) {
            char label = suffix[i];
            npy_intp size = (npy_intp)arr.dim(i);
            auto result = known_dims_.emplace(label, size);
            if (!result.second && result.first->second != size) {
                throw std::invalid_argument(
                    std::string("array '") + name + "': dimension '" +
                    label + "' has size " + std::to_string(size) +
                    " but expected " +
                    std::to_string(result.first->second) +
                    " from a previous array");
            }
        }
    }

    // Assert that a registered dimension label equals a fixed value.
    void require_dim(char label, npy_intp expected)
    {
        auto it = known_dims_.find(label);
        if (it == known_dims_.end())
            throw PythonError(PyExc_SystemError,
                std::string("require_dim: dimension '") + label +
                "' not yet registered");
        if (it->second != expected)
            throw std::invalid_argument(
                std::string("dimension '") + label + "' must be " +
                std::to_string(expected) + " but is " +
                std::to_string(it->second));
    }

    template<int N>
    DoubleArray<N> new_array(const char* dims)
    {
        std::array<npy_intp, N> shape;
        int len = 0;
        while (dims[len]) ++len;
        if (len != N) {
            throw PythonError(PyExc_SystemError,
                std::string("new_array: dimension string '") + dims +
                "' has length " + std::to_string(len) + ", expected " +
                std::to_string(N));
        }

        for (int i = 0; i < N; i++) {
            auto it = known_dims_.find(dims[i]);
            if (it == known_dims_.end())
                throw PythonError(PyExc_SystemError,
                    std::string("new_array: unknown dimension '") +
                    dims[i] + "' in '" + dims + "'");
            shape[i] = it->second;
        }
        return DoubleArray<N>(shape);
    }

    void set_dim(char label, npy_intp size)
    {
        auto result = known_dims_.emplace(label, size);
        if (!result.second && result.first->second != size) {
            throw std::invalid_argument(
                std::string("dimension '") + label + "' has size " +
                std::to_string(result.first->second) + " but expected " +
                std::to_string(size));
        }
    }
};


// Parses keyword arguments from a Python C extension call.
// Constructor takes both args and kwargs; it rejects any positional args.
// array() / optional_array() fetch numpy arrays, verify they are float64 and
// C-contiguous, and cross-check shared dimension labels: splitting the name at
// the last '_' yields one label per axis (e.g. "B_pqL" -> p=dim0, q=dim1,
// L=dim2); any two arrays sharing a label are checked to agree on that size.
// scalar<T>() and callable() fetch non-array arguments.
// require_dim(label, n) asserts a registered label has a specific size.
class Args {
    struct ArrayBase {
        virtual ~ArrayBase() = default;
    };

    template<int N>
    struct ArrayHolder : ArrayBase {
        DoubleArrayView<N> value;
        explicit ArrayHolder(DoubleArrayView<N> a) : value(a) {}
    };

    PyObject* kwargs_;
    std::vector<std::unique_ptr<ArrayBase>> storage_;
    DimSpace dimspace_;

    template<int N>
    DoubleArrayView<N>& store(DoubleArrayView<N> arr)
    {
        auto holder = std::make_unique<ArrayHolder<N>>(arr);
        DoubleArrayView<N>& ref = holder->value;
        storage_.push_back(std::move(holder));
        return ref;
    }

    PyObject* get_obj(const char* name)
    {
        if (!kwargs_)
            throw PythonError(PyExc_TypeError,
                std::string("missing required keyword argument '") +
                name + "'");
        PyObject* obj = PyDict_GetItemString(kwargs_, name);
        if (!obj)
            throw PythonError(PyExc_TypeError,
                std::string("missing required keyword argument '") +
                name + "'");
        return obj;
    }

    template<int N>
    DoubleArrayView<N>& make_array(const char* name, PyObject* obj)
    {
        DoubleArrayView<N> arr(obj);
        if (!arr.c_contiguous())
            throw std::invalid_argument(
                std::string("argument '") + name +
                "': array must be C-contiguous");
        dimspace_.register_array(name, arr);
        return store(arr);
    }

public:
    explicit Args(PyObject* args, PyObject* kwargs) : kwargs_(kwargs)
    {
        if (args && PyTuple_Size(args) != 0) {
            throw PythonError(PyExc_TypeError,
                "function takes keyword arguments only");
        }
    }

    DimSpace& dimspace()
    {
        return dimspace_;
    }

    // Fetch a callable object; throws if missing or not callable.
    PyObject* callable(const char* name)
    {
        PyObject* obj = get_obj(name);
        if (!PyCallable_Check(obj)) {
            throw PythonError(PyExc_TypeError,
                std::string("argument '") + name + "' must be callable");
        }
        return obj;
    }

    // Fetch a scalar. Supported types: double, bool, and integer types.
    // double uses PyFloat_AsDouble (accepts Python int too).
    // bool uses PyObject_IsTrue.
    // Other integral types use PyLong_AsLong.
    template<typename T>
    T scalar(const char* name)
    {
        PyObject* obj = get_obj(name);
        if constexpr (std::is_same_v<T, double>) {
            double val = PyFloat_AsDouble(obj);
            if (val == -1.0 && PyErr_Occurred())
                throw PropagatePythonError();
            return val;
        } else if constexpr (std::is_same_v<T, bool>) {
            int val = PyObject_IsTrue(obj);
            if (val == -1)
                throw PropagatePythonError();
            return val != 0;
        } else if constexpr (std::is_integral_v<T>) {
            long val = PyLong_AsLong(obj);
            if (val == -1 && PyErr_Occurred())
                throw PropagatePythonError();
            return static_cast<T>(val);
        } else {
            static_assert(sizeof(T) == 0,
                "unsupported scalar type for Args::scalar");
            return T{};
        }
    }

    void require_dim(char label, npy_intp expected)
    {
        dimspace_.require_dim(label, expected);
    }

    template<int N>
    DoubleArrayView<N>& array(const char* name)
    {
        PyObject* obj = get_obj(name);
        return make_array<N>(name, obj);
    }

    // Returns nullptr if the value is None; returns a pointer into the Args
    // storage otherwise.  The key must still be present in kwargs.
    template<int N>
    DoubleArrayView<N>* optional_array(const char* name)
    {
        PyObject* obj = get_obj(name);
        if (obj == Py_None) return nullptr;
        return &make_array<N>(name, obj);
    }
};


CLINKAGE PyObject* test_paw_double_array_matmul(PyObject* self,
                                                PyObject* args)
{
    PyObject* a_obj;
    PyObject* b_obj;
    PyObject* out_obj;

    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &out_obj))
        return NULL;

    return translate_exceptions([&]() -> PyObject* {
        DoubleArrayView<2> a(a_obj);
        DoubleArrayView<2> b(b_obj);
        DoubleArrayView<2> out(out_obj);
        out.matmul(a, b);
        Py_RETURN_NONE;
    });
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


static DoubleArrayView<3> spin_middle_view(const DoubleArrayView<3>& array)
{
    return array.view<3>(0,
                         {array.dim(1), array.dim(0), array.dim(2)},
                         {array.stride(1), array.stride(0),
                          array.stride(2)});
}


static void project_density_and_derivative(const DoubleArrayView<3>& n_sLg,
                                           const DoubleArrayView<3>& dndr_sLg,
                                           const DoubleArrayView<2>& Y_nL,
                                           const DoubleArrayView<3>& n_sng,
                                           const DoubleArrayView<3>& a_sng)
{
    const DoubleArrayView<3> n_nsg = spin_middle_view(n_sng);
    const DoubleArrayView<3> a_nsg = spin_middle_view(a_sng);

    contract(Y_nL.select(1), n_sLg.select(1), n_nsg);
    contract(Y_nL.select(1), dndr_sLg.select(1), a_nsg);
}


static void transpose_angular_gradient(const DoubleArrayView<3>& rnablaY_nLv,
                                       const DoubleArrayView<3>& rnablaY_vnL)
{
    rnablaY_vnL.transpose_from(rnablaY_nLv, 2, 0, 1);
}


static void project_angular_gradient(const DoubleArrayView<3>& n_sLg,
                                     const DoubleArrayView<3>& rnablaY_vnL,
                                     const DoubleArrayView<4>& b_vnsg)
{
    contract(rnablaY_vnL.select(2), n_sLg.select(1), b_vnsg);
}


static void calculate_angular_sigma(const DoubleArrayView<4>& b_vnsg,
                                    const DoubleArrayView<3>& sigma_xng)
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


static void scale_angular_sigma(const DoubleArrayView<1>& r_g,
                                const DoubleArrayView<3>& sigma_xng)
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


static void add_radial_sigma(const DoubleArrayView<3>& a_sng,
                             const DoubleArrayView<3>& sigma_xng)
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


static void project_density_compute(const DoubleArrayView<3>& n_sLg,
                                    const DoubleArrayView<3>& dndr_sLg,
                                    const DoubleArrayView<2>& Y_nL,
                                    const DoubleArrayView<3>& rnablaY_nLv,
                                    const DoubleArrayView<1>& r_g,
                                    const DoubleArrayView<3>& n_sng,
                                    const DoubleArrayView<3>& a_sng,
                                    const DoubleArrayView<4>& b_vnsg,
                                    const DoubleArrayView<3>& sigma_xng,
                                    const DoubleArrayView<3>& rnablaY_vnL)
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


static double integrate_energy_compute(const DoubleArrayView<2>& e_ng,
                                       const DoubleArrayView<1>& dv_g,
                                       const DoubleArrayView<1>& weight_n)
{
    return contract(weight_n.select(0), dv_g.select(0), e_ng.select(0, 1));
}


CLINKAGE PyObject* paw_gga_project_density(PyObject* self,
                                           PyObject* args,
                                           PyObject* kwargs)
{
    return translate_exceptions([&]() -> PyObject* {
        Args a(args, kwargs);
        auto& n_sLg       = a.array<3>("n_sLg");
        auto& dndr_sLg    = a.array<3>("dndr_sLg");
        auto& Y_nL        = a.array<2>("Y_nL");
        auto& rnablaY_nLv = a.array<3>("rnablaY_nLv");
        auto& r_g         = a.array<1>("r_g");
        auto& n_sng       = a.array<3>("n_sng");
        auto& a_sng       = a.array<3>("a_sng");
        auto& b_vnsg      = a.array<4>("b_vnsg");
        auto& sigma_xng   = a.array<3>("sigma_xng");
        DimSpace& dimspace = a.dimspace();

        const int nspins = n_sLg.dim(0);
        if (nspins != 1 && nspins != 2)
            throw std::invalid_argument("spin dimension must be 1 or 2");
        dimspace.require_dim('v', 3);
        dimspace.set_dim('x', 2 * nspins - 1);

        const DoubleArray<3> rnablaY_vnL = dimspace.new_array<3>("vnL");

        Py_BEGIN_ALLOW_THREADS
        project_density_compute(n_sLg, dndr_sLg, Y_nL, rnablaY_nLv, r_g,
                                n_sng, a_sng, b_vnsg, sigma_xng,
                                rnablaY_vnL);
        Py_END_ALLOW_THREADS

        Py_RETURN_NONE;
    });
}


CLINKAGE PyObject* paw_gga_radial_gradient_correction(PyObject* self,
                                                      PyObject* args,
                                                      PyObject* kwargs)
{
    return translate_exceptions([&]() -> PyObject* {
        Args a(args, kwargs);
        auto& dr_g          = a.array<1>("dr_g");
        auto& dv_g          = a.array<1>("dv_g");
        auto& dedsigma_xng  = a.array<3>("dedsigma_xng");
        auto& a_sng         = a.array<3>("a_sng");
        auto& vv_sng        = a.array<3>("vv_sng");
        DimSpace& dimspace  = a.dimspace();

        const int nspins = a_sng.dim(0);
        const int nlebedev = a_sng.dim(1);
        const int ngrids = a_sng.dim(2);

        if (nspins != 1 && nspins != 2)
            throw std::invalid_argument("a_sng spin dimension must be 1 or 2");
        if (ngrids < 3)
            throw std::invalid_argument(
                "radial grid must have at least 3 points");
        dimspace.set_dim('x', 2 * nspins - 1);

        Py_BEGIN_ALLOW_THREADS
        radial_gradient_correction_compute(dr_g.data(), dv_g.data(),
                                           dedsigma_xng.data(), a_sng.data(),
                                           vv_sng.data(), nspins, nlebedev,
                                           ngrids);
        Py_END_ALLOW_THREADS

        Py_RETURN_NONE;
    });
}


CLINKAGE PyObject* paw_gga_back_project_density(PyObject* self,
                                                PyObject* args,
                                                PyObject* kwargs)
{
    return translate_exceptions([&]() -> PyObject* {
        Args a(args, kwargs);
        auto& dEdD_sqL = a.array<3>("dEdD_sqL");
        auto& dv_g = a.array<1>("dv_g");
        auto& dedn_sng = a.array<3>("dedn_sng");
        auto* vv_sng = a.optional_array<3>("vv_sng");
        auto& n_qg = a.array<2>("n_qg");
        auto& weight_n = a.array<1>("weight_n");
        auto& Y_nL = a.array<2>("Y_nL");

        const int nspins = dEdD_sqL.dim(0);
        const int nq = dEdD_sqL.dim(1);
        const int Lmax = dEdD_sqL.dim(2);
        const int nlebedev = dedn_sng.dim(1);
        const int ngrids = dedn_sng.dim(2);
        const double* vv_sng_data = vv_sng == nullptr ? NULL : vv_sng->data();

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
        Py_RETURN_NONE;
    });
}


CLINKAGE PyObject* paw_gga_back_project_gradient(PyObject* self,
                                                 PyObject* args,
                                                 PyObject* kwargs)
{
    return translate_exceptions([&]() -> PyObject* {
        Args a(args, kwargs);
        auto& dEdD_sqL = a.array<3>("dEdD_sqL");
        auto& dedsigma_xng = a.array<3>("dedsigma_xng");
        auto& b_vnsg = a.array<4>("b_vnsg");
        auto& n_qg = a.array<2>("n_qg");
        auto& weight_n = a.array<1>("weight_n");
        auto& rnablaY_nLv = a.array<3>("rnablaY_nLv");
        DimSpace& dimspace = a.dimspace();

        const int nspins = dEdD_sqL.dim(0);
        const int nq = dEdD_sqL.dim(1);
        const int Lmax = dEdD_sqL.dim(2);
        const int nlebedev = dedsigma_xng.dim(1);
        const int ngrids = dedsigma_xng.dim(2);

        if (nspins != 1 && nspins != 2)
            throw std::invalid_argument(
                "dEdD_sqL spin dimension must be 1 or 2");
        dimspace.require_dim('v', 3);
        dimspace.set_dim('x', 2 * nspins - 1);

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
    });
}


CLINKAGE PyObject* paw_gga_finish_lebedev(PyObject* self,
                                          PyObject* args,
                                          PyObject* kwargs)
{
    return translate_exceptions([&]() -> PyObject* {
        Args a(args, kwargs);
        auto& dEdD_sqL = a.array<3>("dEdD_sqL");
        auto& e_ng = a.array<2>("e_ng");
        auto& dedn_sng = a.array<3>("dedn_sng");
        auto& dedsigma_xng = a.array<3>("dedsigma_xng");
        auto& a_sng = a.array<3>("a_sng");
        auto& b_vnsg = a.array<4>("b_vnsg");
        auto& n_qg = a.array<2>("n_qg");
        auto& dr_g = a.array<1>("dr_g");
        auto& dv_g = a.array<1>("dv_g");
        auto& weight_n = a.array<1>("weight_n");
        auto& Y_nL = a.array<2>("Y_nL");
        auto& rnablaY_nLv = a.array<3>("rnablaY_nLv");
        DimSpace& dimspace = a.dimspace();

        const int nspins = dEdD_sqL.dim(0);
        const int nq = dEdD_sqL.dim(1);
        const int Lmax = dEdD_sqL.dim(2);
        const int nlebedev = e_ng.dim(0);
        const int ngrids = e_ng.dim(1);

        if (nspins != 1 && nspins != 2)
            throw std::invalid_argument("spin dimension must be 1 or 2");
        dimspace.require_dim('v', 3);
        dimspace.set_dim('x', 2 * nspins - 1);

        const DoubleArray<3> vv_sng = dimspace.new_array<3>("sng");

        double E;
        Py_BEGIN_ALLOW_THREADS
        radial_gradient_correction_compute(dr_g.data(),
                                           dv_g.data(),
                                           dedsigma_xng.data(),
                                           a_sng.data(),
                                           vv_sng.data(),
                                           nspins, nlebedev, ngrids);
        back_project_density_compute(dEdD_sqL.data(),
                                     dv_g.data(),
                                     dedn_sng.data(),
                                     vv_sng.data(),
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

        return PyFloat_FromDouble(E);
    });
}


static void call_kernel(PyObject* kernel_calculate,
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
        throw PropagatePythonError();
    Py_DECREF(result);
}


static void calculate_expansion(PyObject* kernel_calculate,
                                DimSpace& dimspace,
                                const DoubleArrayView<3>& D_sLq,
                                const DoubleArrayView<2>& n_qg,
                                const DoubleArrayView<2>& nc0_sg,
                                const DoubleArrayView<3>& B_pqL,
                                const DoubleArrayView<1>& dr_g,
                                const DoubleArrayView<1>& dv_g,
                                const DoubleArrayView<1>& r_g,
                                const DoubleArrayView<2>& Y_nL,
                                const DoubleArrayView<1>& weight_n,
                                const DoubleArrayView<3>& rnablaY_nLv,
                                DoubleArrayView<2>* dEdD_sp,
                                double dEdD_scale,
                                double* E)
{
    const int nspins   = D_sLq.dim(0);
    const int Lmax     = D_sLq.dim(1);
    const int nq       = D_sLq.dim(2);
    const int np       = B_pqL.dim(0);
    const int ngrids   = n_qg.dim(1);
    const int nlebedev = Y_nL.dim(0);

    const DoubleArray<3> n_sLg = dimspace.new_array<3>("sLg");
    const DoubleArray<3> dndr_sLg = dimspace.new_array<3>("sLg");
    const DoubleArray<3> rnablaY_vnL = dimspace.new_array<3>("vnL");
    const DoubleArray<3> n_sng = dimspace.new_array<3>("sng");
    const DoubleArray<3> a_sng = dimspace.new_array<3>("sng");
    const DoubleArray<4> b_vnsg = dimspace.new_array<4>("vnsg");
    const DoubleArray<3> sigma_xng = dimspace.new_array<3>("xng");
    const DoubleArray<2> e_ng = dimspace.new_array<2>("ng");
    const DoubleArray<3> dedn_sng = dimspace.new_array<3>("sng");
    const DoubleArray<3> dedsigma_xng = dimspace.new_array<3>("xng");
    const DoubleArray<3> dEdD_sqL = dimspace.new_array<3>("sqL");
    const DoubleArray<3> vv_sng = dimspace.new_array<3>("sng");

    dedn_sng.zero();
    dedsigma_xng.zero();
    dEdD_sqL.zero();

    Py_BEGIN_ALLOW_THREADS
    for (int s = 0; s < nspins; s++) {
        row_dgemm('N', 'N', Lmax, ngrids, nq,
                  1.0, D_sLq.data() + idx3(s, 0, 0, Lmax, nq), nq,
                  n_qg.data(), ngrids,
                  0.0, n_sLg.data() + idx3(s, 0, 0, Lmax, ngrids),
                  ngrids);
        for (int L = 0; L < Lmax; L++) {
            if (L == 0)
                for (int g = 0; g < ngrids; g++)
                    n_sLg.data()[idx3(s, L, g, Lmax, ngrids)] +=
                        nc0_sg.data()[s * ngrids + g];
            derivative_term(dr_g.data(),
                            n_sLg.data() + idx3(s, L, 0, Lmax, ngrids),
                            dndr_sLg.data() + idx3(s, L, 0, Lmax, ngrids),
                            ngrids);
        }
    }
    project_density_compute(n_sLg, dndr_sLg, Y_nL,
                            rnablaY_nLv, r_g,
                            n_sng, a_sng, b_vnsg,
                            sigma_xng, rnablaY_vnL);
    Py_END_ALLOW_THREADS

    {
        npy_intp e_dims[2]     = {nlebedev, ngrids};
        npy_intp sng_dims[3]   = {nspins, nlebedev, ngrids};
        npy_intp sigma_dims[3] = {2 * nspins - 1, nlebedev, ngrids};
        PyArrayObject* e_ng_obj = (PyArrayObject*)PyArray_SimpleNewFromData(
            2, e_dims, NPY_DOUBLE, e_ng.data());
        PyArrayObject* n_sng_obj = (PyArrayObject*)PyArray_SimpleNewFromData(
            3, sng_dims, NPY_DOUBLE, n_sng.data());
        PyArrayObject* dedn_sng_obj = (PyArrayObject*)PyArray_SimpleNewFromData(
            3, sng_dims, NPY_DOUBLE, dedn_sng.data());
        PyArrayObject* sigma_xng_obj = (PyArrayObject*)PyArray_SimpleNewFromData(
            3, sigma_dims, NPY_DOUBLE, sigma_xng.data());
        PyArrayObject* dedsigma_xng_obj = (PyArrayObject*)PyArray_SimpleNewFromData(
            3, sigma_dims, NPY_DOUBLE, dedsigma_xng.data());
        if (e_ng_obj && n_sng_obj && dedn_sng_obj &&
            sigma_xng_obj && dedsigma_xng_obj) {
            try {
                call_kernel(kernel_calculate, e_ng_obj, n_sng_obj,
                            dedn_sng_obj, sigma_xng_obj, dedsigma_xng_obj);
            } catch (...) {
                Py_XDECREF(e_ng_obj);
                Py_XDECREF(n_sng_obj);
                Py_XDECREF(dedn_sng_obj);
                Py_XDECREF(sigma_xng_obj);
                Py_XDECREF(dedsigma_xng_obj);
                throw;
            }
        } else {
            Py_XDECREF(e_ng_obj);
            Py_XDECREF(n_sng_obj);
            Py_XDECREF(dedn_sng_obj);
            Py_XDECREF(sigma_xng_obj);
            Py_XDECREF(dedsigma_xng_obj);
            throw std::bad_alloc();
        }
        Py_XDECREF(e_ng_obj);
        Py_XDECREF(n_sng_obj);
        Py_XDECREF(dedn_sng_obj);
        Py_XDECREF(sigma_xng_obj);
        Py_XDECREF(dedsigma_xng_obj);
    }

    Py_BEGIN_ALLOW_THREADS
    radial_gradient_correction_compute(dr_g.data(), dv_g.data(),
                                       dedsigma_xng.data(), a_sng.data(),
                                       vv_sng.data(),
                                       nspins, nlebedev, ngrids);
    back_project_density_compute(dEdD_sqL.data(), dv_g.data(),
                                 dedn_sng.data(), vv_sng.data(),
                                 n_qg.data(), weight_n.data(), Y_nL.data(),
                                 nspins, nq, Lmax, nlebedev, ngrids);
    back_project_gradient_compute(dEdD_sqL.data(), dedsigma_xng.data(),
                                  b_vnsg.data(), n_qg.data(), dr_g.data(),
                                  weight_n.data(), rnablaY_nLv.data(),
                                  nspins, nq, Lmax, nlebedev, ngrids);
    *E = integrate_energy_compute(e_ng, dv_g, weight_n);
    Py_END_ALLOW_THREADS

    if (dEdD_sp != nullptr) {
        const auto dEdD_sqL_flat = dEdD_sqL.view<2>(0,
            {(npy_intp)nspins, (npy_intp)nq * Lmax},
            {(npy_intp)nq * Lmax, 1});
        const auto B_pqL_flat = B_pqL.view<2>(0,
            {(npy_intp)np, (npy_intp)nq * Lmax},
            {(npy_intp)nq * Lmax, 1});
        const DoubleArray<2> dEdD_tmp = dimspace.new_array<2>("sp");

        Py_BEGIN_ALLOW_THREADS
        dEdD_tmp.matmul(dEdD_sqL_flat, B_pqL_flat.transpose());
        dEdD_sp->add_scaled(dEdD_tmp, dEdD_scale);
        Py_END_ALLOW_THREADS
    }
}


CLINKAGE PyObject* paw_gga_single_atom_correction(PyObject* self,
                                                  PyObject* args,
                                                  PyObject* kwargs)
{
    return translate_exceptions([&]() -> PyObject* {
    Args a(args, kwargs);
    auto  kernel_calculate = a.callable("kernel_calculate");
    auto  e_xc0            = a.scalar<double>("e_xc0");
    auto  addcoredensity   = a.scalar<bool>("addcoredensity");
    auto& D_sp             = a.array<2>("D_sp");
    auto& B_pqL            = a.array<3>("B_pqL");
    auto& n_qg             = a.array<2>("n_qg");
    auto& nt_qg            = a.array<2>("nt_qg");
    auto& nc0_sg           = a.array<2>("nc0_sg");
    auto& nct0_sg          = a.array<2>("nct0_sg");
    auto& dr_g             = a.array<1>("dr_g");
    auto& dv_g             = a.array<1>("dv_g");
    auto& r_g              = a.array<1>("r_g");
    auto& Y_nL             = a.array<2>("Y_nL");
    auto& weight_n         = a.array<1>("weight_n");
    auto& rnablaY_nLv      = a.array<3>("rnablaY_nLv");
    auto* dEdD_sp          = a.optional_array<2>("dEdD_sp");
    DimSpace& dimspace     = a.dimspace();
    a.require_dim('v', 3);

    if (D_sp.dim(0) != 1 && D_sp.dim(0) != 2) {
        throw std::invalid_argument("D_sp spin dimension must be 1 or 2");
    }

    dimspace.set_dim('x', 2 * D_sp.dim(0) - 1);

    const DoubleArray<3> D_sLq = dimspace.new_array<3>("sLq");

    {
        const DoubleArray<3> D_sqL = dimspace.new_array<3>("sqL");
        Py_BEGIN_ALLOW_THREADS
        contract(D_sp.select(1), B_pqL.select(0), D_sqL);
        D_sLq.transpose_from(D_sqL, 0, 2, 1);
        Py_END_ALLOW_THREADS
    }

    double E = 0.0;
    double Et = 0.0;
    calculate_expansion(kernel_calculate, dimspace, D_sLq,
                        n_qg, nc0_sg, B_pqL,
                        dr_g, dv_g, r_g, Y_nL, weight_n, rnablaY_nLv,
                        dEdD_sp, 1.0, &E);

    calculate_expansion(kernel_calculate, dimspace, D_sLq,
                        nt_qg, nct0_sg, B_pqL,
                        dr_g, dv_g, r_g, Y_nL, weight_n, rnablaY_nLv,
                        dEdD_sp, -1.0, &Et);

    double energy = E - Et;
    if (addcoredensity)
        energy -= e_xc0;
    return PyFloat_FromDouble(energy);
    });
}
