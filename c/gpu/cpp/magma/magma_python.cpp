#include "python_utils.h"
#include "magma_python.h"
#include "magma_gpaw.hpp"
#include "magma_solvers.hpp"
#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/pybind_cupy_type_caster.hpp"
#include "gpaw_utils.h"
#include "gpu/cpp/ndarray.hpp"

#include <cassert>

namespace py = pybind11;

namespace gpaw
{

struct MagmaStaticData
{
    magma_int_t num_gpus = -1;
};

static MagmaStaticData magma_static_data;

/* Converts uplo string 'U'/'L' to appropriate Magma type.
This also takes care of convention difference between Numpy/Cupy arrays (C-style, row major)
and Magma arrays (Fortran/LAPACK style, column major).
So 'L' in Python conventions actually means MagmaUpper. */
static magma_uplo_t get_magma_uplo(const std::string& uplo)
{
    if (!(uplo == "L" || uplo == "U"))
    {
        throw std::invalid_argument("Invalid UPLO, must be 'L' or 'U'");
    }

    // Handle array convention difference
    return uplo == "L" ? MagmaUpper : MagmaLower;
}

static void check_gpu_request(int64_t requested_num_gpus)
{
    if (requested_num_gpus <= 0)
    {
        throw std::invalid_argument("requested_num_gpus <= 0");
    }

    if (requested_num_gpus > (int64_t)magma_static_data.num_gpus)
    {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
            "Requested %ld GPUs to a MAGMA routine but MAGMA only sees %ld.\n",
           (long)requested_num_gpus, (long)(magma_static_data.num_gpus)
        );

        throw std::invalid_argument(error_msg);
    }
}

/* Checks that the given matrix and eigenvalue arrays have matching dtypes (symmetric/Hermitian solver).
* If matrix dtype is complex<T> -> eigenvalue dtype must be T.
* If matrix dtype is T -> eigenvalue dtype must be T.
* Throws on failure. Note that integer dtypes may pass this.
*/
static void ensure_dtype_match_eigh(py::dtype matrix_dtype, py::dtype eigval_dtype)
{
    // Complex case
    if (matrix_dtype.kind() == 'c')
    {
        if (matrix_dtype.is(py::dtype::of<std::complex<double>>()))
        {
            if (!eigval_dtype.is(py::dtype::of<double>()))
            {
                throw std::invalid_argument("Matrix/eigenvalue dtype mismatch (complex double-precision)");
            }
        }
        else if (matrix_dtype.is(py::dtype::of<std::complex<float>>()))
        {
            if (!eigval_dtype.is(py::dtype::of<float>()))
            {
                throw std::invalid_argument("Matrix/eigenvalue dtype mismatch (complex single-precision)");
            }
        }
    }
    // Real case
    else if (!eigval_dtype.is(matrix_dtype))
    {
        throw std::invalid_argument("Matrix/eigenvalue mismatch (real matrix)");
    }
}


template<typename T, typename Device>
static magma_int_t magma_symmetric_solver(
    const MagmaEighContext& context,
    T* inout_matrix,
    T* inout_eigvals)
{
    if constexpr (std::is_same_v<Device, device::cpu>)
    {
        return magma_symmetric_solver_host<T>(context, inout_matrix, inout_eigvals);
    }
    else
    {
        return magma_symmetric_solver_gpu<T>(context, inout_matrix, inout_eigvals);
    }
}

template<typename T, typename Device>
static magma_int_t magma_hermitian_solver(
    const MagmaEighContext& context,
    magmaComplex<T>* inout_matrix,
    T* inout_eigvals)
{
    if constexpr (std::is_same_v<Device, device::cpu>)
    {
        return magma_hermitian_solver_host<T>(context, inout_matrix, inout_eigvals);
    }
    else
    {
        return magma_hermitian_solver_gpu<T>(context, inout_matrix, inout_eigvals);
    }
}

template<typename Device>
static void eigh_magma(
    NDArray<ndim<2>, c_contig, Device>& inout_matrix,
    NDArray<ndim<1>, c_contig, Device>& inout_eigvals,
    const py::str& uplo,
    int64_t num_gpus = 1)
{
    static_assert(std::is_same_v<Device, device::cpu> || std::is_same_v<Device, device::gpu>, "Device must be device::cpu or device::gpu");
    check_gpu_request(num_gpus);

    if constexpr (std::is_same_v<Device, device::gpu>)
    {
        if (num_gpus > 1)
        {
            throw std::runtime_error("Magma eigensolver with Cupy input currently only supports single-GPU");
        }
    }

    const int64_t matrix_size = inout_matrix.shape(0);
    if (inout_matrix.shape(1) != matrix_size)
    {
        throw std::invalid_argument("Input matrix must be an N x N matrix");
    }
    if (inout_eigvals.shape(0) != matrix_size)
    {
        throw std::invalid_argument("Input/output array for eigenvalues must be 1D and have length N (if the matrix is N x N)");
    }

    ensure_dtype_match_eigh(inout_matrix.dtype(), inout_eigvals.dtype());

    MagmaEighContext solver_context;
    solver_context.matrix_size = static_cast<magma_int_t>(matrix_size);
    solver_context.matrix_lda = solver_context.matrix_size;
    solver_context.uplo = get_magma_uplo(uplo.cast<std::string>());
    solver_context.jobz = MagmaVec; // Always do eigenvectors
    solver_context.num_gpus = num_gpus;

    const py::dtype dtype = inout_matrix.dtype();
    magma_int_t status;

    switch (dtype.normalized_num())
    {
        case py::dtype::num_of<float>():
            status = magma_symmetric_solver<float, Device>(
                solver_context,
                static_cast<float*>(inout_matrix.data()),
                static_cast<float*>(inout_eigvals.data())
            );
            break;
        case py::dtype::num_of<double>():
            status = magma_symmetric_solver<double, Device>(
                solver_context,
                static_cast<double*>(inout_matrix.data()),
                static_cast<double*>(inout_eigvals.data())
            );
            break;
        case py::dtype::num_of<std::complex<float>>():
            status = magma_hermitian_solver<float, Device>(
                solver_context,
                static_cast<magmaComplex<float>*>(inout_matrix.data()),
                static_cast<float*>(inout_eigvals.data())
            );
            break;
        case py::dtype::num_of<std::complex<double>>():
            status = magma_hermitian_solver<double, Device>(
                solver_context,
                static_cast<magmaComplex<double>*>(inout_matrix.data()),
                static_cast<double*>(inout_eigvals.data())
            );
            break;
    default:
        throw std::invalid_argument("Unsupported matrix dtype");
    }

    if (status < 0)
    {
        throw std::runtime_error("Invalid input to MAGMA solver at position " + std::to_string(-status));
    }
    else if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

}

static void init_magma_internals()
{
    MAGMA_CHECK(magma_init());

    // Cache things like number of GPUs available to MAGMA
    magma_int_t ndevices;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices(devices, MagmaMaxGPUs, &ndevices);

    gpaw::magma_static_data.num_gpus = ndevices;
}

static void finalize_magma_internals()
{
    MAGMA_CHECK(magma_finalize());
}

} // namespace gpaw


bool bind_magma_submodule(pybind11::module_ gpu_module)
{
    if (!gpu_module || gpu_module == Py_None)
    {
        return false;
    }

    py::module_ m = py::reinterpret_borrow<py::module_>(gpu_module);
    if (m.is_none())
    {
        return false;
    }
    py::module_ submodule = m.def_submodule("magma", "MAGMA bindings for GPAW");

    // Bind something that tells if magma is available or not. Doing it here instead of in Python ensures it shows up in
    // autogenerated stubs. Purepython side should define it manually if this magma module fails to import.
    // Binding a const bool as attribute is not ideal because it would anyway become mutable in Python.
    submodule.def("is_available", [](){ return true; },
        R"(Returns true if MAGMA is available, false otherwise.)");

    submodule.def("magma_init", &gpaw::init_magma_internals,
    R"(Initializes MAGMA library. Must be called come AFTER any calls to cudaSetValidDevices and cudaSetDeviceFlags.
        Call only if GPUs are available.)"
    );

    submodule.def("magma_finalize", &gpaw::finalize_magma_internals,
        R"(Cleanup of internal MAGMA state.)"
    );

    submodule.def("eigh_magma_numpy", &gpaw::eigh_magma<gpaw::device::cpu>,
    py::arg("inout_matrix"), py::arg("inout_eigenvalues"),
    py::arg("uplo"), py::arg("num_gpus"),
    R"(
        Solves eigensystem on the GPU with Numpy input/output. This is an in-place solver:
        the input matrix will be overwritten with resulting eigenvectors. The input eigenvalue array
        must already be allocated to correct size (its contents don't matter).
        Input matrix is in Numpy/Scipy conventions. Output is still in MAGMA (Fortran) convention,
        so you will need to conjugate transpose to get back to Numpy conventions for eigenvectors.

        Passing num_gpus > 1 will instruct the solver to utilize multiple GPUs. This can be beneficial for large matrices (N >= 10k).
        Note though that the GPUs must be directly reachable from the same Cuda/HIP context, ie. this is a single-node solver without Scalapack-like support.)"
    );

    submodule.def("eigh_magma_cupy", &gpaw::eigh_magma<gpaw::device::gpu>,
        py::arg("inout_matrix"), py::arg("inout_eigenvalues"), py::arg("uplo"), py::arg("num_gpus"),
        R"(
        Solves eigensystem on the GPU with Cupy input/output. This is an in-place solver:
        the input matrix will be overwritten with resulting eigenvectors. The input eigenvalue array
        must already be allocated to correct size (its contents don't matter).
        Input matrix is in Cupy/Scipy conventions. Output is still in MAGMA (Fortran) convention,
        so you will need to conjugate transpose to get back to Numpy conventions for eigenvectors.)"
    );

    return true;
}
