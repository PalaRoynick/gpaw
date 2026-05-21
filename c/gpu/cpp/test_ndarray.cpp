#include "python_utils.h"
#include "gpu/gpu_utils.h"
#include "ndarray.hpp"
#include "test_ndarray.hpp"
#include "array_desc.hpp"

namespace py = pybind11;

namespace gpaw::detail
{

/* Type-erased wrapper for array metadata, for testing only.
* Might be a useful starting point later if we want to support the DLpack interface
*/
struct GPAW_HIDDEN_SYMBOL ArrayDesc
{
    ArrayDesc() = default;

    // Pick array metadata from Numpy or Cupy ndarray. Suboptimal, used for tests only
    ArrayDesc(py::handle handle);

    py::dtype dtype() { return py::dtype(typenum); }

    // Use signed integers for compatibility with dlpack
    void* data = nullptr;
    int typenum = 0; // raw numpy typenum
    bool on_gpu = false;
    int32_t ndim = 0;

    // For Numpy shape/strides we could go for zero-copy (assuming we don't take ownership).
    // For Cupy this is harder because the shape/stride there are Python tuples (not contiguous C-style arrays)
    // So we opt for copies. Hardcode limit on NDIM to avoid heap allocations.
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    static constexpr size_t byte_offset = 0;
};

ArrayDesc::ArrayDesc(py::handle handle)
{
    PyObject* obj = handle.ptr();
    assert(obj);

    if (is_numpy_array(obj))
    {
        py::array arr = py::reinterpret_borrow<py::array>(handle);
        data = arr.mutable_data();
        typenum = arr.dtype().num();
        ndim = static_cast<int32_t>(arr.ndim());
        on_gpu = false;

        shape.resize(ndim);
        strides.resize(ndim);
        for (int32_t i = 0; i < ndim; ++i)
        {
            shape[i] = arr.shape(i);
            strides[i] = arr.strides(i);
        }
    }
    else if (is_cupy_array(obj))
    {
        // dtype
        py::handle array = handle;
        pybind11::object dtype_obj = array.attr("dtype");
        pybind11::object num_obj = dtype_obj.attr("num");
        typenum = num_obj.cast<int>();
        on_gpu = true;

        // Get data pointer
        data = reinterpret_cast<void*>(
            array.attr("data").attr("ptr").cast<std::uintptr_t>()
        );

        if (!data)
        {
            // This should always be a bug
            throw std::invalid_argument("Empty Cupy array passed to C++");
        }

        // This might be one extra copy?
        auto py_shape = array.attr("shape").cast<pybind11::tuple>();
        auto py_strides = array.attr("strides").cast<pybind11::tuple>();

        ndim = py_shape.size();

        assert(py_shape.size() == py_strides.size());
        shape.resize(py_shape.size());
        strides.resize(py_strides.size());

        for (size_t i = 0; i < shape.size(); ++i)
        {
            shape[i] = py_shape[i].cast<int64_t>();
            strides[i] = py_strides[i].cast<int64_t>();
        }
    }
    else
    {
        throw std::runtime_error("Not Numpy nor Cupy array?!");
    }
}

} // namespace gpaw::detail

namespace gpaw
{

/* Bind this to a Python function and call from pytest, passing the same
Numpy or Cupy array as both arguments. Should raise TypeError if the input array
has invalid traits. If the imported metadata does not match, will raise RuntimeError.
Former is a user error (trying to call function with wrong kind of array),
latter hints at a bug in NDArray import code.
*/
template<typename... Args>
static void accept_array(const NDArray<Args...>& a, py::handle same_array)
{
    detail::ArrayDesc desc(same_array);

    if (!a.is_valid())
    {
        throw std::runtime_error("Imported NDArray is invalid");
    }

    if (a.on_gpu() != desc.on_gpu)
    {
        throw std::runtime_error("Device does not match");
    }

    if (!a.dtype().is(desc.dtype()))
    {
        throw std::runtime_error("dtype does not match");
    }

    if (reinterpret_cast<uintptr_t>(a.data()) != reinterpret_cast<uintptr_t>(static_cast<std::byte*>(desc.data) + desc.byte_offset))
    {
        throw std::runtime_error("Data pointer does not match");
    }

    if (a.ndim() != desc.ndim)
    {
        throw std::runtime_error("ndim does not match");
    }

    for (size_t i = 0; i < desc.shape.size(); ++i)
    {
        if (a.shape(i) != (int64_t)desc.shape[i])
        {
            throw std::runtime_error("shape does not match at dimension " + std::to_string(i));
        }
        if (a.stride(i) != (int64_t)desc.strides[i])
        {
            throw std::runtime_error("stride does not match at dimension " + std::to_string(i));
        }
    }
}

} // namespace gpaw

void bind_ndarray_tests(py::module_ parent_module)
{
    using namespace gpaw;

    // Stubgen freaks out if the module name contains underscores...
    py::module_ m = parent_module.def_submodule("arraytests", "Unit testing for gpaw::NDArray");

    // No point trying to bind all combinations => just pick a few
    m.def("accept_any", &accept_array<>);
    m.def("accept_cpu", &accept_array<device::cpu>);
    m.def("accept_gpu", &accept_array<device::gpu>);
    m.def("accept_float", &accept_array<float>);
    m.def("accept_complexdouble", &accept_array<std::complex<double>>);
    m.def("accept_c_contig", &accept_array<c_contig>);
    m.def("accept_2D", &accept_array<ndim<2>>);
    m.def("accept_shape_1_2_3", &accept_array<shape<1, 2, 3>>);
    m.def("accept_shape_any", &accept_array<shape<-1>>);
    m.def("accept_shape_2_any", &accept_array<shape<2, -1>>);
    m.def("accept_double_c_contig_3D", &accept_array<double, c_contig, ndim<3>>);
    m.def("accept_complexdouble_f_contig_cpu_shape_2_3", &accept_array<std::complex<double>, f_contig, device::cpu, shape<2, 3>>);
    m.def("accept_complexfloat_c_contig_gpu_shape_any_any", &accept_array<std::complex<float>, c_contig, device::gpu, shape<-1, -1>>);

}
