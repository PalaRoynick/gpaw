#include "python_utils.h"
#include "gpu/gpu_utils.h"
#include "ndarray.hpp"
#include "test_ndarray.hpp"

namespace py = pybind11;

namespace gpaw
{

/* Bind this to a Python function and call from pytest, passing the same
Numpy or Cupy array as both arguments. Should raise TypeError if the input array
has invalid traits. If the imported metadata does not match, will raise RuntimeError.
Former is a user error, latter hints at a bug in NDArray import code.
*/
template<typename... Args>
static void accept_array(const NDArray<Args...>& a, py::handle same_array)
{
    detail::ArrayDesc desc(same_array);

    if (!a.is_valid())
    {
        throw std::runtime_error("Imported NDArray is invalid");
    }

    if (!a.dtype().is(desc.dtype()))
    {
        throw std::runtime_error("dtype does not match");
    }

    if (reinterpret_cast<uintptr_t>(a.data()) != reinterpret_cast<uintptr_t>(static_cast<std::byte*>(desc.data) + desc.byte_offset))
    {
        throw std::runtime_error("Data pointer does not match");
    }

    if (a.ndim() != (size_t)desc.ndim)
    {
        throw std::runtime_error("ndim does not match");
    }

    for (size_t i = 0; i < desc.shape.size(); ++i)
    {
        if (a.shape(i) != (size_t)desc.shape[i])
        {
            throw std::runtime_error("shape does not match at dimension " + std::to_string(i));
        }
        if (a.stride(i) != (int64_t)desc.strides[i])
        {
            throw std::runtime_error("stride does not match at dimension " + std::to_string(i));
        }
    }
}

}

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
