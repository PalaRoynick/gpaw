#include "python_utils.h"
#include "gpu/gpu_utils.h"
#include "ndarray.hpp"
#include "test_ndarray.hpp"

namespace py = pybind11;

namespace gpaw
{

struct NDArrayMetadata
{
    void* data = nullptr;
    std::vector<int64_t> shape;
};

template<typename... Args>
void accept_array(const NDArray<Args...>& a)
{
}

void test_ndarray(NDArray<>& arr)
{
    // static_assert(sizeof(NDArray<>) == sizeof(py::handle));
    // static_assert(sizeof(NDArray<>) == sizeof(PyObject*));
    if (arr.traits.has_dtype)
    {
        throw std::runtime_error("uhh");
    }
}

void test_ndarray_typed(NDArray<float>& arr)
{
    if (!arr.traits.has_dtype)
    {
        throw std::runtime_error("???");
    }
}

void test_ndarray_complex(NDArray<std::complex<double>>& arr)
{
    // if (!arr.traits.has_dtype)
    // {
    //     throw std::runtime_error("???");
    // }
}

}

void bind_ndarray_tests(py::module_ parent_module)
{
    using namespace gpaw;

    py::module_ m = parent_module.def_submodule("ndarray_tests", "Unit tests for gpaw::NDArray");

    m.def("accept_any", &accept_array<>);
    m.def("accept_cpu", &accept_array<device::cpu>);
    m.def("accept_gpu", &accept_array<device::gpu>);
    m.def("accept_float", &accept_array<float>);
    m.def("accept_complexdouble", &accept_array<std::complex<double>>);
    m.def("accept_c_contig", &accept_array<c_contig>);
    m.def("accept_2D", &accept_array<ndim<2>>);
    m.def("accept_shape<1, 2, 3>", &accept_array<shape<1, 2, 3>>);
    m.def("accept_shape<2, -1>", &accept_array<shape<2, -1>>);
    m.def("accept_double_c_contig_gpu_shape<-1, -1>", &accept_array<double, c_contig, device::gpu, shape<-1, -1>>);

    NDArray<> a;
    NDArray<c_contig> b;
    NDArray<double, c_contig> c;
    NDArray<double, c_contig, shape<-1, 2>> aa;
    NDArray<shape<3>> a1;
    NDArray<ndim<2>>();
    NDArray<ndim<2>, double, f_contig>();
    NDArray<double, c_contig, device::gpu, shape<-1, -1>>();
}
