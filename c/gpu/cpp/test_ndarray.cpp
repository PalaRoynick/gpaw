#include "python_utils.h"
#include "gpu/gpu_utils.h"
#include "ndarray.hpp"
#include "test_ndarray.hpp"

namespace py = pybind11;

namespace gpaw
{


void test_ndarray(NDArray<>& arr)
{
    // static_assert(sizeof(NDArray<>) == sizeof(py::handle));
    // static_assert(sizeof(NDArray<>) == sizeof(PyObject*));
    if (arr.has_dtype)
    {
        throw std::runtime_error("uhh");
    }
}

void test_ndarray_typed(NDArray<float>& arr)
{
    if (!arr.has_dtype)
    {
        throw std::runtime_error("???");
    }
}


}

void bind_ndarray_tests(py::module_ m)
{

    using namespace gpaw;

    m.def("cpp_test_ndarray", &test_ndarray);
    m.def("cpp_test_ndarray_typed", &test_ndarray_typed);

    NDArray<> a;
    NDArray<c_contig> b;

}
