#include "python_utils.h"
#include "gpu_python.h"
#include "test_ndarray.hpp"
#include "ndarray.hpp"

#include "gpu/cpp/pyarray_utils.hpp"

#ifdef GPAW_WITH_MAGMA
    #include "gpu/cpp/magma/magma_python.h"
#endif

namespace py = pybind11;

namespace gpaw
{

// NDArray tester: gets an array from Python side, wraps it with NDArray and prints its metadata
static void inspect(NDArray<>& a)
{
    printf("Array data pointer : %p\n", a.data());
    printf("Array dimension : %d\n", (int)a.ndim());
    for (int32_t i = 0; i < a.ndim(); ++i)
    {
        printf("Array dimension [%d] : %ld\n", i, a.shape(i));
        printf("Array stride    [%d] : %ld\n", i, a.stride(i));
    }
    printf("Is on GPU: %d\n", a.on_gpu());
    printf("Array dtype number: %d\n", a.dtype().num());
}

} // namespace gpaw


bool bind_gpu_submodule(PyObject* module)
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

    py::module_ submodule = m.def_submodule("gpu", "GPU specific C++ bindings for GPAW. (EXPERIMENTAL)");

    bind_ndarray_tests(submodule);
    submodule.def("inspect", gpaw::inspect, R"(NDArray tester: gets an array from Python side, wraps it with NDArray and prints its metadata)");

    bool all_ok = true;

#ifdef GPAW_WITH_MAGMA
    // creates gpaw.gpu.magma
    all_ok &= bind_magma_submodule(submodule);
#endif

    return all_ok;
}
