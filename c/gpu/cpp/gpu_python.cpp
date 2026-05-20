#include "python_utils.h"
#include "gpu_python.h"
#include "test_ndarray.hpp"

#include "gpu/cpp/pyarray_utils.hpp"

#ifdef GPAW_WITH_MAGMA
    #include "gpu/cpp/magma/magma_python.h"
#endif

namespace py = pybind11;

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

    bool all_ok = true;

#ifdef GPAW_WITH_MAGMA
    // creates gpaw.gpu.magma
    all_ok &= bind_magma_submodule(submodule);
#endif

    return all_ok;
}
