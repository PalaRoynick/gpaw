#include "python_utils.h"
#include "basis_functions_python.hpp"
#include "basis_functions.cuh"

#include <pybind11/stl.h>

namespace py = pybind11;

namespace gpaw
{

template<typename T>
static bool bind_basis_functions_impl(py::module_ m)
{
    using ThisClass = BasisFunctionCollection<T>;

    const std::string class_name = std::is_same_v<T, float> ? "BasisFunctionCollectionCpp32" : "BasisFunctionCollectionCpp64";
    py::class_<ThisClass> cls(m, class_name.c_str());

    // Automatic std::vector -> python list conversion. But that involves copy. TODO see if we can (should) make this faster
    cls.def("add_to_density", &ThisClass::add_to_density);

    return true;
}

bool bind_basis_functions(py::module_ m)
{
    bool all_ok = true;
    all_ok &= bind_basis_functions_impl<float>(m);
    all_ok &= bind_basis_functions_impl<double>(m);

    return all_ok;
}

}
