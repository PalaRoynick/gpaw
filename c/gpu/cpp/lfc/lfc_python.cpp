#include "lfc_python.hpp"
#include "lfc_staging.hpp"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace gpaw
{

bool bind_new_lfc(pybind11::module_ m)
{

    // Builder
    {
        using ThisClass = BasisFunctionCollectionBuilder;
        py::class_<ThisClass> cls(m, "BasisFunctionCollectionBuilder");
        cls.def(py::init<>());
        cls.def("add_block", &ThisClass::add_block);
        cls.def("enable_single_precision", &ThisClass::enable_single_precision);
    }

    {
        using ThisClass = BasisFunctionCollection;
        py::class_<ThisClass> cls(m, "BasisFunctionCollection_cpp");
    }

    //cls.def("add_to_density", &ThisClass::add_to_density);


    return true;
}

} // namespace gpaw
