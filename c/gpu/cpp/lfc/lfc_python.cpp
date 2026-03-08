#include "lfc_python.hpp"
#include "lfc_base.hpp"
#include "lfc_builder.hpp"
#include "gpu/cpp/splines/spline_pool.hpp"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace gpaw
{

bool bind_new_lfc(pybind11::module_ m)
{

    {
        using ThisClass = BasisFunctionCollection;
        py::class_<ThisClass> cls(m, "BasisFunctionCollection_cpp");

        cls.def("add_to_density", &ThisClass::add_to_density);

    }

    // Builder
    {
        using ThisClass = BasisFunctionCollectionBuilder;
        py::class_<ThisClass> cls(m, "BasisFunctionCollectionBuilder");
        cls.def(py::init<>());
        cls.def("reserve", &ThisClass::reserve,
            R"(Optional pre-allocs if block/phi number is known.
            num_phi is the number of basis function instances, including periodic copies. WARNING: This will reset the builder's state!)");

        // C++ builder wants the SplinePool shared_ptr member. For Python it's more convenient to pass the Python-bound builder wrapper (ThisClass)
        cls.def("set_spline_pool", [](ThisClass& self, SplinePool& spline_pool)
            {
                self.set_spline_pool(spline_pool.get_impl());
            },
            py::arg("spline_pool")
        );

        cls.def("set_block_size", &ThisClass::set_block_size);
        cls.def("add_block", &ThisClass::add_block);
        cls.def("add_phi", &ThisClass::add_phi);
        cls.def("define_grid", &ThisClass::define_grid);
        cls.def("enable_single_precision", &ThisClass::enable_single_precision);
        cls.def("enable_gpu", &ThisClass::enable_gpu);
        cls.def("build", &ThisClass::build);
    }


    return true;
}

} // namespace gpaw
