#include "gpu/cpp/pyarray_utils.hpp"
#include "lfc_python.hpp"
#include "lfc_builder.hpp"
#include <pybind11/stl.h>
#include <span>

namespace py = pybind11;

static void bind_spline_pool(py::module m)
{
    // Expose just so that we can keep reference to the pool alive on Python side
    py::class_<SplinePool, py::smart_holder> cls(m, "SplinePool");
    cls.doc() = "C++ side object that holds CPU splines. Does not have a Python constructor.";
}

static void bind_lfc_builder(py::module_ m)
{
    py::class_<LFCBuilder> cls(m, "LFCBuilder");
    cls.doc() = "Helper class for building C++ side LFC objects";

    cls.def(py::init());

    auto add_spline = [](LFCBuilder& self, double cutoff, py::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> arr)
    {
        std::span<const double> data(arr.data(), arr.size());
        self.add_spline(cutoff, data);
    };

    cls.def("add_spline", add_spline, py::arg("cutoff"), py::arg("values"));
    cls.def("build_spline_pool", &LFCBuilder::build_spline_pool,
        "Finalizes spline creation and returns a shared SplinePool object.");

    //cls.def("enable_gpu", &LFCBuilder::enable_gpu);

    cls.def("set_block_size", &LFCBuilder::set_block_size);
    cls.def("add_phi", &LFCBuilder::add_phi);
}

bool bind_lfc_submodule(PyObject* module)
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

    py::module_ submodule = m.def_submodule("lfc", "Bindings to new LFC code");
    if (!submodule)
    {
        return false;
    }


    bind_spline_pool(submodule);
    bind_lfc_builder(submodule);

    return true;
}