#include "gpu/cpp/pyarray_utils.hpp"
#include "lfc_python.hpp"
#include "lfc_builder.hpp"
#include <pybind11/stl.h>
#include <span>

#ifdef GPAW_GPU
// LFCGpu builder will be bound to the same Python-side LFCBuilder object
#include "gpu/new_lfc/lfc_gpu_python.hpp"
#endif

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

    cls.def("define_grid",&LFCBuilder::define_grid,
        py::arg("start_c"), py::arg("end_c"), py::arg("spacing_cv"), py::arg("dV"));
    cls.def("set_block_size", &LFCBuilder::set_block_size, py::arg("block_size"));
    cls.def("add_phi", &LFCBuilder::add_phi);

    auto add_block = [](
        LFCBuilder& self,
        const std::array<grid_index_t, 3>& block_start_c,
        const std::vector<size_t>& block_phi_idx_to_global)
    {
        self.add_block(block_start_c,
            std::span<const size_t>(block_phi_idx_to_global.begin(), block_phi_idx_to_global.end())
        );
    };

    cls.def("add_block", add_block,
        py::arg("block_start_c"), py::arg("block_phi_idx_to_global"));

#ifdef GPAW_GPU
    bind_lfc_gpu(m, cls);
    bind_gpu_spline_pool(m);
#endif
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