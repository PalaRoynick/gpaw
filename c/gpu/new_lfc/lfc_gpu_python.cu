#include <pybind11/pybind11.h>
#include "device_arrays/pybind_cupy_type_caster.hpp"

#include "lfc_gpu.cuh"
#include "lfc_builder_gpu.hpp"
#include "new_lfc/lfc_builder.hpp"
#include "spline_pool_gpu.cuh"
#include "lfc_gpu_python.hpp"

namespace py = pybind11;

void bind_lfc_gpu(py::module_ m, py::class_<LFCBuilder> builder_cls)
{
    py::class_<LFCGpu> cls(m, "LFCGpu");
    cls.doc() = "Control object for GPU-LFC calculations (new LFC). No Python constructor: must be constructed using LFCBuilder";

    cls.def("add_to_density", &LFCGpu::add_to_density,
        py::arg("nt_sG"), py::arg("f_sM"));
    cls.def("construct_density", &LFCGpu::construct_density,
        py::arg("nt_G"), py::arg("rho_MM"), py::arg("mu_start"), py::arg("mu_end"));
    cls.def("calculate_potential_matrix", &LFCGpu::calculate_potential_matrix,
        py::arg("vt_G"), py::arg("out_V_MN"), py::arg("mu_start"), py::arg("mu_end"));


    // Bind build helper
    builder_cls.def("build_gpu", [](LFCBuilder& self, std::shared_ptr<const SplinePoolGpu> spline_pool)
    {
        return LFCBuilderGpu::build(std::move(self), std::move(spline_pool));
    }, "Builds GPU-side LFC object");
}

void bind_gpu_spline_pool(pybind11::module_ m)
{
    // Expose so that we can keep reference to the pool alive on Python side.
    // Also expose construction from CPU-side pool.
    py::class_<SplinePoolGpu, py::smart_holder> cls(m, "SplinePoolGpu");
    cls.doc() = "C++ side object that holds GPU splines.";

    cls.def(py::init<const SplinePool&>(), py::arg("spline_pool_cpu"), "Construct from SplinePoolCpu");
}
