#include "lfc_gpu.cuh"
#include "lfc_builder_gpu.hpp"
#include "new_lfc/lfc_builder.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_lfc_gpu(py::module_ m, py::class_<LFCBuilder> builder_cls)
{
    // Must register the GPU builder class with pybind11 even though we don't bind anything methods to it
    py::class_<LFCGpu>(m, "LFCGpu");

    builder_cls.def("build_gpu", [](LFCBuilder& self)
    {
        return LFCBuilderGpu::build(std::move(self));
    }, "Builds GPU-side LFC object");
}
