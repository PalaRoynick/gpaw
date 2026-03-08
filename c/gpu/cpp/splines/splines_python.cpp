#include "splines_python.hpp"
#include "spline_pool.hpp"
#include <pybind11/stl.h>
#include "gpu/cpp/ndarray.hpp"

namespace py = pybind11;

namespace gpaw
{

SplineData spline_data_from_numpy(double cutoff, const NDArray<double, device::cpu, ndim<1>>& f_r)
{
    std::vector<double> data;
    data.reserve(f_r.size());
    for (size_t i = 0; i < f_r.size(); i++)
    {
        data.push_back(f_r.data()[i]);
    }

    SplineData spline_data;
    spline_data.cutoff = cutoff;
    spline_data.f_r = std::move(data);
    return spline_data;
}

void bind_spline_pool(py::module_ m)
{

    {
        py::class_<SplineData> cls(m, "SplineData");
        cls.def(py::init(&spline_data_from_numpy));
    }

    {
        py::class_<SplinePool> cls(m, "SplinePool");
        cls.def(py::init<bool, bool, const std::vector<SplineData>&>(),
            py::arg("single_precision"), py::arg("use_gpu"),
            py::arg("legacy_splines")
        );
        cls.def("num_splines", &SplinePool::num_splines);
    }

}

} // namespace gpaw
