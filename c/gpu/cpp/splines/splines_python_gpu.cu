#include "python_utils.h"
#include "splines_python.hpp"
#include "spline_tools.cuh"
#include "radial_spline_gpu.cuh"
#include "radial_spline.hpp"
#include "gpu/cpp/pybind_cupy_type_caster.hpp"
#include <vector>
#include <string>

namespace py = pybind11;

namespace gpaw
{

/* Temporary class for testing splines on GPU. Can remove once we have some
proper "GpuBasisFunctions" class.
*/
template<typename T>
class SplineTester
{
public:

    SplineTester( T r_cutoff, py::array_t<T, py::array::c_style | py::array::forcecast> f_r)
        : spline(std::move(spline_from_numpy<T>(r_cutoff, f_r)))
    {
        gpu_spline = TRadialSplineGpu<T>::from_cpu(spline);
    }

    ~SplineTester()
    {
        printf("Freeing gpu_spline\n");
        gpu_spline.free();
    }

    py::array_t<T> evaluate_cpu(py::array_t<T, py::array::c_style | py::array::forcecast> x)
    {
        auto info = x.request();
        py::array_t<T> res(info.shape);

        T* res_ptr = res.mutable_data();
        const T* x_ptr = x.data();

        for (py::ssize_t i = 0; i < res.shape(0); i++)
        {
            res_ptr[i] = spline.evaluate(x_ptr[i]);
        }
        return res;
    }

    py::array_t<T> evaluate_gpu(const PyDeviceArray& x)
    {
        if (!x.c_contiguous || x.ndim() != 1 || !x.dtype.is(py::dtype::of<T>()))
        {
            const std::string msg = "Needs 1D contiguous Cupy array with dtype: "
                + std::to_string(py::dtype::of<T>().normalized_num())
                + ". You had dtype: " + std::to_string(x.dtype.normalized_num());
            throw std::invalid_argument(msg);
        }

        // Hacky
        PyDeviceArray out = x;
        out.data = static_cast<void*>(Tgpu_malloc<T>(out.shape[0]));

        launch_evaluate_spline(x, out, gpu_spline, 0);

        // Copy to host
        py::array_t<T> res(out.shape);
        Tgpu_memcpy(res.mutable_data(), static_cast<T*>(out.data), (size_t)x.shape[0], gpuMemcpyDeviceToHost);
        return res;
    }

private:
    TRadialSpline<T> spline;
    TRadialSplineGpu<T> gpu_spline;
};


template<typename T>
static void bind_spline_tester(py::module_& m, const std::string& class_name)
{
    py::class_<SplineTester<T>> cls(m, class_name.c_str());

    cls.def(
        py::init<T, py::array_t<T, py::array::c_style | py::array::forcecast>>(),
        py::arg("r_cutoff"),
        py::arg("f_r")
    );

    cls.def("evaluate_cpu", &SplineTester<T>::evaluate_cpu);
    cls.def("evaluate_gpu", &SplineTester<T>::evaluate_gpu);
}

void bind_gpu_splines(py::module_& m)
{
    bind_spline_tester<float>(m, "SplineTester32");
    bind_spline_tester<double>(m, "SplineTester64");
}


} // namespace gpaw
