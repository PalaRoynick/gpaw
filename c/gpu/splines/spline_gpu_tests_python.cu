#include "python_utils.h"
#include "spline_gpu_tests_python.hpp"
#include "radial_spline_gpu.cuh"
#include "splines/splines_python.hpp"
#include "device_arrays/pybind_cupy_type_caster.hpp"
#include <vector>
#include <string>

namespace py = pybind11;

namespace gpaw
{

// Evaluates the input spline on 1D data
template<typename T>
__global__ void evaluate_spline_kernel(const T* x, T* out, TRadialSplineGpu<T> spline, size_t num_x)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_x)
    {
        out[i] = spline.evaluate(x[i]);
    }
}

/* Launches kernel that evaluates the input spline at given x and stores result in `out`.
Both arrays must be contiguous 1D and have matching dtype and size.
Does not block the host. */
template<typename T>
void launch_evaluate_spline(
    const PyDeviceArray& x,
    PyDeviceArray& out,
    const TRadialSplineGpu<T>& spline,
    gpuStream_t stream = 0)
{
    if (x.ndim() != 1 || !x.is_c_contiguous() || !x.dtype.is(pybind11::dtype::of<T>()))
    {
        throw std::invalid_argument("input array must be contiguous 1D array with dtype matching the template parameter");
    }
    if (out.ndim() != 1 || out.shape[0] != x.shape[0] || !out.is_c_contiguous() || !out.dtype.is(x.dtype))
    {
        throw std::invalid_argument("output array shape and dtype must match that of the input array");
    }

    const auto num_x = out.shape[0];
    constexpr int threads = 256;
    const int blocks = (x.shape[0] + threads - 1) / threads;

    launch_kernel(evaluate_spline_kernel<T>, blocks, threads, 0, stream,
        static_cast<T*>(x.data_ptr), static_cast<T*>(out.data_ptr), spline, num_x
    );
}

/* Helper class for testing GPU splines from Python side. We don't bind the GpuSpline classes directly,
* would lead to too much hassle with memory management and we don't really have the boilerplate in place to
* hande it well.
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
        out.data_ptr = static_cast<void*>(gpu::Tmalloc<T>(out.shape[0]));

        launch_evaluate_spline(x, out, gpu_spline, 0);

        // Copy to host
        py::array_t<T> res(out.shape);
        gpu::Tmemcpy(res.mutable_data(), static_cast<T*>(out.data_ptr), (size_t)x.shape[0], gpuMemcpyDeviceToHost);
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

    cls.doc() = "Utility class for exposing GPU splines for Python-side testing.";
}

void bind_gpu_spline_tests(pybind11::module m)
{
    bind_spline_tester<float>(m, "SplineTester32");
    bind_spline_tester<double>(m, "SplineTester64");
}


} // namespace gpaw
