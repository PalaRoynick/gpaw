#pragma once

#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/gpu_core.hpp"
#include "radial_spline_gpu.cuh"

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
    if (x.ndim() != 1 || !x.is_c_contiguous() || x.dtype != pybind11::dtype::of<T>())
    {
        throw std::invalid_argument("input array must be contiguous 1D array with dtype matching the template parameter");
    }
    if (out.ndim() != 1 || out.shape[0] != x.shape[0] || !out.is_c_contiguous() || out.dtype != x.dtype)
    {
        throw std::invalid_argument("output array shape and dtype must match that of the input array");
    }

    const auto num_x = out.shape[0];
    constexpr int threads = 256;
    const int blocks = (x.shape[0] + threads - 1) / threads;

    launch_kernel(evaluate_spline_kernel<T>, blocks, threads, 0, stream,
        static_cast<T*>(x.data), static_cast<T*>(out.data), spline, num_x
    );
}

}
