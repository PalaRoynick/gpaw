#pragma once

#include "gpu/cpp/gpu_core.hpp"
#include "splines/splines_common.hpp"
#include "splines/radial_spline.hpp"
#include "gpaw_utils.h"
#include <type_traits>
#include <algorithm>

namespace gpaw
{

/* Like TRadialSpline but on the GPU. */
template<typename T>
struct TRadialSplineGpu
{
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");

    // Evaluate the spline S(x)
    __device__ T evaluate(T x) const;
    void free();

    static TRadialSplineGpu<T> from_cpu(const TRadialSpline<T>& cpu_spline);

private:
    // Coefficients in GPU memory
    SplineCoeffs<T>* GPAW_RESTRICT coeffs = nullptr;
    T x_end;
    T dx;
    size_t num_bins;
};

template<typename T>
__device__ inline T TRadialSplineGpu<T>::evaluate(T x) const
{
    // Clamp to [0, x_end]
    if (x >= x_end)
    {
        return T(0);
    }
    x = (x < T(0)) ? T(0) : x;
    const size_t bin_idx = floor(x / dx);
    const SplineCoeffs<T>& c = coeffs[bin_idx];

    // x - x_i
    const T u = x - bin_idx*dx;

    // S(x) = a + b*u + c*u^2 + d*u^3
    return c.a + u * (c.b + u * (c.c + u * c.d));
}

template<typename T>
inline void TRadialSplineGpu<T>::free()
{
    if (coeffs)
    {
        gpu::Tfree(coeffs);
        coeffs = nullptr;
    }
}

template<typename T>
inline TRadialSplineGpu<T> TRadialSplineGpu<T>::from_cpu(const TRadialSpline<T>& h_spline)
{
    TRadialSplineGpu<T> d_spline;
    const std::vector<SplineCoeffs<T>>& h_coeffs = h_spline.get_coefficients();

    d_spline.coeffs = gpu::Tmalloc<SplineCoeffs<T>>(h_coeffs.size());
    gpu::Tmemcpy(d_spline.coeffs, h_coeffs.data(), h_coeffs.size(), gpuMemcpyHostToDevice);

    d_spline.num_bins = h_spline.get_num_bins();
    d_spline.dx = h_spline.get_dx();
    d_spline.x_end = h_spline.get_x_end();

    assert(d_spline.num_bins == h_coeffs.size());
    return d_spline;
}

} // namespace gpaw
