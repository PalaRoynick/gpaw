#pragma once

namespace gpaw
{

// Cubic spline coefficients
template<typename T>
struct alignas(4 * sizeof(T)) SplineCoeffs
{
    T a, b, c, d;
};

// Forward declare GPU-side class
template<typename T>
struct TRadialSplineGpu;

} // namespace gpaw
