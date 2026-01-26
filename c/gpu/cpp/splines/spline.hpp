#pragma once

#ifndef GPAW_CPP
    #error "Must compile GPAW in C++ mode"
#endif

#include "bmgs/bmgs.h"
#include <vector>
#include <type_traits>

/* Defines cubic spline interpolation in 1D. The spline is a piecewise polynomial
*    S(x) = S_i(x) for x_i < x <= x_{i+1},
* such that
*    S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3
* Coefficients are fixed so that the first and second derivatives are continuous,
* and that the second derivatives at the endpoints are zero.
*/

// Cubic spline coefficients
template<typename T>
struct alignas(4 * sizeof(T)) SplineCoeffs
{
    T a, b, c, d;
};

namespace gpaw
{

/* Cubic spline. This is a "natural" spline, ie. second derivatives at endpoints are zero.
* This is currently hardcoded to have zero boundary conditions,
* ie. S(x) = 0 for x outside the range [x_start, x_end]. */
template<typename T>
class TSpline
{
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");

public:

    /* Create a uniformly-spaced cubic spline that interpolates the input data.
    */
    TSpline(T x_start, T x_end, const std::vector<T>& data);

    // Evaluate the spline S(x)
    T evaluate(T x) const;

    static TSpline from_legacy_spline(const bmgsspline& spline);

    T get_x_start() const { return x_start; }
    T get_x_end() const { return x_end; }
    size_t get_num_bins() const { return num_bins; }

    // Returns {x_i}, ie. breakpoints used when the spline was constructed.
    std::vector<T> get_breakpoints() const;

private:

    /* Compute spline coefficients and store in the `coeffs` vector.
    Computation is done at double precision and cast to T at the end. */
    void build_coefficients(const std::vector<T>& data);

    std::vector<SplineCoeffs<T>> coeffs;

    // Spacing of x-axis
    T dx;
    T x_start;
    T x_end;
    size_t num_bins;
};

} // namespace gpaw

#include "spline_impl.tpp"
