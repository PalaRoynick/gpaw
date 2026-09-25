#pragma once

#include "splines_common.hpp"
#include "bmgs/bmgs.h"
#include <vector>
#include <type_traits>

namespace gpaw
{

/* Cubic spline interpolation in 1D with uniform spacing for radial functions.
* The spline is a C2-smooth piecewise polynomial
*    S(x) = S_i(x) for x_i < x <= x_{i+1},
* such that
*    S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3.
* This is intended for radial functions only, ie. x is in range [0, x_max].
* Boundary conditions are hardcoded:
*   - "Clamped" boundaries, ie. first derivatives at endpoints are zero. S'(0) = S'(x_max) = 0.
*   - S(x) = S(0) for x <= 0
*   - S(x) = 0 for x >= x_max.
*/
template<typename T>
class TRadialSpline
{
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
public:

    /* Create a uniformly-spaced cubic spline that interpolates the input data.
    Last element in the input data must be 0. */
    TRadialSpline(double x_end, const std::vector<T>& data);

    /* Construct from raw pointer. Used to avoid an intermediate copy when constructing from Numpy. */
    TRadialSpline(double x_end, const T* data, size_t num_data_elements);

    // Evaluate the spline S(x)
    T evaluate(T x) const;

    static TRadialSpline from_legacy_spline(const bmgsspline& spline);

    constexpr T get_x_start() const { return T(0.0); }
    T get_x_end() const { return x_end; }
    size_t get_num_bins() const { return num_bins; }
    T get_dx() const { return dx; }

    // Returns {x_i}, ie. the breakpoints used when the spline was constructed.
    std::vector<T> get_breakpoints() const;
    const std::vector<SplineCoeffs<T>>& get_coefficients() const { return coeffs; }

private:

    /* Compute spline coefficients and store in the `coeffs` vector. */
    void build_coefficients(const T* data, size_t num_data_elements);

    std::vector<SplineCoeffs<T>> coeffs;

    // Spacing of x-axis
    T dx;
    T x_end;
    size_t num_bins;
};

} // namespace gpaw

#include "radial_spline_impl.tpp"
