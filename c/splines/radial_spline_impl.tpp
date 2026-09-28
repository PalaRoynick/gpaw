#include "radial_spline.hpp"

#include <cassert>
#include <vector>

namespace gpaw
{

template<typename T>
inline TRadialSpline<T>::TRadialSpline(double x_end, const std::vector<T>& data)
    : TRadialSpline(x_end, data.data(), data.size())
{
}

template<typename T>
inline TRadialSpline<T>::TRadialSpline(double x_end, const T* data, size_t num_data_elements)
    : x_end(x_end)
{
    if (num_data_elements < 3)
    {
        throw std::invalid_argument("Spline constructor needs at least 3 data points");
    }
    if (x_end <= T(0.0))
    {
        throw std::invalid_argument("Spline constructor needs x_end > 0.0");
    }
    if (data[num_data_elements -1 ] != T(0.0))
    {
        throw std::invalid_argument("Last element in spline data array must be 0.0");
    }

    num_bins = num_data_elements - 1;
    dx = x_end / num_bins;

    build_coefficients(data, num_data_elements);
}

template<typename T>
inline T TRadialSpline<T>::evaluate(T x) const
{
    if (x >= x_end)
    {
        return 0;
    }

    // Use S(x) = S(0) for x < 0
    x = x > 0 ? x : T(0);
    const size_t bin_idx = floor(x / dx);
    const SplineCoeffs<T>& c = coeffs[bin_idx];

    // u = x - x_i
    const T u = x - bin_idx*dx;

    // S(x) = a + b*u + c*u^2 + d*u^3
    return c.a + u * (c.b + u * (c.c + u * c.d));
}

template<typename T>
inline TRadialSpline<T> TRadialSpline<T>::from_legacy_spline(const bmgsspline& spline)
{
    const double x_start = 0.0;
    const double x_end = x_start + spline.nbins * spline.dr;

    std::vector<T> data(spline.nbins + 1);

    // Compute spline values at its breakpoints
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<T>(bmgs_splinevalue(&spline, x_start + i*spline.dr));
    }

    return TRadialSpline<T>(x_end, std::move(data));
}

template<typename T>
inline std::vector<T> TRadialSpline<T>::get_breakpoints() const
{
    std::vector<T> x(num_bins + 1);
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = i * dx;
    }
    return x;
}

template<typename T>
inline void TRadialSpline<T>::build_coefficients(const T* data, size_t num_data_elements)
{
    /* We have (N+1) data points and N spline sections (bins).
    * Need to solve spline coefficients {a_i, b_i, c_i, d_i} for i = 0, ..., N-1 such that
    * S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3, S_i(x_i) = f_i(x_i).
    * See any text on cubic spline interpolation for derivation.
    * In short, requiring C^2 continuity lets us eliminate the b_i and d_i coefficients
    * in favor of c_i. The c_i form a tridiagonal system that we solve with the Thomas algorithm.
    * Note that c_i = 1/2 S_i''(x_i).
    */

    /* Let y_i = f(x_i) and h = dx = x_{i+1} - x_i (same for all i)
    * We have to solve c_i from:
    *   c_{i-1} + 4c_i + c_{i+1} = 3/h^2 (y_{i+1} - 2y_i + y_{i+1})
    * and we impose clamped boundary conditions.
    * See any article on the tridiagonal matrix algorithm (Thomas algorithm).
    * The following is based on the legacy spline builder in bmgs/spline.c,
    */

    // We only store up to c_{N-1}
    coeffs.resize(num_bins);

    // Shorthand
    const T* y = data;

    // Use doubles for auxiliary calculations
    std::vector<double> f2(num_bins + 1);
    std::vector<double> u(num_bins);

    const double Q = 3.0 / (dx * dx);

    // Thomas forward sweep
    f2[0] = -0.5;
    u[0] = (y[1] - y[0]) * Q;
    for (size_t b = 1; b < num_bins; b++)
    {
        double p = 0.5 * f2[b - 1] + 2.0;
        f2[b] = -0.5 / p;
        u[b] = ((y[b + 1] - 2.0 * y[b] + y[b - 1]) * Q - 0.5 * u[b - 1]) / p;
    }
    f2[num_bins] = ((y[num_bins - 1] * Q - 0.5 * u[num_bins - 1]) / (0.5 * f2[num_bins - 1] + 1.0));

    // Back substitution (unroll b=0 case to make this work with size_t)
    for (size_t b = num_bins - 1; b > 0; b--)
    {
        f2[b] = f2[b] * f2[b + 1] + u[b];
    }
    f2[0] = f2[0] * f2[1] + u[0];

    // Now c_i = 1/2 f2_i. Solve rest of the coeffs
    coeffs.resize(num_bins);
    for (size_t i = 0; i < num_bins; i++)
    {
        // a_i = f(x_i)
        coeffs[i].a = y[i];
        // b_i = (f_{i+1} - y_i) / h - h/3 * (2c_i + c_{i+1})
        coeffs[i].b = (y[i + 1] - y[i]) / dx - (f2[i]/3. + f2[i+1]/6.) * dx;
        // c_i = 1/2 f2_i
        coeffs[i].c = 0.5*f2[i];
        // d_i = (c_{i+1} - c_i) / (3*h)
        coeffs[i].d = (f2[i+1] - f2[i]) / (6. * dx);
    }
}

} // namespace gpaw
