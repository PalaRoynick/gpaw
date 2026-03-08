#pragma once

#include "bmgs/bmgs.h"
#include <vector>
#include <memory>

namespace gpaw
{


struct SplineData
{
    std::vector<double> f_r;
    double cutoff = 0;
};

class SplinePoolImplBase
{
public:
    virtual ~SplinePoolImplBase() = default;
    virtual size_t num_splines() const = 0;
};

// Python sees this
class SplinePool
{
public:
    SplinePool(bool single_precision, bool use_gpu, const std::vector<bmgsspline>& legacy_splines);
    SplinePool(bool single_precision, bool use_gpu, const std::vector<SplineData>& spline_datas);

    size_t num_splines() const { return impl->num_splines(); }

    std::shared_ptr<const SplinePoolImplBase> get_impl() const { return impl; }
private:
    // C++ uses the implementation directly
    std::shared_ptr<SplinePoolImplBase> impl;
};

} // namespace gpaw
