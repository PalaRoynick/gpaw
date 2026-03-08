#include "spline_pool.hpp"
#include "spline_pool_gpu.cuh"

namespace gpaw
{

SplinePool::SplinePool(bool single_precision, bool use_gpu, const std::vector<SplineData>& spline_datas)
{
    if (use_gpu)
    {
        if (single_precision)
        {
            impl = std::make_shared<SplinePoolGpu<float>>(spline_datas);
        }
        else
        {
            impl = std::make_shared<SplinePoolGpu<double>>(spline_datas);
        }
    }
    else
    {
        throw std::runtime_error("WIP: SplinePool is only implemented for GPU");
    }
}


SplinePool::SplinePool(bool single_precision, bool use_gpu, const std::vector<bmgsspline>& legacy_splines)
{
    // bmgsspline is lightweight (holds pointer to data, not the data itself), so taking it as argument should be OK

    if (use_gpu)
    {
        if (single_precision)
        {
            impl = std::make_shared<SplinePoolGpu<float>>(legacy_splines);
        }
        else
        {
            impl = std::make_shared<SplinePoolGpu<double>>(legacy_splines);
        }
    }
    else
    {
        throw std::runtime_error("WIP: SplinePool is only implemented for GPU");
    }
}

} // namespace gpaw