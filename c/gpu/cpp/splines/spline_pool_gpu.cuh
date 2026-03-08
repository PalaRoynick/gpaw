#pragma once

#include "spline_pool.hpp"
#include "radial_spline_gpu.cuh"

namespace gpaw
{

template<typename T>
class SplinePoolGpu final : public SplinePoolImplBase
{
public:
    SplinePoolGpu(const std::vector<bmgsspline>& legacy_splines);
    SplinePoolGpu(const std::vector<SplineData>& spline_datas);
    ~SplinePoolGpu();

    size_t num_splines() const override { return num; }

private:
    size_t num;
    TRadialSplineGpu<T>* d_splines = nullptr;
};

template<typename T>
inline SplinePoolGpu<T>::SplinePoolGpu(const std::vector<SplineData>& spline_datas)
{
    num = spline_datas.size();

    d_splines = Tgpu_malloc<TRadialSplineGpu<T>>(num);

    for (size_t i = 0; i < spline_datas.size(); ++i)
    {
        const SplineData& data = spline_datas[i];
        TRadialSpline<T> cpu_spline(data.cutoff, data.f_r);
        d_splines[i] = TRadialSplineGpu<T>::from_cpu(cpu_spline);
    }

}

template<typename T>
inline SplinePoolGpu<T>::SplinePoolGpu(const std::vector<bmgsspline>& legacy_splines)
{
    num = legacy_splines.size();

    d_splines = Tgpu_malloc<TRadialSplineGpu<T>>(num);

    for (size_t i = 0; i < legacy_splines.size(); ++i)
    {
        auto cpu_spline = TRadialSpline<T>::from_legacy_spline(legacy_splines[i]);
        d_splines[i] = TRadialSplineGpu<T>::from_cpu(cpu_spline);
    }
}

template<typename T>
inline SplinePoolGpu<T>::~SplinePoolGpu()
{
    if (!d_splines)
    {
        return;
    }

    for (size_t i = 0; i < num; ++i)
    {
        d_splines[i].free();
    }
    Tgpu_free(d_splines);
}

} // namespace gpaw
