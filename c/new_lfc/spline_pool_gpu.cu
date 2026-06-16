#include "spline_pool_gpu.cuh"
#include "lfc_builder.hpp"
#include "gpu/cpp/gpu_core.hpp"

SplinePoolGpu::SplinePoolGpu(const SplinePool& cpu_pool)
    : num(cpu_pool.num_splines())
{
    if (num_splines() <= 0)
    {
        throw std::runtime_error("Empty spline pool on CPU side, can't copy to GPU");
    }

    d_splines = gpaw::gpu::Tmalloc<GpuSpline>(num);
    for (size_t i = 0; i < num; ++i)
    {
        const auto& cpu_spline = cpu_pool.get_spline(i);
        d_splines[i] = GpuSpline::from_cpu(cpu_spline);
    }
}

SplinePoolGpu::~SplinePoolGpu()
{
    for (size_t i = 0; i < num_splines(); i++)
    {
        d_splines[i].free();
    }

    if (d_splines)
    {
        gpaw::gpu::Tfree(d_splines);
    }
}
