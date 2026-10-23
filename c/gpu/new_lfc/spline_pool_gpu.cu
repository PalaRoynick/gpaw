#include "spline_pool_gpu.cuh"
#include "new_lfc/lfc_builder.hpp"
#include "gpu/cpp/gpu_core.hpp"

SplinePoolGpu::SplinePoolGpu(const SplinePool& cpu_pool)
    : num(cpu_pool.num_splines()), h_splines(cpu_pool.num_splines())
{
    if (num_splines() <= 0)
    {
        throw std::runtime_error("Empty spline pool on CPU side, can't copy to GPU");
    }

    for (size_t i = 0; i < num; ++i)
    {
        h_splines[i] = GpuSpline::from_cpu(cpu_pool.get_spline(i));
    }

    d_splines = gpaw::gpu::Tmalloc<GpuSpline>(num);
    gpaw::gpu::Tmemcpy(d_splines, h_splines.data(), num, gpuMemcpyHostToDevice);
}

SplinePoolGpu::~SplinePoolGpu()
{
    for (size_t i = 0; i < num_splines(); i++)
    {
        h_splines[i].free();
    }

    if (d_splines)
    {
        gpaw::gpu::Tfree(d_splines);
    }
}
