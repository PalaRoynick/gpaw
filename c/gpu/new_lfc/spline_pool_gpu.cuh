#pragma once

#include "gpu/splines/radial_spline_gpu.cuh"
#include <vector>

class SplinePool;

/* Holds and owns splines allocated in GPU memory. RAII. */
class SplinePoolGpu
{
public:
    using dtype = double;
    using GpuSpline = gpaw::TRadialSplineGpu<dtype>;

    // Must be constructed from a CPU-side spline pool
    SplinePoolGpu(const SplinePool& cpu_pool);
    ~SplinePoolGpu();

    size_t num_splines() const { return num; }

    // Returns a device pointer to array holding the splines
    const GpuSpline* get_device_splines() const { return d_splines; }

private:
    size_t num = 0;
    GpuSpline* d_splines = nullptr;

    // Host-side copies of the GpuSpline structs, used both as a staging buffer and for doing cleanup in the destructor
    std::vector<GpuSpline> h_splines;
};
