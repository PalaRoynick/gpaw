#pragma once

#include "lfc_base.hpp"

namespace gpaw
{

class SplinePoolImplBase;

// Non-templated entry point for the LFC GPU builder. This lets us avoid exposing CUDA code directly to lfc_builder.hpp
std::unique_ptr<LFCBase> build_impl_gpu_entry(bool single_precision, GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data,
    std::shared_ptr<const SplinePoolImplBase> spline_pool);

#ifndef GPAW_GPU
// Preparation for if we ever need this to compile without GPU support => can't use the GPU builder
std::unique_ptr<LFCBase> build_impl_gpu_entry(GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data,
    std::shared_ptr<const SplinePoolImplBase> spline_pool);
{
    throw std::runtime_error("build_impl_gpu requires that GPAW was compiled with -DGPAW_GPU");
}

#endif

} // namespace gpaw
