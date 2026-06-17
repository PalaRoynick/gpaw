#include "lfc_builder_gpu.hpp"
#include "lfc_gpu.cuh"
#include "new_lfc/lfc_common.hpp"
#include "new_lfc/lfc_builder.hpp"

std::unique_ptr<LFCGpu> LFCBuilderGpu::build(LFCBuilder&& cpu_builder)
{
    return std::unique_ptr<LFCGpu>();
}
