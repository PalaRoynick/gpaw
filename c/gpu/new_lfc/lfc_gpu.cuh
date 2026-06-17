#pragma once
#include "gpu/cpp/utils.hpp"
#include <type_traits>
#include <cstdint>
#include <memory>

// TODO move under the gpu/ folder?

class SplinePoolGpu;

template<typename T>
using GpuVec3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;


/* Analogous to CPU-side BlockData, see comments in lfc_common.hpp for explanations. */
struct BlockDataGpu
{
    size_t num_blocks;

    long3* GPAW_RESTRICT block_start_c;
    long3* GPAW_RESTRICT block_shape;

    // CSR style index buffers
    size_t* GPAW_RESTRICT start_j;
    size_t* GPAW_RESTRICT all_blocks_phi_idx_j;
};

struct PhiDataGpu
{
    // TODO should make these const

    GpuVec3<double>* GPAW_RESTRICT position_iv;
    uint8_t* GPAW_RESTRICT l_i;
    size_t* GPAW_RESTRICT first_mu_i;
    size_t* GPAW_RESTRICT spline_idx_i;
    size_t num_phi;
};

// Probs no need to template this, not used for frequent ops
struct GridInfoGpu
{
    // TODO should make these const

    long3 start_c;
    long3 end_c;
    // Grid spacing in XYZ dirs
    double3 spacing;
    // Volume element
    double dV;
};

class LFCGpu
{
public:
    // TODO actual cleanup
    ~LFCGpu() {}

private:
    LFCGpu() = default;

    PhiDataGpu phi_data;
    BlockDataGpu block_data;
    GridInfoGpu grid;

    std::shared_ptr<const SplinePoolGpu> spline_pool;

    friend class LFCBuilderGpu;
};

