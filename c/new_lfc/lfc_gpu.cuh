#pragma once
#include "utils.hpp"
#include <type_traits>
#include <cstdint>

// TODO move under the gpu/ folder?

class SplinePoolGpu;

template<typename T>
using GpuVec3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

//Contains info about all blocks in a SoA manner. So kernel uses block index to index these arrays (except for d_all_blocks_M_m)
struct BlockDataGpu
{
    size_t num_blocks;

    long3* GPAW_RESTRICT block_start_c;
    long3* GPAW_RESTRICT block_shape;

    /* CSR (Compressed Sparse Row) style index lookup. d_start_m[block_idx] gives the first index into d_all_blocks_M_m for that block
    and d_start_m[block_idx+1] gives the end index (exclusive) */
    size_t* GPAW_RESTRICT start_m; // size (num_blocks+1): one extra element to give end m for the last block

    /* Maps block local m to global mu index. But this is now over all blocks: use like
    int block_idx = ...;
    int m_offset = d_start_m[block_idx];
    for (int m = 0; m < d_start_m[block_idx + 1]; m++)
    {
        int mu = all_blocks_M_m[m + m_offset];
    }
    */
    size_t* GPAW_RESTRICT all_blocks_M_m;

    // Same as M_m but for phi index. This is really the one we need, so TODO get rid of hte above
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
    // TODO actual cleanup
    ~LFCGpu() {}

    PhiDataGpu phi_data;
    BlockDataGpu block_data;
    GridInfoGpu grid;

    std::shared_ptr<const SplinePoolGpu> spline_pool;
};
