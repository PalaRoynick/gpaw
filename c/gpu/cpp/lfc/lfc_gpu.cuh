#pragma once

#include "lfc_base.hpp"
#include "lfc_builder.hpp"
#include "gpaw_utils.h"
#include "gpu/cpp/gpu_core.hpp"
#include "gpu/cpp/splines/spline_pool_gpu.cuh"
#include <memory>

namespace gpaw
{

//Contains info about all blocks in a SoA manner. So kernel uses block index to index these arrays (except for d_all_blocks_M_m)
struct BlockDataGpu
{
    size_t num_blocks;

    long3* GPAW_RESTRICT block_start_c;
    long3* GPAW_RESTRICT block_shape;

    /* CSR style index lookup. d_start_m[block_idx] gives the first index into d_all_blocks_M_m for that block
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
};

template<typename T>
using GpuVec3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

template<typename T>
struct PhiDataGpu
{
    GpuVec3<T>* GPAW_RESTRICT position_iv;
    uint8_t* GPAW_RESTRICT l_i;
    size_t* GPAW_RESTRICT first_mu_i;
    size_t num_phi;
};

// Probs no need to template this, not used for frequent ops
struct GridInfoGpu
{
    long3 start_c;
    long3 end_c;
    // Grid spacing in XYZ dirs
    double3 spacing;
    // Volume element
    double dV;
};

template<typename T>
class LFCGpu final : public LFCBase
{
public:
    // TODO actual cleanup
    ~LFCGpu();

    void add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const;

    PhiDataGpu<T> phi_data;
    BlockDataGpu block_data;
    GridInfoGpu grid;

    std::shared_ptr<const SplinePoolGpu<T>> spline_pool;
};


template<typename T>
inline LFCGpu<T>::~LFCGpu()
{
    gpuDeviceSynchronize();
}

template<typename T>
inline void LFCGpu<T>::add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const
{
}



} // namespace gpaw