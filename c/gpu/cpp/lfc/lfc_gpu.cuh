#pragma once

#include "lfc_staging.cuh"
#include "gpaw_utils.h"
#include "gpu/cpp/gpu_core.hpp"

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
    long3* GPAW_RESTRICT start_m; // size (num_blocks+1): one extra element to give end m for the last block

    /* Maps block local m to global mu index. But this is now over all blocks: use like
    int block_idx = ...;
    int m_offset = d_start_m[block_idx];
    for (int m = 0; m < d_start_m[block_idx + 1]; m++)
    {
        int mu = all_blocks_M_m[m + m_offset];
    }
    */
    long3* GPAW_RESTRICT all_blocks_M_m;
};

template<typename T>
struct PhiDataGpu
{
    using Vec3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

    size_t num_phi;
    Vec3* GPAW_RESTRICT position_iv;
    uint8_t* GPAW_RESTRICT l_i;
    size_t* GPAW_RESTRICT first_mu_i;
};

template<typename T>
class LFCGpu
{

public:
    LFCGpu(const LFC_Impl<T>& lfc);
    ~LFCGpu();

private:

    PhiDataGpu<T> phi_data;
    BlockDataGpu block_data;

};

template<typename T>
inline LFCGpu<T>::LFCGpu(const LFC_Impl<T>& lfc)
{
}

template<typename T>
inline LFCGpu<T>::~LFCGpu()
{

}

} // namespace gpaw