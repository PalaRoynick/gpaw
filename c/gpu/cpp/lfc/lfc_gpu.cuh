#pragma once

#include "lfc_base.hpp"
#include "lfc_builder.hpp"
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
class LFCGpu : public LFCBase
{

public:
    // LFCGpu(const LFC_Impl<T>& lfc);
    // ~LFCGpu();

    void add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const;


private:

    PhiDataGpu<T> phi_data;
    BlockDataGpu block_data;

};

template<>
std::unique_ptr<LFCBase> build_impl<double, true>(GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data)
{
    std::unique_ptr<LFCCpu<double>> lfc = std::make_unique<LFCGpu<double>>();
    assert(lfc != nullptr);

    // lfc->blocks = std::move(blocks);
    // lfc->phi_data = std::move(phi_data);

    //lfc->gpu_impl = std::move(lfc->upload_to_gpu());

    return std::move(lfc);
}



template<typename T>
inline void LFCGpu<T>::add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const
{
}

} // namespace gpaw