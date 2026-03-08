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

    // Same as M_m but for phi index. This is really the one we need, so TODO get rid of hte above
    size_t* GPAW_RESTRICT start_j;
    size_t* GPAW_RESTRICT all_blocks_phi_idx_j;
};

template<typename T>
using GpuVec3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

template<typename T>
struct PhiDataGpu
{
    // TODO should make these const

    GpuVec3<T>* GPAW_RESTRICT position_iv;
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

    // TODO probably template this. The builder should coordinate with Python to create the correct object
    static constexpr int32_t BLOCK_SIZE = 8;
};

// Helper struct for mdspan-like behavior. TODO should replace these with proper spans
template<typename T>
struct AddToDensityArrays
{
    T* GPAW_RESTRICT nt_sG;
    const T* GPAW_RESTRICT f_sM;
    int32_t num_spins;
    // Needed to index f_sM[s * ... + mu]
    size_t f_sM_first_stride;
};

// Kernel forward declarations
template<typename T>
__global__ void add_to_density_kernel(AddToDensityArrays<T> arrays, const TRadialSplineGpu<T>* splines, GridInfoGpu grid,
    PhiDataGpu<T> phi, BlockDataGpu blocks);
//~


template<typename T>
inline LFCGpu<T>::~LFCGpu()
{
    gpuDeviceSynchronize();
}

template<typename T>
inline void LFCGpu<T>::add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const
{
    namespace py = pybind11;

    AddToDensityArrays<T> arrays;
    arrays.num_spins = nt_sG.shape(0);
    assert(f_sM.shape(0) == arrays.num_spins);

    // Check dtypes. TODO ndarray casting that would check these and add them to template traits
    if (!nt_sG.dtype().is(py::dtype::of<T>()) || !f_sM.dtype().is(py::dtype::of<T>()))
    {
        throw std::runtime_error("Wrong dtype in add_to_density input array(s)");
    }

    // TODO same check for device::gpu

    arrays.nt_sG = static_cast<T*>(nt_sG.data());
    arrays.f_sM = static_cast<T*>(f_sM.data());

    arrays.f_sM_first_stride = f_sM.stride(0);

    // One thread block per block of the real grid, and one thread for each point in that block.
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    // Launch blocks in 1D shape, so blockIdx.x gives block ID
    dim3 blocks(block_data.num_blocks);
    int shmem = 0;
    const gpuStream_t stream = 0;

    launch_kernel(add_to_density_kernel<T>, blocks, threads, shmem, stream, arrays, spline_pool->get_device_splines(), grid, phi_data, block_data);
}


} // namespace gpaw

#include "lfc_gpu_kernel_impl.tpp"