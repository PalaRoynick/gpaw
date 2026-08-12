#pragma once
#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/utils.hpp"
#include "new_lfc/lfc_common.hpp"
#include "gpu/gpu-runtime.h"  // double3 etc
#include "gpu/splines/radial_spline_gpu.cuh"
#include <type_traits>
#include <cstdint>
#include <memory>

class SplinePoolGpu;

using LFCGpuSpline = gpaw::TRadialSplineGpu<double>;

/**************************************************************/
// Some template helpers. Define GpuVec3<T> as an alias to matchng int3, float3, long3, ...
// CPU side uses grid_index_t for grid point indexing so we want to use the same type in GPU structs.
// FIXME: would be cleaner with C++20 concepts

template<typename> inline constexpr bool always_false = false;

template<typename T, typename = void>
struct GpuVec3Type
{
    static_assert(always_false<T>, "GpuVec3<T>: unsupported scalar type");
};

template<typename T>
struct GpuVec3Type<T, std::enable_if_t<std::is_same_v<T, float>>>
{ using type = float3; };

template<typename T>
struct GpuVec3Type<T, std::enable_if_t<std::is_same_v<T, double>>>
{ using type = double3; };

template<typename T>
struct GpuVec3Type<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4 && std::is_signed_v<T>>>
{ using type = int3; };

template<typename T>
struct GpuVec3Type<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4 && std::is_unsigned_v<T>>>
{ using type = uint3; };

template<typename T>
struct GpuVec3Type<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 8 && std::is_signed_v<T>>>
{ using type = long3; };

template<typename T>
using GpuVec3 = typename GpuVec3Type<T>::type;

template<typename T>
GpuVec3<T> make_gpu_vec3(T x, T y, T z) { return {x, y, z}; }
/*************************************************************/

/* Analogous to CPU-side BlockData, see comments in lfc_common.hpp for explanations. */
struct BlockDataGpu
{
    size_t num_blocks;

    GpuVec3<grid_index_t> block_shape;
    GpuVec3<grid_index_t>* GPAW_RESTRICT block_start_bc;

    // CSR style index buffers
    size_t* GPAW_RESTRICT start_j;
    size_t* GPAW_RESTRICT all_blocks_phi_idx_j;
};

struct PhiDataGpu
{
    GpuVec3<double>* GPAW_RESTRICT position_iv;
    uint8_t* GPAW_RESTRICT l_i;
    size_t* GPAW_RESTRICT first_mu_i;
    size_t* GPAW_RESTRICT spline_idx_i;
    size_t num_phi;
};

struct GridInfoGpu
{
    GpuVec3<grid_index_t> start_c;
    GpuVec3<grid_index_t> end_c;
    double3 spacing_cv[3];
    // Volume element
    double dV;
};

// Keep templated implementations in a separate helper class
class LFCGpuImpl;

class LFCGpu
{
public:
    ~LFCGpu();

    LFCGpu(const LFCGpu&) = delete;
    LFCGpu& operator=(const LFCGpu&) = delete;
    // Could in theory be made moveable but it shouldn't be needed. Python side uses this through std::unique_ptr
    LFCGpu(LFCGpu&&) = delete;
    LFCGpu& operator=(LFCGpu&&) = delete;

    /* nt_sG: NDArray<device::gpu, c_contig, ndim<4>>
    *  f_sM: const NDArray<c_contig, ndim<2>>
    */
    void add_to_density(gpaw::PyDeviceArray& nt_sG, const gpaw::PyDeviceArray& f_sM) const;

    /* nt_G: NDArray<device::gpu, c_contig, ndim<3>>
    *  rho_MM: NDArray<device::gpu, c_contig, ndim<2>>
    */
    void construct_density(gpaw::PyDeviceArray& nt_G, const gpaw::PyDeviceArray& rho_MM) const;

    /* vt_G: const NDArray<device::gpu, c_contig, ndim<3>>
    *  out_V_MN: NDArray<c_contig, ndim<2>, shape=(mu_end-mu_start, Mmax)>
    *  mu_start, mu_end: row distribution range — only rows [mu_start, mu_end) are written
    */
    void calculate_potential_matrix(const gpaw::PyDeviceArray& vt_G, gpaw::PyDeviceArray& out_V_MN,
                                    size_t mu_start, size_t mu_end) const;

private:
    // Construction only via LFCBuilderGpu
    LFCGpu() = default;

    PhiDataGpu phi_data;
    BlockDataGpu block_data;
    GridInfoGpu grid;

    std::shared_ptr<const SplinePoolGpu> spline_pool;

    friend class LFCBuilderGpu;
    friend class LFCGpuImpl;
};
