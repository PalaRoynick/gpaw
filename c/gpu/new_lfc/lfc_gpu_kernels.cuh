#pragma once
#include "gpu/gpu-runtime.h"
#include "gpaw_utils.h"
#include "lfc_gpu.cuh"


namespace kernel_utils
{

    // Largest number of warps we need to support for block reductions, which are implemented via warp reductions under the hood.
    // Value of 32 is OK for both warp size of 32/64
    inline constexpr int MAX_WARPS_PER_BLOCK = 32;

    /**
     * Debug assertions for device code. This is no-op if GPAW_KERNEL_DEBUG is undefined.
     */
    __device__ __forceinline__ void kernel_assert(
        [[maybe_unused]] bool condition,
        [[maybe_unused]] const char* msg,
        [[maybe_unused]] const char* file = __FILE__,
        [[maybe_unused]] int line = __LINE__)
    {
    #ifdef GPAW_KERNEL_DEBUG
        if (!condition)
        {
            printf("%s:%d: GPAW GPU kernel assertion failed: %s\n", file, line, msg);
        }
        assert(condition);
    #endif
    }

    /**
     * Performs a sum-reduction across threads of the calling warp. Every thread in the warp must
     * call this, and the warp must be fully populated. Note that in practice this requires the
     * calling thread block size to be a multiple of warpSize.
     * Only the result in the warp's first lane is meaningful.
     */
    template<typename T>
    __device__ __forceinline__ T warp_reduce_sum(T val)
    {
        // See any modern reference on warp primitives for the implementation.
        // Ideally we'd replace this with cooperative-group based reduction once ROCm supports it (>= 7.0)
    #ifdef GPAW_CUDA
        kernel_assert(__popc(__activemask()) == warpSize, "warp_reduce_sum called by a partial warp");
        const unsigned mask = __activemask();
        #pragma unroll
        for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(mask, val, offset);
        }

    #else // HIP
        kernel_assert(__popcll(__ballot(1)) == warpSize, "warp_reduce_sum called by a partial warp");
        #pragma unroll
        for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down(val, offset);
        }
    #endif

        return val;
    }

    /**
     * Performs sum-reduction of val across every thread of the calling thread block.
     * Every thread in the block must call this with the same shared_buf.
     * shared_buf must be a __shared__ array of at least kernel_utils::MAX_WARPS_PER_BLOCK elements.
     * Only the result on the thread with linear index 0 is meaningful.
     * This is a blocking routine: shared_buf is safe to modify once this function returns.
     */
    template<typename T>
    __device__ T block_reduce_sum(T val, T* shared_buf)
    {
        const int linear_tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * blockDim.y * blockDim.z;
        const int lane = linear_tid % warpSize;
        const int warp_id = linear_tid / warpSize;
        const int num_warps = (num_threads + warpSize - 1) / warpSize;

        // Reduce at warp level, storing the result for warp n warp at shared_buf[n]
        val = warp_reduce_sum(val);

        if (lane == 0)
        {
            shared_buf[warp_id] = val;
        }
        __syncthreads();

        // Warp 0 sums the per-warp results: tid `i` in warp 0 loads result of warp `i`
        val = (linear_tid < num_warps) ? shared_buf[linear_tid] : static_cast<T>(0);
        if (warp_id == 0)
        {
            val = warp_reduce_sum(val);
        }
        // Barrier to ensure shared_buf is safe to modify once this function exits.
        __syncthreads();

        return val;
    }

} // namespace kernel_utils


template<typename T>
struct AddToDensityArrays
{
    T* GPAW_RESTRICT nt_sG;
    const T* GPAW_RESTRICT f_sM;
    int32_t num_spins;
    // Needed to index f_sM[s * ... + mu]. The stride is in BYTES so take that into account.
    int64_t f_sM_first_stride;
};

template<typename T>
struct ConstructDensityArrays
{
    T* GPAW_RESTRICT nt_G;
    const T* GPAW_RESTRICT rho_MM;
    // Column count of rho_MM (= Mmax); used for striding
    int64_t ncols;
    // Row distribution: processes only rows [mu_start, mu_end) of rho_MM
    int64_t mu_start;
    int64_t mu_end;
};

template<typename T>
struct CalculatePotentialMatrixArrays
{
    const T* GPAW_RESTRICT vt_G;
    T* GPAW_RESTRICT V_MN;
    // Row distribution: only rows [mu_start, mu_end) are computed
    int64_t mu_start;
    int64_t mu_end;
    // Column count of V_MN (= Mmax); used for striding
    int64_t ncols;
};

// Helpers for computing spherical harmonics
struct SphericalHarmonicStatics
{
    // We don't support Y_lm with l larger than this
    static constexpr uint8_t max_l = 4;
    static constexpr uint8_t OUT_ARRAY_SIZE = 2*max_l + 1;

    /* Computes f * Y_lm(R) for all allowed values of m. In practice, f will be the radial spline and this then gives the associated basis function(s) phi_lm(R).
    Only the first 2*l + 1 elements of the output array will be touched.
    x, y, z are the components of R and r2 is the length-squared of this vector. r2 is assumed to be precomputed anyway for spline evaluation. */
    template<typename T>
    __device__ static void fr_times_Ylm(uint8_t l, T f, T x, T y, T z, T r2, T (&out_m)[OUT_ARRAY_SIZE])
    {
        switch (l)
        {
        case 0:
            out_m[0] = f * 0.28209479177387814;
            break;
        case 1:
            out_m[0] = f * 0.48860251190291992 * y;
            out_m[1] = f * 0.48860251190291992 * z;
            out_m[2] = f * 0.48860251190291992 * x;
            break;
        case 2:
            out_m[0] = f * 1.0925484305920792 * x*y;
            out_m[1] = f * 1.0925484305920792 * y*z;
            out_m[2] = f * 0.31539156525252005 * (-r2+3*z*z);
            out_m[3] = f * 1.0925484305920792 * x*z;
            out_m[4] = f * 0.54627421529603959 * (-y*y+x*x);
            break;
        case 3:
            out_m[0] = f * 0.59004358992664352 * (-y*y*y+3*x*x*y);
            out_m[1] = f * 2.8906114426405538 * x*y*z;
            out_m[2] = f * 0.45704579946446577 * (5*y*z*z-y*r2);
            out_m[3] = f * 0.3731763325901154 * (-3*z*r2+5*z*z*z);
            out_m[4] = f * 0.45704579946446577 * (-x*r2+5*x*z*z);
            out_m[5] = f * 1.4453057213202769 * (-y*y*z+x*x*z);
            out_m[6] = f * 0.59004358992664352 * (x*x*x-3*x*y*y);
            break;
        case 4:
            out_m[0] = f * 2.5033429417967046 * (x*x*x*y-x*y*y*y);
            out_m[1] = f * 1.7701307697799307 * (3*x*x*y*z-y*y*y*z);
            out_m[2] = f * 0.94617469575756008 * (-x*y*r2+7*x*y*z*z);
            out_m[3] = f * 0.66904654355728921 * (-3*y*z*r2+7*y*z*z*z);
            out_m[4] = f * 0.10578554691520431 * (3*r2*r2-30*z*z*r2+35*z*z*z*z);
            out_m[5] = f * 0.66904654355728921 * (7*x*z*z*z-3*x*z*r2);
            out_m[6] = f * 0.47308734787878004 * (y*y*r2+7*x*x*z*z-x*x*r2-7*y*y*z*z);
            out_m[7] = f * 1.7701307697799307 * (x*x*x*z-3*x*y*y*z);
            out_m[8] = f * 0.62583573544917614 * (-6*x*x*y*y+x*x*x*x+y*y*y*y);
            break;
        default:
            assert(false && "We only support spherical harmonics up to l=4");
        }
    }
};

template<typename T>
struct LFCThreadGridData
{
    // Real-space coordinates of the thread on the full grid
    GpuVec3<T> xyz;
    // Flattened index into 3D grid arrays for this thread
    int64_t grid_array_idx;

    // Number of grid points in this domain. Needed only in some kernels, unused in others (compiler should optimize them out)
    [[maybe_unused]]
    int64_t nx, ny, nz;
};

namespace LFCKernelUtils
{
    /**
     * Common helper for computing coordinates on the simulation grid for a GPU thread.
     * Returns false if the thread is out of grid bounds, in which case the calling kernel should return immediately.
     * @param thread_block_idx index of the calling thread block
     * @param grid the simulation grid struct
     * @param blocks flattened arrays about simulation grid blocking
     * @param out output struct
     */
    template<typename T, bool should_store_domain_sizes = false>
    __device__ bool compute_thread_coords(uint thread_block_idx, const GridInfoGpu& grid, const BlockDataGpu& blocks, LFCThreadGridData<T>& out)
    {
        // In current implementation the thread-block index and the real-grid-block index are the same:
        if (thread_block_idx >= blocks.num_blocks)
        {
            return false;
        }

        const auto block_start_c = blocks.block_start_bc[thread_block_idx];

        // Global indices on the full simulation grid
        const grid_coord_t gx = block_start_c.x + threadIdx.x;
        const grid_coord_t gy = block_start_c.y + threadIdx.y;
        const grid_coord_t gz = block_start_c.z + threadIdx.z;

        if (gx >= grid.end_c.x || gy >= grid.end_c.y || gz >= grid.end_c.z)
        {
            return false;
        }

        // Compute flattened index into 3D grid arrays like vt_G. Strides:
        const int64_t ny = grid.end_c.y - grid.start_c.y;
        const int64_t nz = grid.end_c.z - grid.start_c.z;
        // Offset by grid.start_c to account for domain decomposition
        out.grid_array_idx = ((gx - grid.start_c.x) * ny + (gy - grid.start_c.y)) * nz + (gz - grid.start_c.z);

        // Compute XYZ coords of the grid point in real units
        out.xyz.x = static_cast<T>(gx * grid.spacing_cv[0].x + gy * grid.spacing_cv[1].x + gz * grid.spacing_cv[2].x);
        out.xyz.y = static_cast<T>(gx * grid.spacing_cv[0].y + gy * grid.spacing_cv[1].y + gz * grid.spacing_cv[2].y);
        out.xyz.z = static_cast<T>(gx * grid.spacing_cv[0].z + gy * grid.spacing_cv[1].z + gz * grid.spacing_cv[2].z);

        if constexpr (should_store_domain_sizes)
        {
            out.nx = grid.end_c.x - grid.start_c.x;
            out.ny = ny;
            out.nz = nz;
        }

        return true;
    }

    /**
     * Evaluates all m-components of a basis function phi_i at a given location.
     * @param phi_idx index of the phi to evaluate, ie. index used for phi lookups in PhiDataGpu.
     * @param xyz Real-space coordinates where the basis function should be evaluated.
     * @param phi_data collects flattened lookup arrays for basis function data
     * @param splines pointer to GPU spline lookup table
     * @param out_phi_m output array for the (2*l + 1) components of the evaluated basis function.
     *                  Only the first (2*l + 1) elements will be written to.
     */
    template<typename T>
    __device__ void evaluate_phi(int64_t phi_idx, const GpuVec3<T>& xyz, const PhiDataGpu& phi_data,
        const LFCGpuSpline* GPAW_RESTRICT splines, T (&out_phi_m)[SphericalHarmonicStatics::OUT_ARRAY_SIZE])
    {
        const GpuVec3<double>& phi_pos = phi_data.position_iv[phi_idx];
        const uint8_t l = phi_data.l_i[phi_idx];
        const int64_t spline_idx = phi_data.spline_idx_i[phi_idx];

        const auto& spline = splines[spline_idx];

        // Find distance from the phi center, then evaluate phi(x)
        const T dx = xyz.x - static_cast<T>(phi_pos.x);
        const T dy = xyz.y - static_cast<T>(phi_pos.y);
        const T dz = xyz.z - static_cast<T>(phi_pos.z);
        const T distance_sq = dx*dx + dy*dy + dz*dz;
        const T distance = sqrt(distance_sq);

        const T spline_val = spline.evaluate(distance);

        SphericalHarmonicStatics::fr_times_Ylm(l, spline_val, dx, dy, dz, distance_sq, out_phi_m);
    }
};

template<typename T>
__global__ void add_to_density_kernel(AddToDensityArrays<T> arrays, const LFCGpuSpline* GPAW_RESTRICT splines, GridInfoGpu grid,
    PhiDataGpu phi, BlockDataGpu blocks)
{
    LFCThreadGridData<T> thread_coords;
    if (!LFCKernelUtils::compute_thread_coords<T, true>(blockIdx.x, grid, blocks, thread_coords))
    {
        return;
    }
    const uint block_idx = blockIdx.x;

    // nt_sG(x) += f_sM * |phi_M|^2.
    // Each block does phi_M that it has overlap with and writes to a unique index in the grid array
    // => no atomics or reduction needed

    const int64_t j_begin = blocks.start_j[block_idx];
    const int64_t j_end = blocks.start_j[block_idx + 1];

    // Array indexing is in units of T but the input stride is in bytes
    const int64_t s_stride = arrays.f_sM_first_stride / sizeof(T);

    // Array for holding phi_lm components for all possible m
    T phi_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];

    const int64_t spin_stride = thread_coords.nx * thread_coords.ny * thread_coords.nz;

    for (int32_t s = 0; s < arrays.num_spins; ++s)
    {
        T accum = 0;

        for (int64_t j = j_begin; j < j_end; ++j)
        {
            const int64_t phi_idx = blocks.all_blocks_phi_idx_j[j];
            const uint8_t l = phi.l_i[phi_idx];
            LFCKernelUtils::evaluate_phi<T>(phi_idx, thread_coords.xyz, phi, splines, phi_m);

            const int64_t first_mu = phi.first_mu_i[phi_idx];

            const int32_t num_m = 2*l + 1;
            for (int32_t m = 0; m < num_m; ++m)
            {
                const int64_t mu = first_mu + m;
                accum += arrays.f_sM[s * s_stride + mu] * phi_m[m]*phi_m[m];
            }
        }

        arrays.nt_sG[s * spin_stride + thread_coords.grid_array_idx] += accum;
    }
}

/* nt_G(x) += sum_{mu, nu} phi_mu(x) * rho_{mu,nu} * phi_nu(x)
* Each thread does one grid point x.
*/
template<typename T>
__global__ void construct_density_kernel(ConstructDensityArrays<T> arrays, const LFCGpuSpline* GPAW_RESTRICT splines, GridInfoGpu grid, PhiDataGpu phi, BlockDataGpu blocks)
{
    LFCThreadGridData<T> thread_coords;
    if (!LFCKernelUtils::compute_thread_coords<T>(blockIdx.x, grid, blocks, thread_coords))
    {
        return;
    }
    const uint block_idx = blockIdx.x;

    // Needs a double loop over phi_mu and phi_nu. Let's call them phi_r and phi_s in the following
    const int64_t j_begin = blocks.start_j[block_idx];
    const int64_t j_end = blocks.start_j[block_idx + 1];

    // Each thread writes to a unique grid index so no atomics or reductions needed.
    // Accumulate it locally and write once at the end. This has a huge impact if the compiler fails to apply noalias semantics (clang!)
    T accumulated_sum = static_cast<T>(0.0);

    // phi_r loop
    for (int64_t j_r = j_begin; j_r < j_end; ++j_r)
    {
        const int64_t phi_idx_r = blocks.all_blocks_phi_idx_j[j_r];
        const uint8_t l_r = phi.l_i[phi_idx_r];
        const int64_t first_mu_r = phi.first_mu_i[phi_idx_r];
        T phi_r_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
        LFCKernelUtils::evaluate_phi<T>(phi_idx_r, thread_coords.xyz, phi, splines, phi_r_m);

        // phi_s loop
        for (int64_t j_s = j_begin; j_s < j_end; ++j_s)
        {
            const int64_t phi_idx_s = blocks.all_blocks_phi_idx_j[j_s];
            const uint8_t l_s = phi.l_i[phi_idx_s];
            const int64_t first_mu_s = phi.first_mu_i[phi_idx_s];
            T phi_s_n[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
            LFCKernelUtils::evaluate_phi<T>(phi_idx_s, thread_coords.xyz, phi, splines, phi_s_n);

            // TODO can we do just mu <= nu and symmetrize?

            // Compute phi_r(x) * rho_rs * phi_s(x)
            for (int32_t m = 0; m < 2*l_r + 1; ++m)
            {
                const int64_t mu = first_mu_r + m;
                if (mu < arrays.mu_start || mu >= arrays.mu_end)
                {
                    continue;
                }

                for (int32_t n = 0; n < 2*l_s + 1; ++n)
                {
                    const int64_t nu = first_mu_s + n;
                    // rho_MM only has rows relevant for this rank so must offset the row index
                    const int64_t rho_idx = arrays.ncols * (mu - arrays.mu_start) + nu;
                    accumulated_sum += phi_r_m[m] * phi_s_n[n] * arrays.rho_MM[rho_idx];
                }
            }

        } // phi_s
    } // phi_r

    arrays.nt_G[thread_coords.grid_array_idx] += accumulated_sum;
}

/* Each thread processes one real grid point, computing overlaps of phi_mu and phi_nu for all (mu, nu) that overlap at that point.
* The output array V_MN can receive writes to the same (mu, nu) indices from many threads across many blocks so some atomics are needed.
* We perform a block-wide reduction for each (mu, nu), then write the reduced result with one atomic per block.
*/
template<typename T>
__global__ void calculate_potential_matrix_kernel(CalculatePotentialMatrixArrays<T> arrays, const LFCGpuSpline* GPAW_RESTRICT splines, GridInfoGpu grid, PhiDataGpu phi, BlockDataGpu blocks)
{
    LFCThreadGridData<T> thread_coords;

    // Can happen that some threads lie outside the valid grid region (boundary blocks "too large").
    // Can't early-exit because we still need them to participate in the block reductions below.
    const bool valid = LFCKernelUtils::compute_thread_coords<T>(blockIdx.x, grid, blocks, thread_coords);
    const uint block_idx = blockIdx.x;

    const T vt_dV = valid ? arrays.vt_G[thread_coords.grid_array_idx] * grid.dV : static_cast<T>(0);

    // Needs to integrate phi_mu * vt * phi_nu, so we need a double loop over the phi index.
    // Call them phi_r and phi_s in the following to avoid confusion with m, n and mu, nu that we already use.
    const int64_t j_begin = blocks.start_j[block_idx];
    const int64_t j_end = blocks.start_j[block_idx + 1];

    // Scratch space for the block-wide reductions in the loop below
    __shared__ T reduce_buf[kernel_utils::MAX_WARPS_PER_BLOCK];

    // phi_r loop
    for (int64_t j_r = j_begin; j_r < j_end; ++j_r)
    {
        const int64_t phi_idx_r = blocks.all_blocks_phi_idx_j[j_r];
        const uint8_t l_r = phi.l_i[phi_idx_r];
        const int64_t first_mu_r = phi.first_mu_i[phi_idx_r];
        T phi_r_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE] = {};
        if (valid)
        {
            LFCKernelUtils::evaluate_phi<T>(phi_idx_r, thread_coords.xyz, phi, splines, phi_r_m);
        }

        // phi_s loop
        for (int64_t j_s = j_begin; j_s < j_end; ++j_s)
        {
            const int64_t phi_idx_s = blocks.all_blocks_phi_idx_j[j_s];
            const uint8_t l_s = phi.l_i[phi_idx_s];
            const int64_t first_mu_s = phi.first_mu_i[phi_idx_s];
            T phi_s_n[SphericalHarmonicStatics::OUT_ARRAY_SIZE] = {};
            if (valid)
            {
                LFCKernelUtils::evaluate_phi<T>(phi_idx_s, thread_coords.xyz, phi, splines, phi_s_n);
            }

            // Accumulate phi_rm * vt * phi_sn for each phi component m,n.
            // All threads loop m,n (and consequently mu, nu) in the same order so the reduction below works as intended.
            for (int32_t m = 0; m < 2*l_r + 1; ++m)
            {
                const int64_t mu = first_mu_r + m;
                // temp until parallel V_MN properly implemented?
                if (mu < arrays.mu_start || mu >= arrays.mu_end)
                {
                    continue;
                }

                const T phi_rm_times_vt = phi_r_m[m] * vt_dV;

                for (int32_t n = 0; n < 2*l_s + 1; ++n)
                {
                    const int64_t nu = first_mu_s + n;
                    if (mu >= nu)
                    {
                        const T phi_vt_phi = phi_rm_times_vt * phi_s_n[n];
                        // block-wide reduction for this (mu, nu) result
                        const T block_sum = kernel_utils::block_reduce_sum<T>(phi_vt_phi, reduce_buf);
                        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
                        {
                            atomicAdd(&arrays.V_MN[arrays.ncols * (mu - arrays.mu_start) + nu], block_sum);
                        }
                    }
                }
            }
        } // j_s (phi_nu)
    } // j_r (phi_mu)

}

/** Variant of calculate_potential_matrix_kernel that assumes one thread block per overlapping (grid_block, phi_i, phi_j) triplet.
 * This should be better for compute coherence (less warp divergence) at the cost of larger memory fetch requirement.
 * The other kernel has exactly one fetch of vt_G[G] for each grid point G, this kernel needs to re-fetch it for each phi pair in that block.
 * Similar block-reduction-to-atomic is needed here because neighboring blocks may want to write to same (mu, nu) index.
 */
template<typename T>
__global__ void calculate_potential_matrix_phi_pairs_kernel(
    CalculatePotentialMatrixArrays<T> arrays,
    const LFCGpuSpline* GPAW_RESTRICT splines,
    GridInfoGpu grid,
    PhiDataGpu phi,
    BlockDataGpu blocks,
    PhiPairDataGpu pairs)
{
    const int64_t pair_idx = blockIdx.x;
    if (pair_idx >= pairs.num_pairs)
    {
        return;
    }

    const int64_t block_idx = pairs.block_idx_p[pair_idx];
    const int64_t phi_idx_r = pairs.phi_idx_r_p[pair_idx];
    const int64_t phi_idx_s = pairs.phi_idx_s_p[pair_idx];

    // Metadata for the phi pair, uniform across the thread block
    const uint8_t l_r = phi.l_i[phi_idx_r];
    const uint8_t l_s = phi.l_i[phi_idx_s];
    const int64_t first_mu_r = phi.first_mu_i[phi_idx_r];
    const int64_t first_mu_s = phi.first_mu_i[phi_idx_s];

    const int64_t m_max =  (2*l_r + 1);
    const int64_t mu_end_r = first_mu_r + m_max;

    // TODO Some optimization possibilities below if we require matrix row dist to be fixed during LFC init time

    if (first_mu_r >= arrays.mu_end || mu_end_r <= arrays.mu_start)
    {
        // Entire phi_r mu range is outside the assigned row distribution.
        // This is an uniform early exit across the block
        return;
    }

    LFCThreadGridData<T> thread_coords;
    const bool valid = LFCKernelUtils::compute_thread_coords<T>(block_idx, grid, blocks, thread_coords);

    const T vt_dV = valid ? arrays.vt_G[thread_coords.grid_array_idx] * grid.dV : static_cast<T>(0);

    T phi_r_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE] = {};
    T phi_s_n[SphericalHarmonicStatics::OUT_ARRAY_SIZE] = {};
    if (valid)
    {
        LFCKernelUtils::evaluate_phi<T>(phi_idx_r, thread_coords.xyz, phi, splines, phi_r_m);
        LFCKernelUtils::evaluate_phi<T>(phi_idx_s, thread_coords.xyz, phi, splines, phi_s_n);
    }

    // Scratch space for the block-wide reductions in the loop below
    __shared__ T reduce_buf[kernel_utils::MAX_WARPS_PER_BLOCK];

    for (int32_t m = 0; m < m_max; ++m)
    {
        const int64_t mu = first_mu_r + m;
        if (mu < arrays.mu_start || mu >= arrays.mu_end)
        {
            continue;
        }

        const T phi_rm_vt = phi_r_m[m] * vt_dV;

        for (int32_t n = 0; n < 2*l_s + 1; ++n)
        {
            const int64_t nu = first_mu_s + n;
            if (mu < nu)
            {
                continue;
            }

            const T contrib = phi_rm_vt * phi_s_n[n];
            const T block_sum = kernel_utils::block_reduce_sum<T>(contrib, reduce_buf);
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                atomicAdd(&arrays.V_MN[arrays.ncols * (mu - arrays.mu_start) + nu], block_sum);
            }
        }
    }

}
