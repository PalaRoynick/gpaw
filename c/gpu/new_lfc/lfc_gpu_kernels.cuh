#pragma once
#include "gpu/gpu-runtime.h"
#include "gpaw_utils.h"
#include "lfc_gpu.cuh"

// FIXME tons of code duplication in the kernels. Eg. when computing grid array strides and computing distances for splines and spherical harmonics

/* atomicAdd not supported for doubles on devices with compute capability < 6.0.
* These are ancient, but currently the CI build still includes these devices since the siteconfigs there don't specify the arch.
* This implementation is from https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#atomic-functions.
* Note that it violates strict aliasing!
*/
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    }
    while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template<typename T>
struct AddToDensityArrays
{
    T* GPAW_RESTRICT nt_sG;
    const T* GPAW_RESTRICT f_sM;
    int32_t num_spins;
    // Needed to index f_sM[s * ... + mu]. The stride is in BYTES so take that into account.
    size_t f_sM_first_stride;
};

template<typename T>
struct ConstructDensityArrays
{
    T* GPAW_RESTRICT nt_G;
    const T* GPAW_RESTRICT rho_MM;
    // Column count of rho_MM (= Mmax); used for striding
    size_t ncols;
    // Row distribution: processes only rows [mu_start, mu_end) of rho_MM
    size_t mu_start;
    size_t mu_end;
};

template<typename T>
struct CalculatePotentialMatrixArrays
{
    const T* GPAW_RESTRICT vt_G;
    T* GPAW_RESTRICT V_MN;
    // Row distribution: only rows [mu_start, mu_end) are computed
    size_t mu_start;
    size_t mu_end;
    // Column count of V_MN (= Mmax); used for striding
    size_t ncols;
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
    size_t grid_array_idx;

    // Number of grid points in this domain. Needed only in some kernels, unused in others (compiler should optimize them out)
    [[maybe_unused]]
    long nx, ny, nz;
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
        const int gx = block_start_c.x + threadIdx.x;
        const int gy = block_start_c.y + threadIdx.y;
        const int gz = block_start_c.z + threadIdx.z;

        if (gx >= grid.end_c.x || gy >= grid.end_c.y || gz >= grid.end_c.z)
        {
            return false;
        }

        // Compute flattened index into 3D grid arrays like vt_G. Strides:
        const long ny = grid.end_c.y - grid.start_c.y;
        const long nz = grid.end_c.z - grid.start_c.z;
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
    __device__ void evaluate_phi(size_t phi_idx, const GpuVec3<T>& xyz, const PhiDataGpu& phi_data,
        const LFCGpuSpline* GPAW_RESTRICT splines, T (&out_phi_m)[SphericalHarmonicStatics::OUT_ARRAY_SIZE])
    {
        const GpuVec3<double>& phi_pos = phi_data.position_iv[phi_idx];
        const uint8_t l = phi_data.l_i[phi_idx];
        const size_t spline_idx = phi_data.spline_idx_i[phi_idx];

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

    const size_t j_begin = blocks.start_j[block_idx];
    const size_t j_end = blocks.start_j[block_idx + 1];

    // Array indexing is in units of T but the input stride is in bytes
    const int64_t s_stride = arrays.f_sM_first_stride / sizeof(T);

    // Array for holding phi_lm components for all possible m
    T phi_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];

    const long spin_stride = thread_coords.nx * thread_coords.ny * thread_coords.nz;

    for (int s = 0; s < arrays.num_spins; ++s)
    {
        T accum = 0;

        for (size_t j = j_begin; j < j_end; ++j)
        {
            const size_t phi_idx = blocks.all_blocks_phi_idx_j[j];
            const uint8_t l = phi.l_i[phi_idx];
            LFCKernelUtils::evaluate_phi<T>(phi_idx, thread_coords.xyz, phi, splines, phi_m);

            const size_t first_mu = phi.first_mu_i[phi_idx];

            const int num_m = 2*l + 1;
            for (int m = 0; m < num_m; ++m)
            {
                const int mu = first_mu + m;
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
    const size_t j_begin = blocks.start_j[block_idx];
    const size_t j_end = blocks.start_j[block_idx + 1];

    // Each thread writes to a unique grid index so no atomics or reductions needed.
    // Accumulate it locally and write once at the end. This has a huge impact if the compiler fails to apply noalias semantics (clang!)
    T accumulated_sum = static_cast<T>(0.0);

    // phi_r loop
    for (size_t j_r = j_begin; j_r < j_end; ++j_r)
    {
        const size_t phi_idx_r = blocks.all_blocks_phi_idx_j[j_r];
        const uint8_t l_r = phi.l_i[phi_idx_r];
        const size_t first_mu_r = phi.first_mu_i[phi_idx_r];
        T phi_r_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
        LFCKernelUtils::evaluate_phi<T>(phi_idx_r, thread_coords.xyz, phi, splines, phi_r_m);

        // phi_s loop
        for (size_t j_s = j_begin; j_s < j_end; ++j_s)
        {
            const size_t phi_idx_s = blocks.all_blocks_phi_idx_j[j_s];
            const uint8_t l_s = phi.l_i[phi_idx_s];
            const size_t first_mu_s = phi.first_mu_i[phi_idx_s];
            T phi_s_n[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
            LFCKernelUtils::evaluate_phi<T>(phi_idx_s, thread_coords.xyz, phi, splines, phi_s_n);

            // TODO can we do just mu <= nu and symmetrize?

            // Compute phi_r(x) * rho_rs * phi_s(x)
            for (int m = 0; m < 2*l_r + 1; ++m)
            {
                const size_t mu = first_mu_r + m;
                if (mu < arrays.mu_start || mu >= arrays.mu_end)
                {
                    continue;
                }

                for (int n = 0; n < 2*l_s + 1; ++n)
                {
                    const size_t nu = first_mu_s + n;
                    // rho_MM only has rows relevant for this rank so must offset the row index
                    const size_t rho_idx = arrays.ncols * (mu - arrays.mu_start) + nu;
                    accumulated_sum += phi_r_m[m] * phi_s_n[n] * arrays.rho_MM[rho_idx];
                }
            }

        } // phi_s
    } // phi_r

    arrays.nt_G[thread_coords.grid_array_idx] += accumulated_sum;
}

template<typename T>
__global__ void calculate_potential_matrix_kernel(CalculatePotentialMatrixArrays<T> arrays, const LFCGpuSpline* GPAW_RESTRICT splines, GridInfoGpu grid, PhiDataGpu phi, BlockDataGpu blocks)
{
    LFCThreadGridData<T> thread_coords;
    if (!LFCKernelUtils::compute_thread_coords<T>(blockIdx.x, grid, blocks, thread_coords))
    {
        return;
    }
    const uint block_idx = blockIdx.x;

    const T vt_dV = arrays.vt_G[thread_coords.grid_array_idx] * grid.dV;

    // Needs to integrate phi_mu * vt * phi_nu, so we need a double loop over the phi index.
    // Call them phi_r and phi_s in the following to avoid confusion with m, n and mu, nu that we already use.
    const size_t j_begin = blocks.start_j[block_idx];
    const size_t j_end = blocks.start_j[block_idx + 1];

    // phi_r loop
    for (size_t j_r = j_begin; j_r < j_end; ++j_r)
    {
        const size_t phi_idx_r = blocks.all_blocks_phi_idx_j[j_r];
        const uint8_t l_r = phi.l_i[phi_idx_r];
        const size_t first_mu_r = phi.first_mu_i[phi_idx_r];
        T phi_r_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
        LFCKernelUtils::evaluate_phi<T>(phi_idx_r, thread_coords.xyz, phi, splines, phi_r_m);

        // phi_s loop
        for (size_t j_s = j_begin; j_s < j_end; ++j_s)
        {
            const size_t phi_idx_s = blocks.all_blocks_phi_idx_j[j_s];
            const uint8_t l_s = phi.l_i[phi_idx_s];
            const size_t first_mu_s = phi.first_mu_i[phi_idx_s];
            T phi_s_n[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
            LFCKernelUtils::evaluate_phi<T>(phi_idx_s, thread_coords.xyz, phi, splines, phi_s_n);

            // Accumulate phi_rm * vt * phi_sn for each phi component m,n
            for (int m = 0; m < 2*l_r + 1; ++m)
            {
                const size_t mu = first_mu_r + m;
                // temp until parallel V_MN properly implemented?
                if (mu < arrays.mu_start || mu >= arrays.mu_end)
                {
                    continue;
                }

                const T phi_rm_times_vt = phi_r_m[m] * vt_dV;

                for (int n = 0; n < 2*l_s + 1; ++n)
                {
                    const size_t nu = first_mu_s + n;
                    if (mu >= nu)
                    {
                        const T phi_vt_phi = phi_rm_times_vt * phi_s_n[n];
                        atomicAdd(&arrays.V_MN[arrays.ncols * (mu - arrays.mu_start) + nu], phi_vt_phi);
                    }
                }
            }
        } // j_s (phi_nu)
    } // j_r (phi_mu)

}
