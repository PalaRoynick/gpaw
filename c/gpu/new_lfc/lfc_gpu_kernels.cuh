#pragma once
#include "gpu/gpu-runtime.h"
#include "gpaw_utils.h"
#include "lfc_gpu.cuh"

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

/* Real-space XYZ coords for grid point (gx, gy, gz) on a parallelepipid grid.
* spacing_cv[c] is the Cartesian displacement vector for one step along grid axis c. */
template<typename T>
__device__ inline void grid_coords_to_xyz(int gx, int gy, int gz, const GridInfoGpu& grid, T& x, T& y, T& z)
{
    x = static_cast<T>(gx * grid.spacing_cv[0].x + gy * grid.spacing_cv[1].x + gz * grid.spacing_cv[2].x);
    y = static_cast<T>(gx * grid.spacing_cv[0].y + gy * grid.spacing_cv[1].y + gz * grid.spacing_cv[2].y);
    z = static_cast<T>(gx * grid.spacing_cv[0].z + gy * grid.spacing_cv[1].z + gz * grid.spacing_cv[2].z);
}

template<typename T>
__global__ void add_to_density_kernel(AddToDensityArrays<T> arrays, const LFCGpuSpline* splines, GridInfoGpu grid,
    PhiDataGpu phi, BlockDataGpu blocks)
{
    const int block_idx = blockIdx.x;
    if (block_idx >= blocks.num_blocks)
    {
        return;
    }

    const auto block_start_c = blocks.block_start_bc[block_idx];

    // Thread idx within the block
    const int bx = threadIdx.x;
    const int by = threadIdx.y;
    const int bz = threadIdx.z;

    // Each thread does one grid point. gx, gy, gz are the global grid indices for this thread
    const int gx = block_start_c.x + bx;
    const int gy = block_start_c.y + by;
    const int gz = block_start_c.z + bz;

    // Return here if grid point is outside our domain (can happen if the blocking was uneven)
    if (gx >= grid.end_c.x || gy >= grid.end_c.y || gz >= grid.end_c.z)
    {
        return;
    }

    T x, y, z;
    grid_coords_to_xyz(gx, gy, gz, grid, x, y, z);

    const long nx = grid.end_c.x - grid.start_c.x;
    const long ny = grid.end_c.y - grid.start_c.y;
    const long nz = grid.end_c.z - grid.start_c.z;
    // Compute index into grid arrays. Note offsetting by grid.start_c to get back to "domain local coordinates"
    const size_t grid_array_idx = ((gx - grid.start_c.x) * ny + (gy - grid.start_c.y)) * nz + (gz - grid.start_c.z);


    // nt_sG(x) += f_sM * |phi_M|^2.
    // Each block does phi_M that it has overlap with and writes to a unique index in the grid array
    // => no atomics or reduction needed

    const size_t j_begin = blocks.start_j[block_idx];
    const size_t j_end = blocks.start_j[block_idx + 1];

    // const size_t m_begin = blocks.start_m[block_idx];
    // const size_t m_end = blocks.start_m[block_idx + 1];


    // Array indexing is in units of T but the input stride is in bytes
    const int64_t s_stride = arrays.f_sM_first_stride / sizeof(T);

    // Array for holding phi_lm components for all possible m
    T phi_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];

    for (int s = 0; s < arrays.num_spins; ++s)
    {
        T accum = 0;

        for (size_t j = j_begin; j < j_end; ++j)
        {
            const size_t phi_idx = blocks.all_blocks_phi_idx_j[j];

            const GpuVec3<double>& phi_pos = phi.position_iv[phi_idx];
            const uint8_t l = phi.l_i[phi_idx];
            const size_t first_mu = phi.first_mu_i[phi_idx];
            const size_t spline_idx = phi.spline_idx_i[phi_idx];

            const auto& spline = splines[spline_idx];

            // Find distance from the phi center, then evaluate phi(x)
            T dx = x - static_cast<T>(phi_pos.x);
            T dy = y - static_cast<T>(phi_pos.y);
            T dz = z - static_cast<T>(phi_pos.z);
            T distance_sq = dx*dx + dy*dy + dz*dz;
            T distance = sqrt(distance_sq);

            // TODO remove branch inside evaluate?
            const T spline_val = spline.evaluate(distance);

            SphericalHarmonicStatics::fr_times_Ylm(l, spline_val, dx, dy, dz, distance_sq, phi_m);

            const int num_m = 2*l + 1;
            for (int m = 0; m < num_m; ++m)
            {
                const int mu = first_mu + m;
                accum += arrays.f_sM[s * s_stride + mu] * phi_m[m]*phi_m[m];
            }
        }

        arrays.nt_sG[s * nx * ny * nz + grid_array_idx] += accum;
    }
}

template<typename T>
__global__ void calculate_potential_matrix(CalculatePotentialMatrixArrays<T> arrays, const LFCGpuSpline* splines, GridInfoGpu grid, PhiDataGpu phi, BlockDataGpu blocks)
{
    const int block_idx = blockIdx.x;
    if (block_idx >= blocks.num_blocks)
    {
        return;
    }

    const auto block_start_c = blocks.block_start_bc[block_idx];
    const auto block_shape = blocks.block_shape;

    // Thread idx within the block; could go out of real-block bounds so check for that
    const int bx = threadIdx.x;
    const int by = threadIdx.y;
    const int bz = threadIdx.z;

    if (bx >= block_shape.x || by >= block_shape.y || bz >= block_shape.z)
    {
        return;
    }

    const int gx = block_start_c.x + bx;
    const int gy = block_start_c.y + by;
    const int gz = block_start_c.z + bz;

    if (gx >= grid.end_c.x || gy >= grid.end_c.y || gz >= grid.end_c.z)
    {
        return;
    }

    T x, y, z;
    grid_coords_to_xyz(gx, gy, gz, grid, x, y, z);

    //const long nx = grid.end_c.x - grid.start_c.x;
    const long ny = grid.end_c.y - grid.start_c.y;
    const long nz = grid.end_c.z - grid.start_c.z;
    const size_t grid_array_idx = ((gx - grid.start_c.x) * ny + (gy - grid.start_c.y)) * nz + (gz - grid.start_c.z);


    // Array for holding phi_lm components for all possible m
    T phi_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];

    const T vt = arrays.vt_G[grid_array_idx];

    // Needs double loop over phi index. The following is not great because it computes the each spline more than once.
    // TODO: more optimal would be to precalculate all phi for this gridpoint ?

    const size_t j_begin = blocks.start_j[block_idx];
    const size_t j_end = blocks.start_j[block_idx + 1];

    for (size_t jm = j_begin; jm < j_end; ++jm)
    {
        const size_t phi_idx = blocks.all_blocks_phi_idx_j[jm];

        const GpuVec3<double>& phi_pos = phi.position_iv[phi_idx];
        const uint8_t l_m = phi.l_i[phi_idx];
        const size_t first_mu = phi.first_mu_i[phi_idx];
        const size_t spline_idx = phi.spline_idx_i[phi_idx];

        const auto& spline = splines[spline_idx];

        // Find distance from the phi center, then evaluate phi(x)
        T dx = x - static_cast<T>(phi_pos.x);
        T dy = y - static_cast<T>(phi_pos.y);
        T dz = z - static_cast<T>(phi_pos.z);
        T distance_sq = dx*dx + dy*dy + dz*dz;
        T distance = sqrt(distance_sq);

        const T spline_val_m = spline.evaluate(distance);
        SphericalHarmonicStatics::fr_times_Ylm(l_m, spline_val_m, dx, dy, dz, distance_sq, phi_m);

        const int num_m = 2*l_m + 1;

        for (int m = 0; m < num_m; ++m)
        {
            const size_t mu = m + first_mu;
            const T phi_mu_times_vt = phi_m[m] * vt;

            // Inner phi loop. Basically repeats the phi(x) from above...
            for (size_t jn = j_begin; jn < j_end; ++jn)
            {
                const size_t phi_idx_n = blocks.all_blocks_phi_idx_j[jn];

                const GpuVec3<double>& phi_pos_n = phi.position_iv[phi_idx_n];
                const uint8_t l_n = phi.l_i[phi_idx_n];
                const size_t first_nu = phi.first_mu_i[phi_idx_n];
                const size_t spline_idx_n = phi.spline_idx_i[phi_idx_n];

                const auto& spline_n = splines[spline_idx_n];
                T dx_n = x - static_cast<T>(phi_pos_n.x);
                T dy_n = y - static_cast<T>(phi_pos_n.y);
                T dz_n = z - static_cast<T>(phi_pos_n.z);
                T distance_sq_n = dx_n*dx_n + dy_n*dy_n + dz_n*dz_n;
                T distance_n = sqrt(distance_sq_n);

                T phi_n[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
                const T spline_val_n = spline_n.evaluate(distance_n);
                SphericalHarmonicStatics::fr_times_Ylm(l_n, spline_val_n, dx_n, dy_n, dz_n, distance_sq_n, phi_n);

                const int num_n = 2*l_n + 1;
                for (int n = 0; n < num_n; ++n)
                {
                    const size_t nu = first_nu + n;

                    if (mu >= nu && mu >= arrays.mu_start && mu < arrays.mu_end)
                    {
                        const T phi_vt_phi = phi_mu_times_vt * phi_n[n] * grid.dV;
                        atomicAdd(&arrays.V_MN[arrays.ncols * (mu - arrays.mu_start) + nu], phi_vt_phi);
                    }

                }

            }
        }

    }

}
