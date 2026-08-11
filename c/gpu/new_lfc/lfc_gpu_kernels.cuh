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

    const T vt_dV = arrays.vt_G[grid_array_idx] * grid.dV;

    // Needs to integrate phi_mu * vt * phi_nu, so we need a double loop over the phi index.
    // Call them phi_r and phi_s in the following to avoid confusion with m, n and mu, nu that we already use.
    const size_t j_begin = blocks.start_j[block_idx];
    const size_t j_end = blocks.start_j[block_idx + 1];

    // phi_r loop
    for (size_t j_r = j_begin; j_r < j_end; ++j_r)
    {
        const size_t phi_idx_r = blocks.all_blocks_phi_idx_j[j_r];

        const GpuVec3<double>& phi_pos_r = phi.position_iv[phi_idx_r];
        const uint8_t l_r = phi.l_i[phi_idx_r];
        const size_t first_mu_r = phi.first_mu_i[phi_idx_r];
        const size_t spline_idx_r = phi.spline_idx_i[phi_idx_r];

        const T dx_r = x - static_cast<T>(phi_pos_r.x);
        const T dy_r = y - static_cast<T>(phi_pos_r.y);
        const T dz_r = z - static_cast<T>(phi_pos_r.z);
        const T distance_sq_r = dx_r*dx_r + dy_r*dy_r + dz_r*dz_r;
        const T distance_r = sqrt(distance_sq_r);

        const T spline_val_r = splines[spline_idx_r].evaluate(distance_r);
        if (spline_val_r == T(0))
        {
            continue;
        }

        // Array for holding phi_lm components for all possible m.
        // TODO this is wasting VGPRs for systems that don't need l=4. We should template based on max l
        T phi_r_m[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
        SphericalHarmonicStatics::fr_times_Ylm(l_r, spline_val_r, dx_r, dy_r, dz_r, distance_sq_r, phi_r_m);

        // phi_s loop
        for (size_t j_s = j_begin; j_s < j_end; ++j_s)
        {
            const size_t phi_idx_s = blocks.all_blocks_phi_idx_j[j_s];

            const GpuVec3<double>& phi_pos_s = phi.position_iv[phi_idx_s];
            const uint8_t l_s = phi.l_i[phi_idx_s];
            const size_t first_mu_s = phi.first_mu_i[phi_idx_s];
            const size_t spline_idx_s = phi.spline_idx_i[phi_idx_s];

            const T dx_s = x - static_cast<T>(phi_pos_s.x);
            const T dy_s = y - static_cast<T>(phi_pos_s.y);
            const T dz_s = z - static_cast<T>(phi_pos_s.z);
            const T distance_sq_s = dx_s*dx_s + dy_s*dy_s + dz_s*dz_s;
            const T distance_s = sqrt(distance_sq_s);

            const T spline_val_s = splines[spline_idx_s].evaluate(distance_s);
            if (spline_val_s == T(0))
            {
                continue;
            }

            T phi_s_n[SphericalHarmonicStatics::OUT_ARRAY_SIZE];
            SphericalHarmonicStatics::fr_times_Ylm(l_s, spline_val_s, dx_s, dy_s, dz_s, distance_sq_s, phi_s_n);

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
