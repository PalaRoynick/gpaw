#include "lfc_gpu.cuh"

namespace gpaw
{

/* Output goes to p; does all values of m for given l. r2 is the distance squared and xyz are real-space coords (relative to the phi??)
f is the spline value, apparently. Could use some cleanup! */
#define spherical_harmonics(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0.28209479177387814;\
            break;\
          case 1:\
            p[0] = f * 0.48860251190291992 * y;\
            p[1] = f * 0.48860251190291992 * z;\
            p[2] = f * 0.48860251190291992 * x;\
            break;\
          case 2:\
            p[0] = f * 1.0925484305920792 * x*y;\
            p[1] = f * 1.0925484305920792 * y*z;\
            p[2] = f * 0.31539156525252005 * (-r2+3*z*z);\
            p[3] = f * 1.0925484305920792 * x*z;\
            p[4] = f * 0.54627421529603959 * (-y*y+x*x);\
            break;\
          case 3:\
            p[0] = f * 0.59004358992664352 * (-y*y*y+3*x*x*y);\
            p[1] = f * 2.8906114426405538 * x*y*z;\
            p[2] = f * 0.45704579946446577 * (5*y*z*z-y*r2);\
            p[3] = f * 0.3731763325901154 * (-3*z*r2+5*z*z*z);\
            p[4] = f * 0.45704579946446577 * (-x*r2+5*x*z*z);\
            p[5] = f * 1.4453057213202769 * (-y*y*z+x*x*z);\
            p[6] = f * 0.59004358992664352 * (x*x*x-3*x*y*y);\
            break;\
          case 4:\
            p[0] = f * 2.5033429417967046 * (x*x*x*y-x*y*y*y);\
            p[1] = f * 1.7701307697799307 * (3*x*x*y*z-y*y*y*z);\
            p[2] = f * 0.94617469575756008 * (-x*y*r2+7*x*y*z*z);\
            p[3] = f * 0.66904654355728921 * (-3*y*z*r2+7*y*z*z*z);\
            p[4] = f * 0.10578554691520431 * (3*r2*r2-30*z*z*r2+35*z*z*z*z);\
            p[5] = f * 0.66904654355728921 * (7*x*z*z*z-3*x*z*r2);\
            p[6] = f * 0.47308734787878004 * (y*y*r2+7*x*x*z*z-x*x*r2-7*y*y*z*z);\
            p[7] = f * 1.7701307697799307 * (x*x*x*z-3*x*y*y*z);\
            p[8] = f * 0.62583573544917614 * (-6*x*x*y*y+x*x*x*x+y*y*y*y);\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

template<typename T>
__global__ void add_to_density_kernel(AddToDensityArrays<T> arrays, const TRadialSplineGpu<T>* splines, GridInfoGpu grid,
    PhiDataGpu<T> phi, BlockDataGpu blocks)
{
    const int block_idx = blockIdx.x;
    if (block_idx >= blocks.num_blocks)
    {
        return;
    }

    const long3 block_start_c = blocks.block_start_c[block_idx];
    const long3 block_shape = blocks.block_shape[block_idx];

    // Thread idx within the block; could go out of real-block bounds so check for that
    const int bx = threadIdx.x;
    const int by = threadIdx.y;
    const int bz = threadIdx.z;

    if (bx >= block_shape.x || by >= block_shape.y || bz >= block_shape.z)
    {
        return;
    }

    // // XYZ coords on the real grid for this thread
    // const int x = block_start_c.x + bx;
    // const int y = block_start_c.y + by;
    // const int z = block_start_c.z + bz;
    // XYZ of this thread in real units
    const T x = grid.spacing.x * (block_start_c.x + bx);
    const T y = grid.spacing.y * (block_start_c.y + by);
    const T z = grid.spacing.z * (block_start_c.z + bz);

    // Compute index into (MPI-distributed) grid arrays for this thread, and the total number of grid points (for striding)
    // TODO double check if block_start_c is relative to full grid or the MPI domain...? I think its full grid
    const int gx = block_start_c.x + bx;
    const int gy = block_start_c.y + by;
    const int gz = block_start_c.z + bz;

    const long nx = grid.end_c.x - grid.start_c.x;
    const long ny = grid.end_c.y - grid.start_c.y;
    const long nz = grid.end_c.z - grid.start_c.z;
    const size_t grid_array_idx = (gx * ny + gy) * nz + gz;


    // nt_sG(x) += f_sM * |phi_M|^2.
    // Each block does phi_M that it has overlap with and writes to a unique index in the grid array
    // => no atomics or reduction needed

    const size_t j_begin = blocks.start_j[block_idx];
    const size_t j_end = blocks.start_j[block_idx + 1];

    // const size_t m_begin = blocks.start_m[block_idx];
    // const size_t m_end = blocks.start_m[block_idx + 1];


    // spherical_harmonics macro wants to fill at max 9 elements
    T phi_val[9];

    for (int s = 0; s < arrays.num_spins; ++s)
    {
        T accum = 0;

        for (size_t j = j_begin; j < j_end; ++j)
        {
            const size_t phi_idx = blocks.all_blocks_phi_idx_j[j];

            const GpuVec3<T>& phi_pos = phi.position_iv[phi_idx];
            const uint8_t l = phi.l_i[phi_idx];
            const size_t first_mu = phi.first_mu_i[phi_idx];
            const size_t spline_idx = phi.spline_idx_i[phi_idx];

            const TRadialSplineGpu<T>& spline = splines[spline_idx];

            // Find distance from the phi center, then evaluate phi(x)
            T dx = x - phi_pos.x;
            T dy = y - phi_pos.y;
            T dz = z - phi_pos.z;
            T distance_sq = dx*dx + dy*dy + dz*dz;
            T distance = sqrt(distance_sq);

            // TODO remove branch inside evaluate?
            const T spline_val = spline.evaluate(distance);

            spherical_harmonics(l, spline_val, dx, dy, dz, distance_sq, phi_val);

            const int num_m = 2*l + 1;
            for (int m = 0; m < num_m; ++m)
            {
                const int mu = first_mu + m;
                accum += arrays.f_sM[s * arrays.f_sM_first_stride + mu] * phi_val[m]*phi_val[m];
            }
        }

        arrays.nt_sG[s * nx * ny * nz + grid_array_idx] += accum;
    }
}


} // namespace gpaw
