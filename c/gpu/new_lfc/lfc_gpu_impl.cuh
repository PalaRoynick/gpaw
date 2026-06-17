#pragma once

// Templated implementations for LFC GPU kernels. This is intended to be included only from lfc_gpu.cu

#include "lfc_gpu.cuh"
#include "lfc_gpu_kernels.cuh"
#include "spline_pool_gpu.cuh"
#include "gpu/cpp/gpu_core.hpp"

struct LFCGpuImpl
{
    template<typename T>
    static void launch_add_to_density(
        const LFCGpu& lfc,
        gpaw::PyDeviceArray& nt_sG,
        const gpaw::PyDeviceArray& f_sM)
    {
        AddToDensityArrays<T> arrays;
        arrays.nt_sG = static_cast<T*>(nt_sG.data);
        arrays.f_sM = static_cast<const T*>(f_sM.data);
        arrays.num_spins = static_cast<int32_t>(nt_sG.shape[0]);
        arrays.f_sM_first_stride = f_sM.strides[0];

        // FIXME will break if blocks are too large.
        // TODO should have some generic kernel_traits struct with constexpr block size etc
        const dim3 threads(lfc.block_data.block_shape.x,
                           lfc.block_data.block_shape.y,
                           lfc.block_data.block_shape.z);
        const dim3 blocks(lfc.block_data.num_blocks);

        gpaw::launch_kernel(add_to_density_kernel<T>, blocks, threads, 0, nullptr,
            arrays,
            lfc.spline_pool->get_device_splines(),
            lfc.grid, lfc.phi_data, lfc.block_data);
    }

    template<typename T>
    static void launch_calculate_potential_matrix(
        const LFCGpu& lfc,
        const gpaw::PyDeviceArray& vt_G,
        gpaw::PyDeviceArray& out_V_MN)
    {
        CalculatePotentialMatrixArrays<T> arrays;
        arrays.vt_G = static_cast<const T*>(vt_G.data);
        arrays.V_MN = static_cast<T*>(out_V_MN.data);
        arrays.max_mu = out_V_MN.shape[0];

        const dim3 threads(lfc.block_data.block_shape.x,
                           lfc.block_data.block_shape.y,
                           lfc.block_data.block_shape.z);
        const dim3 blocks(lfc.block_data.num_blocks);

        gpaw::launch_kernel(calculate_potential_matrix<T>, blocks, threads, 0, nullptr,
            arrays,
            lfc.spline_pool->get_device_splines(),
            lfc.grid, lfc.phi_data, lfc.block_data);
    }
};
