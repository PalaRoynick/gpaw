#pragma once

// Templated implementations for LFC GPU kernels. This is intended to be included only from lfc_gpu.cu

#include "lfc_gpu.cuh"
#include "lfc_gpu_kernels.cuh"
#include "spline_pool_gpu.cuh"
#include "gpu/cpp/gpu_core.hpp"

class LFCGpuImpl
{
public:

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
    static void launch_construct_density(
        const LFCGpu& lfc,
        gpaw::PyDeviceArray& nt_G,
        const gpaw::PyDeviceArray& rho_MM,
        size_t mu_start,
        size_t mu_end
    )
    {
        ConstructDensityArrays<T> inputs;
        inputs.nt_G = static_cast<T*>(nt_G.data);
        inputs.rho_MM = static_cast<T*>(rho_MM.data);
        inputs.ncols = static_cast<size_t>(rho_MM.shape[1]);
        inputs.mu_start = mu_start;
        inputs.mu_end = mu_end;

        const dim3 threads(lfc.block_data.block_shape.x,
                           lfc.block_data.block_shape.y,
                           lfc.block_data.block_shape.z);
        const dim3 blocks(lfc.block_data.num_blocks);

        gpaw::launch_kernel(construct_density_kernel<T>, blocks, threads, 0, nullptr,
            inputs,
            lfc.spline_pool->get_device_splines(),
            lfc.grid, lfc.phi_data, lfc.block_data);
    }

    template<typename T>
    static void launch_calculate_potential_matrix(
        const LFCGpu& lfc,
        const gpaw::PyDeviceArray& vt_G,
        gpaw::PyDeviceArray& out_V_MN,
        size_t mu_start,
        size_t mu_end)
    {
        CalculatePotentialMatrixArrays<T> arrays;
        arrays.vt_G = static_cast<const T*>(vt_G.data);
        arrays.V_MN = static_cast<T*>(out_V_MN.data);
        arrays.mu_start = mu_start;
        arrays.mu_end = mu_end;
        arrays.ncols = static_cast<size_t>(out_V_MN.shape[1]);

        const dim3 threads(lfc.block_data.block_shape.x,
                           lfc.block_data.block_shape.y,
                           lfc.block_data.block_shape.z);
        const dim3 blocks(lfc.block_data.num_blocks);

        gpaw::launch_kernel(calculate_potential_matrix_kernel<T>, blocks, threads, 0, nullptr,
            arrays,
            lfc.spline_pool->get_device_splines(),
            lfc.grid, lfc.phi_data, lfc.block_data);
    }
};
