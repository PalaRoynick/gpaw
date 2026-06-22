#include "lfc_builder_gpu.hpp"
#include "lfc_gpu.cuh"
#include "new_lfc/lfc_common.hpp"
#include "new_lfc/lfc_builder.hpp"
#include "spline_pool_gpu.cuh"
#include "gpu/cpp/utils.hpp"
#include <cassert>
#include <memory>

std::unique_ptr<LFCGpu> LFCBuilderGpu::build(LFCBuilder&& cpu_builder, std::shared_ptr<const SplinePoolGpu> spline_pool)
{
    // TODO validations:
    // - Indices in PhiData sensible
    // - BlockData refers sensible phi
    // - no integer overflows: GPU side uses GpuVec3<grid_index_t>

    assert(spline_pool);

    auto out_lfc = std::unique_ptr<LFCGpu>(new LFCGpu());

    // Lifehack to help IDEs that tend to have trouble with smart pointers of template classes
    LFCGpu* lfc = out_lfc.get();

    lfc->spline_pool = std::move(spline_pool);

    // ###### Grid ######
    const GridInfo grid = std::move(cpu_builder.grid);
    lfc->grid.dV = grid.dV;
    for (int32_t i = 0; i < 3; ++i)
    {
        lfc->grid.spacing_cv[i] = make_double3(grid.spacing_cv[i][0], grid.spacing_cv[i][1], grid.spacing_cv[i][2]);
    }

    lfc->grid.start_c = make_gpu_vec3<grid_index_t>(grid.start_c[0], grid.start_c[1], grid.start_c[2]);
    lfc->grid.end_c = make_gpu_vec3<grid_index_t>(grid.end_c[0], grid.end_c[1], grid.end_c[2]);

    // // ###### Blocks ######
    assert(cpu_builder.block_staging.num_blocks() > 0);
    BlockData cpu_blocks = cpu_builder.build_flattened_block_data(std::move(cpu_builder.block_staging));

    namespace gpu = gpaw::gpu;
    // Copy blocks to GPU
    const size_t num_blocks = cpu_blocks.get_num_blocks();
    lfc->block_data.num_blocks = num_blocks;
    lfc->block_data.block_shape = make_gpu_vec3<grid_index_t>(cpu_blocks.block_shape[0], cpu_blocks.block_shape[1], cpu_blocks.block_shape[2]);

    // CUDA/HIP thread block limitation. TODO: separate logical blocking from GPU kernel blocking
    constexpr grid_index_t max_block_points = static_cast<grid_index_t>(1024);
    if (cpu_blocks.block_shape[0] * cpu_blocks.block_shape[1] * cpu_blocks.block_shape[2] > max_block_points)
    {
        throw std::runtime_error(
            "Logical block size exceeds GPU thread block limit of 1024 threads!"
            "This limitation will be addressed in a future update to blocked LFC");
    }


    lfc->block_data.block_start_bc = gpu::Tmalloc<GpuVec3<grid_index_t>>(num_blocks);
    lfc->block_data.start_j = gpu::Tmalloc<size_t>(cpu_blocks.block_start_j.size());
    lfc->block_data.all_blocks_phi_idx_j = gpu::Tmalloc<size_t>(cpu_blocks.all_blocks_phi_idx_j.size());

    // Scary reinterpret_cast needed if we insist on avoiding extra copying
    static_assert(sizeof(GpuVec3<grid_index_t>) == sizeof(std::array<grid_index_t, 3>));
    static_assert(alignof(GpuVec3<grid_index_t>) == alignof(std::array<grid_index_t, 3>));

    gpu::Tmemcpy(lfc->block_data.block_start_bc,
        reinterpret_cast<const GpuVec3<grid_index_t>*>(cpu_blocks.block_start_bc.data()),
        cpu_blocks.block_start_bc.size(), gpuMemcpyHostToDevice);

    gpu::Tmemcpy(lfc->block_data.start_j, cpu_blocks.block_start_j.data(), cpu_blocks.block_start_j.size(), gpuMemcpyHostToDevice);
    gpu::Tmemcpy(lfc->block_data.all_blocks_phi_idx_j, cpu_blocks.all_blocks_phi_idx_j.data(), cpu_blocks.all_blocks_phi_idx_j.size(), gpuMemcpyHostToDevice);


    // ##### Phi ######
    const PhiData phi_data = std::move(cpu_builder.phi_data);
    const size_t num_phi = phi_data.l_i.size();
    assert(num_phi > 0);

    lfc->phi_data.num_phi = num_phi;
    lfc->phi_data.l_i = gpu::Tmalloc<uint8_t>(num_phi);
    lfc->phi_data.position_iv = gpu::Tmalloc<GpuVec3<double>>(num_phi);
    lfc->phi_data.first_mu_i = gpu::Tmalloc<size_t>(num_phi);
    lfc->phi_data.spline_idx_i = gpu::Tmalloc<size_t>(num_phi);

    gpu::Tmemcpy(lfc->phi_data.l_i, phi_data.l_i.data(), num_phi, gpuMemcpyHostToDevice);
    gpu::Tmemcpy(lfc->phi_data.first_mu_i, phi_data.first_mu_i.data(), num_phi, gpuMemcpyHostToDevice);

    static_assert(sizeof(GpuVec3<double>) == sizeof(Position));
    static_assert(alignof(GpuVec3<double>) == alignof(Position));

    gpu::Tmemcpy(lfc->phi_data.position_iv,
        reinterpret_cast<const GpuVec3<double>*>(phi_data.position_iv.data()),
        num_phi, gpuMemcpyHostToDevice);

    gpu::Tmemcpy(lfc->phi_data.spline_idx_i, phi_data.spline_index_i.data(), num_phi, gpuMemcpyHostToDevice);

    return out_lfc;
}
