#include "lfc_gpu_builder.hpp"
#include "lfc_gpu.cuh"
#include <vector>
#include <array>

namespace gpaw
{

struct BlockDataStaging
{
    std::vector<long3> block_start_c;
    std::vector<long3> block_shape;
    std::vector<size_t> start_m;
    std::vector<size_t> all_blocks_M_m;
};

template<typename T>
std::unique_ptr<LFCBase> build_impl_gpu(GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data)
{
    auto out_lfc = std::make_unique<LFCGpu<T>>();
    assert(out_lfc);

    // Lifehack to help intellisense, tends to have trouble with smart pointers of template classes
    LFCGpu<T>* lfc = out_lfc.get();

    // ###### Grid ######
    lfc->grid.dV = grid.dV;
    lfc->grid.spacing = make_double3(grid.spacing[0], grid.spacing[1], grid.spacing[2]);
    lfc->grid.start_c = make_long3(grid.start_c[0], grid.start_c[1], grid.start_c[2]);
    lfc->grid.end_c = make_long3(grid.end_c[0], grid.end_c[1], grid.end_c[2]);

    // ###### Blocks ######

    assert(!blocks.empty());
    // Count how many mu indices we need in total
    size_t num_mu = 0;
    for (const GridBlock& block : blocks)
    {
        num_mu += block.M_m.size();
    }
    assert(num_mu > 0);

    const size_t num_blocks = blocks.size();

    // Flatten the host-side block data to format like what the GPU wants
    BlockDataStaging staging_blocks;
    staging_blocks.block_shape.resize(num_blocks);
    staging_blocks.block_start_c.resize(num_blocks);
    staging_blocks.start_m.resize(num_blocks + 1);
    staging_blocks.all_blocks_M_m.resize(num_mu);

    size_t mu = 0;
    for (size_t b = 0; b < blocks.size(); ++b)
    {
        const GridBlock& block = blocks[b];
        staging_blocks.block_shape[b] = { block.shape[0], block.shape[1], block.shape[2] };
        staging_blocks.block_start_c[b] = { block.start_c[0], block.start_c[1], block.start_c[2] };

        // flattened index lookups
        for (size_t m = 0; block.M_m.size(); ++m)
        {
            staging_blocks.all_blocks_M_m[mu + m] = block.M_m[m];
        }

        staging_blocks.start_m[b] = mu;
    }

    staging_blocks.start_m[blocks.size()] = mu + 1;
    assert(mu == num_mu);

    // Copy block data to GPU
    lfc->block_data.num_blocks = num_blocks;

    lfc->block_data.block_shape = Tgpu_malloc<long3>(blocks.size());
    lfc->block_data.block_start_c = Tgpu_malloc<long3>(blocks.size());
    lfc->block_data.start_m = Tgpu_malloc<size_t>(blocks.size() + 1);
    lfc->block_data.all_blocks_M_m = Tgpu_malloc<size_t>(num_mu);

    Tgpu_memcpy(lfc->block_data.block_shape, staging_blocks.block_shape.data(), staging_blocks.block_shape.size(), gpuMemcpyHostToDevice);
    Tgpu_memcpy(lfc->block_data.block_start_c, staging_blocks.block_start_c.data(), staging_blocks.block_start_c.size(), gpuMemcpyHostToDevice);
    Tgpu_memcpy(lfc->block_data.start_m, staging_blocks.start_m.data(), staging_blocks.start_m.size(), gpuMemcpyHostToDevice);
    Tgpu_memcpy(lfc->block_data.all_blocks_M_m, staging_blocks.all_blocks_M_m.data(), staging_blocks.all_blocks_M_m.size(), gpuMemcpyHostToDevice);

    // ##### Phi ######
    const size_t num_phi = phi_data.l_i.size();
    assert(num_phi > 0);

    lfc->phi_data.num_phi = num_phi;
    lfc->phi_data.l_i = Tgpu_malloc<uint8_t>(num_phi);
    lfc->phi_data.position_iv = Tgpu_malloc<GpuVec3<T>>(num_phi);
    lfc->phi_data.first_mu_i = Tgpu_malloc<size_t>(num_phi);

    Tgpu_memcpy(lfc->phi_data.l_i, phi_data.l_i.data(), num_phi, gpuMemcpyHostToDevice);
    Tgpu_memcpy(lfc->phi_data.first_mu_i, phi_data.first_mu_i.data(), num_phi, gpuMemcpyHostToDevice);
    Tgpu_memcpy(lfc->phi_data.position_iv, reinterpret_cast<GpuVec3<T>*>(phi_data.position_iv.data()), num_phi, gpuMemcpyHostToDevice);

    return out_lfc;
}

std::unique_ptr<LFCBase> build_impl_gpu_entry(bool single_precision, GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data)
{
    if (single_precision)
    {
        return build_impl_gpu<float>(std::move(grid), std::move(blocks), std::move(phi_data));
    }
    else
    {
        return build_impl_gpu<double>(std::move(grid), std::move(blocks), std::move(phi_data));
    }
}

} // namespace gpaw
