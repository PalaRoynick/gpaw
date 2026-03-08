#include "lfc_builder.hpp"
#include "lfc_cpu.hpp"
#include "lfc_gpu_builder.hpp"
#include "gpu/cpp/splines/spline_pool.hpp"

namespace gpaw
{

void BasisFunctionCollectionBuilder::reserve(size_t num_phi, size_t num_blocks)
{
    blocks.clear();
    blocks.reserve(num_blocks);

    phi_data.position_iv.clear();
    phi_data.l_i.clear();
    phi_data.first_mu_i.clear();

    phi_data.position_iv.reserve(num_phi);
    phi_data.l_i.reserve(num_phi);
    phi_data.position_iv.reserve(num_phi);
}

void BasisFunctionCollectionBuilder::set_spline_pool(std::shared_ptr<const SplinePoolImplBase> spline_pool)
{
    this->spline_pool = spline_pool;
}

void BasisFunctionCollectionBuilder::define_grid(const std::array<size_t, 3>& start_c, const std::array<size_t, 3>& end_c, const std::array<double, 3>& spacing, double dV)
{
    grid.start_c = start_c;
    grid.end_c = end_c;
    grid.spacing = spacing;
    grid.dV = dV;
}

void BasisFunctionCollectionBuilder::add_phi(int32_t l, size_t spline_index, const Position& position_v, size_t first_mu)
{
    constexpr auto l_max = std::numeric_limits<uint8_t>::max();
    if (l < 0)
    {
        throw std::invalid_argument("l must be >= 0, got " + std::to_string(l));
    }
    if (l > l_max)
    {
        throw std::invalid_argument("l too large, max allowed is " + std::to_string(l_max));
    }

    phi_data.l_i.push_back(l);
    phi_data.position_iv.push_back(position_v);
    phi_data.first_mu_i.push_back(first_mu);
    phi_data.spline_index_i.push_back(spline_index);
}


void BasisFunctionCollectionBuilder::add_block(const std::array<grid_index_t, 3>& block_start_c,
    const std::array<grid_index_t, 3>& block_shape, const std::vector<size_t>& M_m, const std::vector<size_t>& phi_idx_j)
{
    GridBlock block;
    block.start_c = block_start_c;
    block.shape = block_shape;
    block.M_m = M_m;
    block.phi_idx_j = phi_idx_j;

    blocks.push_back(block);
}

void BasisFunctionCollectionBuilder::prebuild_validate()
{

    if (block_size != ALLOWED_BLOCK_SIZE)
    {
        throw std::runtime_error("WIP: Block size must currently be 8");
    }

    if (!spline_pool)
    {
        throw std::runtime_error("SplinePool is invalid! Did you forget set_spline_pool(...) ?");
    }

    for (size_t spline_idx : phi_data.spline_index_i)
    {
        if (spline_idx >= spline_pool->num_splines())
        {
            throw std::runtime_error("Invalid spline index " + std::to_string(spline_idx) \
                + ", max in spline pool is: " + std::to_string(spline_pool->num_splines()));
        }
    }
}

BasisFunctionCollection BasisFunctionCollectionBuilder::build()
{
    prebuild_validate();
    BasisFunctionCollection lfc(single_precision);

    if (!use_gpu)
    {
        if (single_precision)
        {
            lfc.impl = std::move(
                build_impl_cpu<float>(std::move(grid), std::move(blocks), std::move(phi_data))
            );
        }
        else
        {
            lfc.impl = std::move(
                build_impl_cpu<double>(std::move(grid), std::move(blocks), std::move(phi_data))
            );
        }
    }
#ifdef GPAW_GPU
    else
    {
        lfc.impl = std::move(
                build_impl_gpu_entry(single_precision, std::move(grid), std::move(blocks), std::move(phi_data), std::move(spline_pool))
        );
    }
#endif

    return lfc;
}

} // namespace gpaw
