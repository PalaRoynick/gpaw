#include "lfc_builder.hpp"
#include "splines/radial_spline.hpp"

SplinePool::SplinePool(std::vector<SplineData> spline_datas)
{
    if (spline_datas.size() == 0)
    {
        throw std::runtime_error("Empty spline_datas");
    }

    splines.reserve(spline_datas.size());
    for (const SplineData& data : spline_datas)
    {
        gpaw::TRadialSpline<double> new_spline(data.cutoff, data.values.data(), data.values.size());
        splines.push_back(new_spline);
    }
}

void LFCBuilder::set_block_size(const std::array<int32_t, 3>& in_block_size)
{
    block_staging.block_shape = in_block_size;
}

void LFCBuilder::add_spline(double cutoff, std::span<const double> data)
{
    SplineData new_spline;
    new_spline.cutoff = cutoff;
    new_spline.values.reserve(data.size());

    for (double v : data)
    {
        new_spline.values.push_back(v);
    }

    spline_datas.push_back(new_spline);
}

std::shared_ptr<SplinePool> LFCBuilder::build_spline_pool()
{
    return std::make_shared<SplinePool>(std::move(spline_datas));
}

void LFCBuilder::define_grid(const std::array<grid_index_t, 3>& start_c, const std::array<grid_index_t, 3>& end_c, const std::array<std::array<double, 3>, 3>& spacing_cv, double dV)
{
    grid.start_c = start_c;
    grid.end_c = end_c;
    grid.spacing_cv = spacing_cv;
    grid.dV = dV;
}

void LFCBuilder::add_phi(int32_t l, size_t spline_index, const Position& position_v, size_t first_mu, const std::array<grid_index_t, 3>& cell_coords)
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
    phi_data.cell_coords_i.push_back(cell_coords);
}

void LFCBuilder::add_block(const std::array<grid_index_t, 3>& block_start_c, std::span<const size_t> phi_idx_j)
{
    block_staging.block_start_bc.push_back(block_start_c);

    block_staging.block_phi_j_to_phi_i.push_back(
        std::vector<size_t>(phi_idx_j.begin(), phi_idx_j.end()));
}

BlockData LFCBuilder::build_flattened_block_data(BlockDataStaging&& staging_data)
{
    BlockData out;
    const size_t num_blocks = staging_data.num_blocks();

    out.block_shape = std::move(staging_data.block_shape);
    out.block_start_bc = std::move(staging_data.block_start_bc);

    // Build CSR lookups
    out.block_start_j.resize(num_blocks + 1);
    out.all_blocks_phi_idx_j.reserve(num_blocks); // will need at least this many
    size_t j = 0;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const std::vector<size_t>& phi_j_to_global = staging_data.block_phi_j_to_phi_i[b];
        out.block_start_j[b] = j;

        for (size_t global_phi_idx : phi_j_to_global)
        {
            out.all_blocks_phi_idx_j.push_back(global_phi_idx);
        }
        j += phi_j_to_global.size();
    }
    // End CSR index for the last block
    out.block_start_j[num_blocks] = j;

    return out;
}

void LFCBuilder::prebuild_validate()
{
    // TODO ensure phi indices in block data are valid
}
