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

void LFCBuilder::prebuild_validate()
{
}
