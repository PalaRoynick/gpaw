#pragma once

#include "lfc_common.hpp"
#include "splines/radial_spline.hpp"
#include "utils.hpp"
#include <memory>
#include <array>
#include <vector>
#include <cstdint>

struct SplineData
{
    double cutoff;
    std::vector<double> values;
};


class SplinePool
{
    using Spline = gpaw::TRadialSpline<double>;
public:
    SplinePool(std::vector<SplineData> spline_datas);
    size_t num_splines() const { return splines.size(); }
    const std::vector<Spline>& get_splines() const { return splines; }
    const Spline& get_spline(size_t i) const { return splines[i]; }

private:
    std::vector<Spline> splines;
};

/* Staging struct for block data. Easy to work with but not efficient for actual computations. */
struct BlockDataStaging
{
    int64_t num_blocks() const { return static_cast<int64_t>(block_start_bc.size()); }
    std::array<grid_coord_t, 3> block_shape;
    std::vector<std::array<grid_coord_t, 3>> block_start_bc;
    // Maps block-local phi_j to global phi_i index
    std::vector<std::vector<int64_t>> block_phi_j_to_phi_i;
};

/* Python sees this. Currently this is responsible also for building spline pools */
class LFCBuilder
{
    // Always build double-precision splines
    using SplineDtype = double;
    using CpuSpline = gpaw::TRadialSpline<SplineDtype>;

public:
    void set_block_size(const std::array<grid_coord_t, 3>& in_block_size);

    // Volume element
    double dV;

    void add_spline(double cutoff, gpaw::Span<const double> data);
    std::shared_ptr<SplinePool> build_spline_pool();

    // start_c and end_c are the MPI-aware domain start and end ranges
    void define_grid(const std::array<grid_coord_t, 3>& start_c, const std::array<grid_coord_t, 3>& end_c, const std::array<std::array<double, 3>, 3>& spacing_cv, double dV);
    void add_phi(int32_t l, int64_t spline_index, const Position& position_v, int64_t first_mu, const std::array<grid_coord_t, 3>& cell_coords);
    /* Add a new grid block. Shape is implied to be the same as that defined by set_block_size()
    * `phi_idx_j` should map block-local phi indices to the global phi index for all phi_j in that block. */
    void add_block(const std::array<grid_coord_t, 3>& block_start_c, gpaw::Span<const int64_t> phi_idx_j);

    // Turns staged block data into a format that is more suitable for computations.
    static BlockData build_flattened_block_data(BlockDataStaging&& staging_data);

    // Does some moderate checking on the inputs before build, throws if something is off
    void prebuild_validate();

    // These are public because the GPU side builder needs to access/steal them. Not ideal but works

    GridInfo grid;
    PhiData phi_data;
    BlockDataStaging block_staging;
    std::array<grid_coord_t, 3> block_size = {};

private:
    // Staging data for splines
    std::vector<SplineData> spline_datas;
};
