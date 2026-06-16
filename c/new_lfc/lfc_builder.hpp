#pragma once

#include "lfc_common.hpp"
#include "splines/radial_spline.hpp"
#include <memory>
#include <array>
#include <vector>
#include <span>
#include <cstdint>

class LFCGPU;

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

/* Python sees this. Currently this builds splines pools too because I'm too lazy to separate and expose that to Python */
class LFCBuilder
{
    // LFCGpu has block size as template parameter (optimization) => We only allow certain values of block size
    //static constexpr std::array<int32_t, 2> gpu_allowed_block_sizes = { -1, 8 };

    // Always build double-precision splines
    using SplineDtype = double;
    using CpuSpline = gpaw::TRadialSpline<SplineDtype>;

public:

    //std::unique_ptr<LFCGPU> build_gpu();
    //void enable_single_precision(bool enabled) { single_precision = enabled; }
    //void enable_gpu(bool enabled) { use_gpu = enabled; }
    void set_block_size(const std::array<int32_t, 3>& in_block_size) { block_size = in_block_size; }

    // Optional pre-allocs. But this resets the builder's state!
    //void reserve(size_t num_phi, size_t num_blocks);

    std::array<size_t, 3> start_c;
    std::array<size_t, 3> end_c;
    // Volume element
    double dV;

    void add_spline(double cutoff, std::span<const double> data);
    std::shared_ptr<SplinePool> build_spline_pool();

    //void set_spline_pool(std::shared_ptr<const SplinePoolImplBase> spline_pool_impl);
    // MPI aware grid bounds
    // void define_grid(const std::array<size_t, 3>& start_c, const std::array<size_t, 3>& end_c, const std::array<double, 3>& spacing, double dV);

    void add_phi(int32_t l, size_t spline_index, const Position& position_v, size_t first_mu, const std::array<grid_index_t, 3>& cell_coords);

    //void add_block(const std::array<grid_index_t, 3>& block_start_c, const std::array<grid_index_t, 3>& block_shape, const std::vector<size_t>& M_m, const std::vector<size_t>& phi_idx_j);

    // constexpr bool is_allowed_block_size(int32_t block)
    // {
    //     return std::find(gpu_allowed_block_sizes.begin(), gpu_allowed_block_sizes.end(), block) != gpu_allowed_block_sizes.end();
    // }

private:

    // Does some moderate checking on the inputs before build, throws if something is off
    void prebuild_validate();

    // GridInfo grid;
    // std::vector<GridBlock> blocks;
    PhiData phi_data;
    //std::shared_ptr<const SplinePoolImplBase> spline_pool;
    bool single_precision = false;
    bool use_gpu = false;
    std::array<int32_t, 3> block_size = {};

    // Staging data for splines
    std::vector<SplineData> spline_datas;
};
