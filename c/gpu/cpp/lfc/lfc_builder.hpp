#pragma once

#include "lfc_base.hpp"
#include <memory>

namespace gpaw
{

template<typename T, bool use_gpu>
std::unique_ptr<LFCBase> build_impl(GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data);


class BasisFunctionCollectionBuilder
{
public:

    BasisFunctionCollection build();
    void enable_single_precision(bool enabled) { single_precision = enabled; }
    void enable_gpu(bool enabled) { use_gpu = enabled; }

    // Optional pre-allocs. But this resets the builder's state!
    void reserve(size_t num_phi, size_t num_blocks);

    void add_phi(int32_t l, const Position& position_v, size_t first_mu);
    void add_block(const std::array<grid_index_t, 3>& block_start_c, const std::array<grid_index_t, 3>& block_shape, const std::vector<size_t>& M_m);

private:

    GridInfo grid;
    std::vector<GridBlock> blocks;
    PhiData phi_data;
    bool single_precision = false;
    bool use_gpu = false;
};

}
