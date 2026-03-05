#pragma once

#include "lfc_base.hpp"
#include <memory>
#include <stdexcept>


namespace gpaw
{

/* Same pattern as with BasisFunctionCollection: Python sees the type-erased generic builder,
and internally we have templated versions for doing the actual work.
For GPU building there's one extra non-template layer so that we don't have to depend on CUDA code here */


class BasisFunctionCollectionBuilder
{
public:

    BasisFunctionCollection build();
    void enable_single_precision(bool enabled) { single_precision = enabled; }
    void enable_gpu(bool enabled) { use_gpu = enabled; }

    // Optional pre-allocs. But this resets the builder's state!
    void reserve(size_t num_phi, size_t num_blocks);

    std::array<size_t, 3> start_c;
    std::array<size_t, 3> end_c;
    // Volume element
    double dV;

    // MPI aware grid bounds
    void define_grid(const std::array<size_t, 3>& start_c, const std::array<size_t, 3>& end_c, const std::array<double, 3>& spacing, double dV);
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
