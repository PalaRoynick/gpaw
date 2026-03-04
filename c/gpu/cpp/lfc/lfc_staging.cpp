#include "lfc_staging.hpp"

namespace gpaw
{

void BasisFunctionCollectionBuilder::add_block(const std::array<grid_index_t, 3>& block_start_c,
    const std::array<grid_index_t, 3>& block_shape, const std::vector<size_t>& M_m)
{
    GridBlock block;
    block.start_c = block_start_c;
    block.shape = block_shape;
    block.M_m = M_m;

    blocks.push_back(block);
}

} // namespace gpaw
