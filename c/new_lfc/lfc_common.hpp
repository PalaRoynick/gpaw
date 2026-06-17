#pragma once

#include <array>
#include <vector>
#include <cstdint>


using Position = std::array<double, 3>;
// Describes spacings on a generic parallelepipid grid. 3x3 matrix
using GridSpacing = std::array<std::array<double, 3>, 3>;
using grid_index_t = int32_t;

struct GridInfo
{
    std::array<grid_index_t, 3> start_c;
    std::array<grid_index_t, 3> end_c;
    GridSpacing spacing_cv;
    // Volume element
    double dV;
};

// CPU side staging struct. TODO might as well turn this into SoA as well?
// struct GridBlock
// {
//     // Start indices to the full (x,y,z) grid for this block. Inclusive.
//     std::array<grid_index_t, 3> start_c;
//     // Shape of the grid block
//     std::array<grid_index_t, 3> shape;
//     // Map block-local basis function index m to global mu (redundant! see below)
//     std::vector<size_t> M_m;
//     // Map block-local m to global phi-index. Does not include the 2*l+1 internal indices of phi
//     std::vector<size_t> phi_idx_j;
// };

/* SoA style data for grid blocks. */
struct BlockData
{
    size_t get_num_blocks() const { return block_start_j.size() - 1; }

    std::array<grid_index_t, 3> block_shape;

    // Start grid indices of each block
    std::vector<std::array<grid_index_t, 3>> block_start_bc;

    /* Index buffers for implementing CSR (Compressed Sparse Row) style index lookup.
    *   - `all_blocks_phi_idx_j` maps block-local phi indices j to the global phi indices i.
    *     This is over all blocks => need a separate lookup array for giving (start_j, end_j) indices for a given block.
    *   - `block_start_j[b]` gives the first index into `all_blocks_phi_idx_j` for block `b`.
    *     `block_start_j[b+1]` gives the last index for that block.
    */
    std::vector<size_t> all_blocks_phi_idx_j;

    // Size is (num_blocks + 1). One extra element for holding the end index for the last block
    std::vector<size_t> block_start_j;

    // Original implementation had also similar lookups for block-local m to global mu but at least for now, we don't need them.
};


/* SoA style data for basis function instances ("spheres") in the system.
* We have N such phi instances phi_i, each having a quantum number l and a spline reference.
* Each phi_i spans (2*l + 1) global mu indices, and we provide a lookup list `first_mu_i` that gives the first mu for given phi_i.
*/
struct PhiData
{
    // Real-space positions of the phi
    std::vector<Position> position_iv;

    // Angular momentum number for each phi
    std::vector<uint8_t> l_i;

    /* Gives the first global mu index for given phi_i. That phi will span [first_mu, first_mu + 2*l + 1) mu indices.
    */
    std::vector<size_t> first_mu_i;

    // Splines are stored separately. This gives the spline index for given phi_i
    std::vector<size_t> spline_index_i;

    /* Coordinates of the cell containing the center of this phi. 3D array of ints.
    * These are relative to the main unit cell which has coords (0, 0, 0). */
    std::vector<std::array<grid_index_t, 3>> cell_coords_i;

    size_t num() const { return position_iv.size(); }
};
