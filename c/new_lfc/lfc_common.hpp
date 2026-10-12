#pragma once

#include <array>
#include <vector>
#include <cstdint>


/** Used to index grid coordinates along a single Cartesian grid axis, eg:
 *      arr[x][y][z]     where x,y,z are grid_coord_t valued.
 * Not guaranteed to be large enough to index flattened grid arrays; for that you have to use int64_t or similar.
 */
using grid_coord_t = int32_t;

using Position = std::array<double, 3>;
// Describes spacings on a generic parallelepipid grid. 3x3 matrix
using GridSpacing = std::array<std::array<double, 3>, 3>;

struct GridInfo
{
    std::array<grid_coord_t, 3> start_c;
    std::array<grid_coord_t, 3> end_c;
    GridSpacing spacing_cv;
    // Volume element
    double dV;
};

/* SoA style data for grid blocks. */
struct BlockData
{
    int64_t get_num_blocks() const
    {
        if (block_start_j.size() > 1)
        {
            return static_cast<int64_t>(block_start_j.size() - 1);
        }
        return 0;
    }

    std::array<grid_coord_t, 3> block_shape;

    // Start grid indices of each block
    std::vector<std::array<grid_coord_t, 3>> block_start_bc;

    /* Index buffers for implementing CSR (Compressed Sparse Row) style index lookup.
    *   - `all_blocks_phi_idx_j` maps block-local phi indices j to the global phi indices i.
    *     This is over all blocks => need a separate lookup array for giving (start_j, end_j) indices for a given block.
    *   - `block_start_j[b]` gives the first index into `all_blocks_phi_idx_j` for block `b`.
    *     `block_start_j[b+1]` gives the last index for that block.
    */
    std::vector<int64_t> all_blocks_phi_idx_j;

    // Size is (num_blocks + 1). One extra element for holding the end index for the last block
    std::vector<int64_t> block_start_j;

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
    std::vector<int64_t> first_mu_i;

    // Splines are stored separately. This gives the spline index for given phi_i
    std::vector<int64_t> spline_index_i;

    /* Coordinates of the cell containing the center of this phi. 3D array of ints.
    * These are relative to the main unit cell which has coords (0, 0, 0). */
    std::vector<std::array<grid_coord_t, 3>> cell_coords_i;

    size_t num() const { return position_iv.size(); }
};
