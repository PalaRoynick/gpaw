#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <memory>
#include <type_traits>

/* Let's call i = global basis function index. So this runs over all phi in the system, including periodic copies.
*/

namespace gpaw
{

template<typename T> class TRadialSpline;

using Position = std::array<double, 3>;
using grid_index_t = int32_t;

struct GridInfo
{
    std::array<size_t, 3> start_c;
    std::array<size_t, 3> end_c;
    // Volume element
    double dV;
};

// CPU side staging struct. TODO might as well turn this into SoA as well?
struct GridBlock
{
    // Start indices to the full (x,y,z) grid for this block. Inclusive.
    std::array<grid_index_t, 3> start_c;
    // Shape of the grid block
    std::array<grid_index_t, 3> shape;
    // Map block-local basis function index m to global mu
    std::vector<size_t> M_m;
};


// GOOD, but move elsewhere:
// Contains info about all blocks in a SoA manner. So kernel uses block index to index these arrays (except for d_all_blocks_M_m)
// struct BlockDataGpu
// {
//     size_t num_blocks;
//     int3* d_block_start_c;
//     int3* d_block_shape;

//     /* CSR style index lookup. d_start_m[block_idx] gives the first index into d_all_blocks_M_m for that block
//     and d_start_m[block_idx+1] gives the end index (exclusive) */
//     int* d_start_m; // size (num_blocks+1): one extra element to give end m for the last block

//     /* Maps block local m to global mu index. But this is now over all blocks: use like
//     int block_idx = ...;
//     int m_offset = d_start_m[block_idx];
//     for (int m = 0; m < d_start_m[block_idx + 1]; m++)
//     {
//         int mu = d_all_blocks_M_m[m + m_offset];
//     }
//     */
//     int* d_all_blocks_M_m;

// };

// struct PhiInstance
// {
//     Position position_v;
//     uint8_t l;
//     size_t first_mu;
//     // TODO spline reference (shared ptr?)

// };

// Might just as well go with SoA directly on host side. Python already has intuitive AoS data structs.

/* SoA style data for basis function instances ("spheres") in the system.
* We have N such phi instances phi_i, each having a quantum number l and a spline reference.
* Each phi_i spans (2*l + 1) global mu indices, and we provide a lookup list `first_mu_i` that gives the first mu for given phi_i.
*/
struct PhiData
{
    //static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    // Real-space positions of the phi
    std::vector<Position> position_iv;

    // Angular momentum number for each phi
    std::vector<uint8_t> l_i;

    /* Gives the first global mu index for given phi_i. That phi will span [first_mu, first_mu + 2*l + 1) mu indices.
    */
    std::vector<size_t> first_mu_i;
    //std::vector<std::shared_ptr<TRadialSpline<T>>> spline_i;

    size_t num() const { return position_iv.size(); }
};


class BasisFunctionCollectionBuilder
{
public:

    void add_block(const std::array<grid_index_t, 3>& block_start_c, const std::array<grid_index_t, 3>& block_shape, const std::vector<size_t>& M_m);
    void enable_single_precision(bool enabled) { single_precision = enabled; }

private:

    GridInfo grid;
    std::vector<GridBlock> blocks;
    PhiData phi_data;
    bool single_precision = false;
};

template<typename T>
class BasisFunctionCollectionImpl
{
public:

    using Spline = TRadialSpline<T>;

    void add_to_density(...);
    //void calculate_potential_matrix(...);

    BasisFunctionCollection build();

private:

    std::vector<std::shared_ptr<Spline>> splines;
};


class BasisFunctionCollection
{
public:
    //BasisFunctionCollection(std::vector<GridBlock>)

private:

    // Construction through the builder class only
    BasisFunctionCollection() = default;
    friend class BasisFunctionCollectionBuilder;

    // Fancy type erasure pattern to combine templated implementation with runtime polymorphism
    struct Concept
    {
        virtual ~Concept() = default;
        //virtual void add_to_density(...) = 0;
    };

    template<typename T>
    struct Model : Concept
    {
        BasisFunctionCollectionImpl<T> impl;

        // void add_to_density(...) override
        // {
        //     impl.add_to_density(...);
        // }
    };

    std::unique_ptr<Concept> impl;

    const bool single_precision;
};

}
