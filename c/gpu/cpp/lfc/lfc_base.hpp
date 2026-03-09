#pragma once

#include "gpu/cpp/ndarray.hpp"

#include <array>
#include <vector>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <cassert>


/* Let's call i = global basis function index. So this runs over all phi in the system, including periodic copies.
*/

namespace gpaw
{

static constexpr int32_t ALLOWED_BLOCK_SIZE = 8;

class SplinePool;
template<typename T> class TRadialSpline;

using Position = std::array<double, 3>;
using grid_index_t = int32_t;

struct GridInfo
{
    std::array<size_t, 3> start_c;
    std::array<size_t, 3> end_c;
    // Grid spacing in XYZ dirs
    std::array<double, 3> spacing;
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
    // Map block-local basis function index m to global mu (redundant! see below)
    std::vector<size_t> M_m;
    // Map block-local m to global phi-index. Does not include the 2*l+1 internal indices of phi
    std::vector<size_t> phi_idx_j;
};


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

    std::vector<size_t> spline_index_i;

    size_t num() const { return position_iv.size(); }
};

class LFCBase
{
public:
    virtual ~LFCBase() = default;
    virtual void add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const = 0;
    virtual void calculate_potential_matrix(const NDArray<c_contig, ndim<3>>& vt_G, NDArray<c_contig, ndim<2>>& out_V_MN) const = 0;
};


class BasisFunctionCollection
{

public:

    BasisFunctionCollection(const BasisFunctionCollection&) = delete;
    BasisFunctionCollection& operator=(const BasisFunctionCollection&) = delete;

    BasisFunctionCollection(BasisFunctionCollection&&) noexcept = default;
    BasisFunctionCollection& operator=(BasisFunctionCollection&&) noexcept = default;

    inline bool is_single_precision() const { return single_precision; }

    // TODO NDArray should have some "cast" routine that can add traits.
    // Would be useful here because in this layer we can't fix the dtype, but in the impl layer it should be fixed.

    void add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const;
    void calculate_potential_matrix(const NDArray<c_contig, ndim<3>>& vt_G, NDArray<c_contig, ndim<2>>& out_V_MN) const;

private:

    // Construction through the builder class only
    BasisFunctionCollection(bool use_single_precision)
        : single_precision(use_single_precision) {}

    friend class BasisFunctionCollectionBuilder;

    // Implementation should have shared ownership of appropriate SplinePoolImplBase object
    std::unique_ptr<LFCBase> impl;
    bool single_precision;
};

} // namespace gpaw
