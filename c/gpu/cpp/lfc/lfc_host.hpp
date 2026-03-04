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

template<typename T>
class LFC_Impl
{
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

public:
    static constexpr bool is_single_precision() { return std::is_same_v<T, float>; }

    // TODO fix dtype
    void add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const;

private:

    std::vector<GridBlock> blocks;
    PhiData phi_data;

    //std::unique_ptr<LFCGpu<T>> upload_to_gpu();

    //std::unique_ptr<LFCGpu<T>> gpu_impl;

    friend class BasisFunctionCollectionBuilder;
};

/* Python chooses dtype => We need at least one layer of type erasure in the main LFC class in order to support both 32- and 64-bit versions.
Options:
    1. Easiest is to template the whole thing and separately expose float/double specializations to Python, and let Python
       choose which object to create. This doesn't extend very well though and leads to a lot of noise on the Python side, and
       is also annoying to work with because the whole thing is templated.
    2. Non-template LFC with std::variant holding the actual implementation. Downside: horrible syntax.
    3. Template implementation hidden behind a layer of runtime polymorphism. Gets quite convoluted
    4. Simpler version of 3: LFC has unique_ptr to both implementations, only one is really created. Then branch to correct implementation
       whenever implementation details are important.
    5. Void pointers everywhere, cast when passing to functions. Might be kinda OK in the GPU layer but that is created later.

Going with 4 for now, it's certainly primitive but who cares.
*/

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

private:

    // Construction through the builder class only
    BasisFunctionCollection(bool use_single_precision)
        : single_precision(use_single_precision) {}

    friend class BasisFunctionCollectionBuilder;

    // Only one of these gets allocated:
    std::unique_ptr<LFC_Impl<float>> impl32;
    std::unique_ptr<LFC_Impl<double>> impl64;

    const bool single_precision;
};

class BasisFunctionCollectionBuilder
{
public:

    BasisFunctionCollection build();
    void enable_single_precision(bool enabled) { single_precision = enabled; }

    // Optional pre-allocs. But this resets the builder's state!
    void reserve(size_t num_phi, size_t num_blocks);

    void add_phi(int32_t l, const Position& position_v, size_t first_mu);
    void add_block(const std::array<grid_index_t, 3>& block_start_c, const std::array<grid_index_t, 3>& block_shape, const std::vector<size_t>& M_m);

private:

    template<typename T>
    std::unique_ptr<LFC_Impl<T>> build_impl();

    GridInfo grid;
    std::vector<GridBlock> blocks;
    PhiData phi_data;
    bool single_precision = false;
};

template<typename T>
inline std::unique_ptr<LFC_Impl<T>> BasisFunctionCollectionBuilder::build_impl()
{
    std::unique_ptr<LFC_Impl<T>> lfc = std::make_unique<LFC_Impl<T>>();
    assert(lfc != nullptr);

    lfc->blocks = std::move(blocks);
    lfc->phi_data = std::move(phi_data);

    //lfc->gpu_impl = std::move(lfc->upload_to_gpu());

    return std::move(lfc);
}

template<typename T>
inline void LFC_Impl<T>::add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const
{
}

// template<typename T>
// inline std::unique_ptr<LFCGpu<T>> LFC_Impl<T>::upload_to_gpu()
// {
//     std::make_unique<LFCGpu<T>>();
//     return std::unique_ptr<LFCGpu<T>>();
// }

} // namespace gpaw
